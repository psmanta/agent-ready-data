#!/usr/bin/env python3
"""
Planted-data tests for evaluate_h2.py (stratified ladders and the pre-registered shape test) and validate_h2_schema.py.
No API calls. Each scenario writes a small synthetic experiments_output (12 H2 conditions plus the clean re-run control)
whose drift is planted exactly, per stratum and per customer half, so every number the evaluator reports has a known
right answer. Run from 1b_dithering/:

    python3 test_evaluate_h2.py
"""
import atexit, contextlib, copy, io, json, random, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_h2 as H2
import validate_h2_schema as V
import customer_halves as CH

WORK = Path(tempfile.mkdtemp(prefix="h2_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a, **k)

TIERS = ["stable"] * 400 + ["deeply_boundary"] * 60 + ["lightly_boundary"] * 50 + ["tied_no_majority"] * 6 + [None] * 4
IDS = [f"CUST_{i:06d}" for i in range(1, len(TIERS) + 1)]
TIER = dict(zip(IDS, TIERS))
STABLE = [c for c in IDS if TIER[c] == "stable"]
BOUNDARY = [c for c in IDS if TIER[c] in ("deeply_boundary", "lightly_boundary", "tied_no_majority")]
UNCHANGED = {f: set(STABLE[i::40]) for i, f in enumerate(H2.FIELDS)}   # 10 per field, never changed at one rung
POP = {f: [c for c in STABLE if c not in UNCHANGED[f]] for f in H2.FIELDS}
CONTROL_STABLE, BOUNDARY_RATE = 0.03, 0.45


def pick(ids, rate, salt):
    """Exactly round(rate * n) drifters inside EACH customer half, so halves carry the planted rate too."""
    out = set()
    for h, hids in CH.split(ids).items():
        r = rate[h] if isinstance(rate, dict) else rate
        k = round(r * len(hids))
        out |= set(random.Random(f"{salt}|{h}").sample(sorted(hids), k))
    return out


def build(name, curves):
    """curves[field] = 4 net offsets over the control (stable, on the shape population); a value may be {0: x, 1: y} per half."""
    eo = WORK / name
    gt = {c: {"final_decision": "MEDIUM_PRIORITY", "stability_tier": TIER[c], "decision_source": "primary_vote"} for c in IDS}
    (eo).mkdir(parents=True)
    (eo / "gt.json").write_text(json.dumps(gt))
    ctrl_stable = pick(STABLE, CONTROL_STABLE, "ctrl")
    def write(cid, drifters, changed_field=None, unchanged=()):
        d = eo / "conditions" / cid; d.mkdir(parents=True)
        ref, dec = [], []
        for i, c in enumerate(IDS):
            rid = f"COND_{i:06d}"
            fields = [] if changed_field is None or c in unchanged else [changed_field]
            ref.append({"record_id": rid, "customer_id": c, "customer_segment": ["high_value", "medium_value", "low_value"][i % 3],
                        "_dither_applied": bool(fields), "_dither_fields": fields, "_dither_original": {f: 1 for f in fields},
                        **({changed_field: 2} if fields else {})})
            dec.append({"record_id": rid, "business_decision": "HIGH_PRIORITY" if c in drifters else "MEDIUM_PRIORITY",
                        "agent_confidence": 0.8, "decision_reasoning": "planted", "key_factors": []})
        (d / "dither_reference.json").write_text(json.dumps(ref))
        (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in dec))
        return d
    cd = write(H2.CONTROL_CONDITION, ctrl_stable | pick(BOUNDARY, BOUNDARY_RATE, "ctrl-b"))
    (cd / "control.json").write_text(json.dumps({"control": True, "kind": "clean_rerun"}))
    pop_ctrl = {f: [c for c in POP[f]] for f in H2.FIELDS}
    for f in H2.FIELDS:
        for j, m in enumerate(H2.MAGNITUDE_LEVELS):
            off = curves[f][j]
            # planted on the shape population: drift = control drifters' rate + offset, realised as an exact count per half
            base = {h: sum(c in ctrl_stable for c in hids) / len(hids) for h, hids in CH.split(POP[f]).items()}
            rate = {h: base[h] + (off[h] if isinstance(off, dict) else off) for h in (0, 1)}
            drifters = pick(POP[f], rate, f"{f}|{m}") | pick(BOUNDARY, BOUNDARY_RATE, f"{f}|{m}|b")
            # the never-changed customers are unchanged at the 40% rung only (so they leave the shape population)
            write(H2.condition_id_for(f, m), drifters, changed_field=f, unchanged=UNCHANGED[f] if m == 40 else ())
    return eo


def run(eo):
    return quiet(H2.evaluate_h2, eo, eo / "gt.json")


CLIFF = [0.14, 0.16, 0.18, 0.20]
SMOOTH = [0.015, 0.04, 0.095, 0.20]
DECOR = [0.0, 0.02, 0.01, 0.02]

print("S1  strata: stable is clean of boundary noise; boundary and all are reported, never dropped")
A = build("a", {"churn_risk_score": CLIFF, "total_spend": CLIFF, "tenure_months": DECOR})
ra = run(A)
assert ra["stratum_sizes"] == {"stable": 400, "boundary": 116, "all": 520}; ok("stratum sizes: stable 400; boundary 116 (lightly + deeply + 6 tied); all 520 (including 4 customers with no tier)")
st = ra["fields"]["churn_risk_score"]["stable"]; al = ra["fields"]["churn_risk_score"]["all"]; bd = ra["fields"]["churn_risk_score"]["boundary"]
assert abs(st["net_of_control"]["control_drift_rate"] - CONTROL_STABLE) < 0.006 and abs(bd["net_of_control"]["control_drift_rate"] - BOUNDARY_RATE) < 0.01
ok("the control's drift is computed per stratum: about 3% for stable, about 45% for boundary")
assert al["drift_rates_by_magnitude"]["5"] > st["drift_rates_by_magnitude"]["5"] + 0.03; ok("mixing boundary customers in ('all') inflates the absolute drift rate, which is why stable is primary")
assert "within_segment_breakdown" in ra["fields"]["churn_risk_score"]["stable"] and "within_segment_breakdown" in ra["fields"]["churn_risk_score"]["all"] \
    and "within_segment_breakdown" not in ra["fields"]["churn_risk_score"]["boundary"] and "within_segment_breakdown" not in ra["fields"]["tenure_months"]["stable"]
ok("within-segment breakdown: churn and spend only, on the stable and all strata")

print("S2  the shape test: two cliffs and a decorative field")
t = ra["shape_test"]; pf = t["per_field"]
assert pf["churn_risk_score"]["n_customers"] == len(POP["churn_risk_score"]) == 390; ok("population: the 390 stable customers changed at every rung (10 unchanged at one rung are excluded)")
for f in ("churn_risk_score", "total_spend"):
    assert abs(pf[f]["s"] - 0.14) < 0.006 and abs(pf[f]["F"] - 0.20) < 0.006 and pf[f]["F_rung_pct"] == 100 and pf[f]["classification"] == "cliff"
ok("churn and spend: s = 14 points, F = 20 points (at 100%), s/F = 0.7, significant after Holm: cliff")
assert pf["tenure_months"]["classification"] == "below_floor" and pf["tenure_months"]["non_monotone"] is True; ok("tenure: F = 2 points is below M = 10: no shape claim; its zig-zag ladder is flagged non-monotone")
assert t["cliff_fields"] == ["churn_risk_score", "total_spend"] and t["qualified"] is True and abs(t["candidate_effect_median_s"] - 0.14) < 0.006 and t["effect_at_least_M"]
ok("two cliffs, replicated in both halves: the cliff candidate qualifies, with effect = median s = 14 points (at least M)")
lo, hi = pf["churn_risk_score"]["s_over_F_ci95_bootstrap"]; assert lo < 0.7 < hi and hi - lo < 0.6; ok(f"s/F carries a customer-level bootstrap interval around 0.7 ({lo} to {hi})")
assert run(A)["shape_test"] == t; ok("the same data gives the same result, interval included (fixed bootstrap seed)")

print("S3  smooth, mixed, the floor and its sensitivity")
B = build("b", {"churn_risk_score": SMOOTH, "total_spend": [0.05, 0.09, 0.12, 0.14], "tenure_months": [0.03, 0.05, 0.06, 0.07]})
tb = run(B)["shape_test"]; pb = tb["per_field"]
assert pb["churn_risk_score"]["classification"] == "smooth"; ok("churn 1.5 / 4 / 9.5 / 20 points: s/F = 0.075: smooth")
assert pb["total_spend"]["classification"] == "mixed"; ok("spend 5 / 9 / 12 / 14 points: s/F = 0.36, between the smooth and cliff lines: mixed")
assert pb["tenure_months"]["classification"] == "below_floor" and tb["sensitivity"]["M_5"]["classification"]["tenure_months"] != "below_floor" \
    and tb["sensitivity"]["M_15"]["classification"]["total_spend"] == "below_floor"
ok("the floor moves with M: tenure (F = 7) is assessed at M = 5 but not at 10; spend (F = 14) drops below the floor at M = 15")
assert tb["qualified"] is False and tb["candidate_effect_median_s"] is None; ok("no cliffs: the candidate does not qualify and has no effect")

print("S4  significance and replication are part of the gate")
C = build("c", {"churn_risk_score": CLIFF, "total_spend": [{0: 0.28, 1: 0.0}, 0.16, 0.18, 0.20], "tenure_months": DECOR})
tc = run(C)["shape_test"]
assert tc["per_field"]["total_spend"]["classification"] == "cliff" and tc["replicated_in_both_halves"] is False and tc["qualified"] is False
ok("an effect that lives in one customer half only: two cliffs pooled, but it fails replication, so the candidate does not qualify")
D = build("d", {"churn_risk_score": [0.02, 0.03, 0.035, 0.04], "total_spend": [0.02, 0.03, 0.035, 0.04], "tenure_months": DECOR})
td = run(D)["shape_test"]
assert all(td["per_field"][f]["classification"] == "below_floor" for f in H2.FIELDS) and not td["qualified"]; ok("a curve whose peak is under M is never a cliff, however steep its shape")
E = build("e", {"churn_risk_score": [0.06, 0.08, 0.10, 0.11], "total_spend": [0.06, 0.08, 0.10, 0.11], "tenure_months": DECOR})
pe = run(E)["shape_test"]["per_field"]["churn_risk_score"]
assert pe["s_over_F"] >= 0.5 and pe["mcnemar_5pct_vs_control"]["p_holm"] >= pe["mcnemar_5pct_vs_control"]["p_value"]; ok("Holm-adjusted p-values are never smaller than raw ones")
holm = H2._holm([0.01, 0.04, 0.03, None]); assert holm[0] == 0.03 and holm[2] == 0.06 and holm[1] == 0.06 and holm[3] is None; ok("Holm step-down is correct on a known example (0.01, 0.03, 0.04 -> 0.03, 0.06, 0.06) and skips untestable fields")

print("S5  the control is required and must be clean")
F = WORK / "f"; shutil.copytree(A, F); shutil.rmtree(F / "conditions" / H2.CONTROL_CONDITION)
try: run(F); raise SystemExit("should refuse")
except FileNotFoundError as e: assert "clean re-run control" in str(e)
ok("without the clean re-run control the evaluator refuses: H2's test is net of it")

print("V1  the validator")
okv, issues = V.validate_h2_results_schema(json.loads(json.dumps(ra)))
assert okv, issues[:5]; ok("passes on real evaluator output (scenarios a to e)")
for r in (run(B), run(C), run(D), run(E)):
    assert V.validate_h2_results_schema(json.loads(json.dumps(r)))[0]
mut = [("drop a stratum", lambda r: r["fields"]["total_spend"].pop("boundary")),
       ("a drift rate above 1", lambda r: r["fields"]["churn_risk_score"]["stable"]["drift_rates_by_magnitude"].__setitem__("5", 1.2)),
       ("an unknown classification", lambda r: r["shape_test"]["per_field"]["total_spend"].__setitem__("classification", "cliffish")),
       ("qualified with one cliff", lambda r: (r["shape_test"].__setitem__("cliff_fields", ["churn_risk_score"]), r["shape_test"]["per_field"]["total_spend"].__setitem__("classification", "mixed"), r["shape_test"].__setitem__("n_cliff_fields", 1))),
       ("Holm below raw p", lambda r: r["shape_test"]["per_field"]["churn_risk_score"]["mcnemar_5pct_vs_control"].__setitem__("p_holm", -0.1)),
       ("primary stratum changed", lambda r: r.__setitem__("primary_stratum", "all")),
       ("sensitivity missing", lambda r: r["shape_test"].pop("sensitivity")),
       ("halves do not add up", lambda r: r["shape_test"]["per_field"]["churn_risk_score"]["halves"]["0"].__setitem__("n_customers", 1))]
caught = 0
for label, fn in mut:
    r = json.loads(json.dumps(ra)); fn(r)
    caught += not V.validate_h2_results_schema(r)[0]
assert caught == len(mut); ok(f"catches all {len(mut)} planted structural faults")
buf = io.StringIO()
p = WORK / "h2.json"; r = json.loads(json.dumps(ra)); r["fields"]["churn_risk_score"]["stable"]["drift_rates_by_magnitude"]["5"] = 0.7777123
p.write_text(json.dumps(r))
old = sys.argv; sys.argv = ["validate_h2_schema.py", str(p)]
with contextlib.redirect_stdout(buf): V.main()
sys.argv = old
assert "0.7777" not in buf.getvalue() and "PASSED" in buf.getvalue(); ok("its output never prints a substantive value")

print(f"\n{n_checks} check groups passed")

#!/usr/bin/env python3
"""
Planted-data tests for H1's stratification (R3-9 / D54) and its decision rule (R3-12 / D58). No API calls. Each scenario
writes a small synthetic experiments_output with the 18 conditions H1 loads, with drift planted exactly per stratum, so
the evaluator's stable-primary numbers and the rule's verdict have known right answers. (test_evaluate_h1_manifest.py
covers the manifest-driven 1b list and recomputes Question A independently; this file covers strata and the rule.)
Run from 1b_dithering/:

    python3 test_evaluate_h1_strata.py
"""
import atexit, contextlib, io, json, random, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_h1 as E
import validate_h1_schema as V

WORK = Path(tempfile.mkdtemp(prefix="h1_strata_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a, **k)

TIERS = ["stable"] * 400 + ["deeply_boundary"] * 50 + ["lightly_boundary"] * 40 + ["tied_no_majority"] * 5 + [None] * 5
IDS = [f"CUST_{i:06d}" for i in range(1, len(TIERS) + 1)]
TIER = dict(zip(IDS, TIERS))
STABLE = [c for c in IDS if TIER[c] == "stable"]
BOUNDARY = [c for c in IDS if TIER[c] in ("deeply_boundary", "lightly_boundary", "tied_no_majority")]
OTHER = [c for c in IDS if TIER[c] is None]
TOP = list(E.TOP5_FIELDS.items())                                     # (cid, field)
COMP = [(cid, f) for cid, (f, _) in E.COMPARISON_FIELDS.items()]
CATEGORY = {c: ["nps_score", "email_open_rate"] for c in E.CATEGORY_CONDITIONS}


def pick(ids, rate, salt):
    return set(random.Random(salt).sample(sorted(ids), round(rate * len(ids))))


def build(name, stable_rates, boundary_rates=None):
    """stable_rates / boundary_rates: {condition_id: drift rate}; any condition not named drifts 5% (stable) or 45% (boundary)."""
    eo = WORK / name
    (eo / "baseline").mkdir(parents=True)
    gt = {c: {"final_decision": "MEDIUM_PRIORITY", "stability_tier": TIER[c], "decision_source": "primary_vote"} for c in IDS}
    (eo / "gt.json").write_text(json.dumps(gt))
    (eo / "baseline" / "baseline_reference.json").write_text(json.dumps(
        {"customers": {c: {"run_details": [{"key_factors": ["churn_risk_score", "nps_score"]}] * 5} for c in IDS}}))
    fields_of = {**{c: [f] for c, f in TOP}, **{c: [f] for c, f in COMP}, **CATEGORY, "h1_distributed": ["nps_score", "total_spend"]}
    for cid, fields in fields_of.items():
        d = eo / "conditions" / cid; d.mkdir(parents=True)
        drift = (pick(STABLE, stable_rates.get(cid, 0.05), f"{cid}|s")
                 | pick(BOUNDARY, (boundary_rates or {}).get(cid, 0.45), f"{cid}|b")
                 | pick(OTHER, 0.2, f"{cid}|o"))
        ref, dec = [], []
        for i, c in enumerate(IDS):
            rid = f"COND_{i:06d}"
            ref.append({"record_id": rid, "customer_id": c, "customer_segment": ["high_value", "medium_value", "low_value"][i % 3],
                        "_dither_applied": True, "_dither_fields": list(fields), "_dither_original": {f: 1 for f in fields},
                        **{f: 2 for f in fields}})
            dec.append({"record_id": rid, "business_decision": "HIGH_PRIORITY" if c in drift else "MEDIUM_PRIORITY",
                        "agent_confidence": 0.8, "decision_reasoning": "planted", "key_factors": []})
        (d / "dither_reference.json").write_text(json.dumps(ref))
        (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in dec))
    return eo


def run(eo):
    return quiet(E.evaluate_h1, eo, eo / "gt.json", eo / "baseline" / "baseline_reference.json", manifest_path=eo / "none.json")


def rates(top, comp):
    return {**{cid: r for (cid, _), r in zip(TOP, top)}, **{cid: r for (cid, _), r in zip(COMP, comp)}}


COMP_LOW = [0.10, 0.10, 0.08, 0.06, 0.05, 0.05]

print("S1  strata")
A = build("a", rates([0.40, 0.38, 0.36, 0.34, 0.32], COMP_LOW))
ra = run(A)
assert ra["primary_stratum"] == "stable" and ra["stratum_sizes"] == {"stable": 400, "boundary": 95, "all": 500}; ok("stable is primary; boundary = lightly + deeply + tied (95); all = everyone incl. 5 with no tier (500)")
tf = ra["question_a_field_importance"]["top5_fields"]
assert abs(tf["churn_risk_score"]["effective_drift_rate"] - 0.38) < 1e-9; ok("the top-level Question A numbers are the stable stratum's (planted 38% for churn)")
alls = ra["by_stratum"]["all"]["question_a_field_importance"]["comparison_fields"]["email"]["effective_drift_rate"]
assert alls > 0.10 + 0.05; ok("in 'all', 45%-drifting boundary customers inflate a quiet comparison field's drift (why stable is primary)")
assert ra["by_stratum"]["boundary"]["n_customers"] == 95 and "decision_rule_legacy_list" in ra["by_stratum"]["boundary"]; ok("boundary and all carry every analysis, the rule's numbers included (descriptive)")
assert ra["question_b_category_ranking"]["ranked_categories"][0]["drift_rate"] <= 0.06; ok("Questions B to D are stable-primary too")

print("S2  the decision rule (stable, legacy list)")
r = ra["decision_rule"]
assert r["fires"] and r["n_top_fields_qualifying"] == 5 and r["median_gap"] >= 0.10 and r["list"] == "legacy_1a"; ok("five strong top fields against quiet comparisons: every top field beats at least 5 of 6 after Holm; the median gap is far above M: FIRES")
pairs = ra["question_a_field_importance"]["pairwise_mcnemar_primary"]["all_pairs"]
assert len(pairs) == 30 and all(p["p_holm"] >= p["p_value"] for p in pairs); ok("Holm runs across all 30 pairs; adjusted p-values are never below raw ones")
B = build("b", rates([0.40, 0.38, 0.36, 0.08, 0.07], COMP_LOW))
rb = run(B)["decision_rule"]
assert not rb["fires"] and rb["n_top_fields_qualifying"] == 3; ok("three strong and two no better than the comparisons: only 3 of 5 qualify: does NOT fire")
C = build("c", rates([0.40, 0.38, 0.36, 0.34, 0.07], COMP_LOW))
rc = run(C)["decision_rule"]
assert rc["fires"] and rc["n_top_fields_qualifying"] == 4; ok("four strong and one no better: 4 of 5 qualify and the gap is material: FIRES")
D = build("d", rates([0.16, 0.16, 0.16, 0.16, 0.16], [0.08] * 6))
rd = run(D)["decision_rule"]
assert not rd["fires"] and rd["median_gap_at_least_M"] is False; ok("a uniform 8-point gap: below M = 10, so the rule does not fire whatever the p-values say")

print("S3  boundary noise cannot make the rule fire")
F = build("f", rates([0.06] * 5, [0.06] * 6), boundary_rates=rates([0.90] * 5, [0.10] * 6))
rf = run(F)
assert not rf["decision_rule"]["fires"]; ok("stable shows no top-field advantage: the primary rule does not fire")
assert rf["by_stratum"]["boundary"]["decision_rule_legacy_list"]["fires"]; ok("the same data's boundary stratum alone would 'fire': it is reported, never used for the verdict")

print("V1  the validator")
for res in (ra, run(B), run(C), run(D), rf):
    good, issues = V.validate_h1_results_schema(json.loads(json.dumps(res)))
    assert good, issues[:5]
ok("passes on every scenario's output")
muts = [("primary stratum changed", lambda x: x.__setitem__("primary_stratum", "all")),
        ("fires without a material gap", lambda x: (x["decision_rule"].__setitem__("median_gap_at_least_M", False), x["decision_rule"].__setitem__("fires", True))),
        ("qualifying count disagrees", lambda x: x["decision_rule"].__setitem__("n_top_fields_qualifying", 2)),
        ("Holm below raw p", lambda x: (lambda pr: pr.__setitem__("p_holm", pr["p_value"] / 2))(max(x["question_a_field_importance"]["pairwise_mcnemar_primary"]["all_pairs"], key=lambda q: q["p_value"]))),
        ("a stratum missing", lambda x: x["by_stratum"].pop("boundary")),
        ("stratum size disagrees", lambda x: x["stratum_sizes"].__setitem__("all", 10))]
caught = 0
for _, fn in muts:
    x = json.loads(json.dumps(ra)); fn(x); caught += not V.validate_h1_results_schema(x)[0]
assert caught == len(muts); ok(f"catches all {len(muts)} planted structural faults")

print(f"\n{n_checks} check groups passed")

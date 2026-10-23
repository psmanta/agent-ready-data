#!/usr/bin/env python3
"""
Planted-data tests for evaluate_h3.py (stratified, stable-primary; Holm within F2 and F3; the D58 decision rule) and
validate_h3_schema.py. No API calls. Each scenario writes a small synthetic experiments_output with the conditions H3
loads, with drift planted exactly per stratum, so the difference-in-differences and the rule's verdict have known
right answers. H3 had no tests before this file. Run from 1b_dithering/:

    python3 test_evaluate_h3.py
"""
import atexit, contextlib, io, json, random, shutil, sys, tempfile, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_h3 as E
import validate_h3_schema as V

warnings.filterwarnings("ignore")
WORK = Path(tempfile.mkdtemp(prefix="h3_test_"))
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
BASE = 0.10


def pick(ids, rate, salt):
    return set(random.Random(salt).sample(sorted(ids), round(rate * len(ids))))


def arm_rates(did):
    """Planted cells: correlated arm at BASE for every set; a real set's uncorrelated arm at BASE + did (or, for a
    negative did, its correlated arm raised instead); references flat, so the raw DiD equals `did`."""
    rates = {}
    for name, spec in E.SETS.items():
        d = 0.0 if spec["kind"] == "reference" else did.get(name, 0.0)
        rates[f"h3_{name}_correlated"] = BASE + max(0.0, -d)
        rates[f"h3_{name}_uncorrelated"] = BASE + max(0.0, d)
    return rates


def build(name, stable_did, boundary_did=None):
    eo = WORK / name
    (eo / "baseline").mkdir(parents=True)
    gt = {c: {"final_decision": "MEDIUM_PRIORITY", "stability_tier": TIER[c], "decision_source": "primary_vote"} for c in IDS}
    (eo / "gt.json").write_text(json.dumps(gt))
    texts = ["solid value steady engagement low churn risk", "steady engagement solid value moderate risk",
             "moderate risk solid value steady engagement"]
    (eo / "baseline" / "baseline_reference.json").write_text(json.dumps({"customers": {c: {"run_details": [
        {"business_decision": "MEDIUM_PRIORITY", "decision_reasoning": texts[k % 3], "key_factors": []} for k in range(5)]} for c in IDS}}))
    srates, brates = arm_rates(stable_did), arm_rates(boundary_did or {})
    cids = [f"h3_{n}_{a}" for n in E.SETS for a in ("correlated", "uncorrelated")]
    cids += ["h3_individual_avg_resolution_time_hours", "h3_individual_refund_rate"] + sorted(set(E.FIELD_TO_INDIVIDUAL_CONDITION.values()))
    field_of = {v: k for k, v in E.FIELD_TO_INDIVIDUAL_CONDITION.items()}
    for cid in dict.fromkeys(cids):
        setname = next((n for n in E.SETS if cid.startswith(f"h3_{n}_")), None)
        fields = E.SETS[setname]["fields"] if setname else [field_of.get(cid, cid.replace("h3_individual_", ""))]
        drift = pick(STABLE, srates.get(cid, BASE), f"{cid}|s") | pick(BOUNDARY, brates.get(cid, BASE), f"{cid}|b") | pick(OTHER, BASE, f"{cid}|o")
        rng = random.Random(cid)
        d = eo / "conditions" / cid; d.mkdir(parents=True)
        ref, dec = [], []
        for i, c in enumerate(IDS):
            rid = f"COND_{i:06d}"
            moved = fields if (i % 3) else fields[:1]          # some customers have only one field moved (2x2 directionality)
            ref.append({"record_id": rid, "customer_id": c, "customer_segment": ["high_value", "medium_value", "low_value"][i % 3],
                        "_dither_applied": True, "_dither_fields": list(moved), "_dither_original": {f: 1 for f in moved}, **{f: 2 for f in moved}})
            dec.append({"record_id": rid, "business_decision": "HIGH_PRIORITY" if c in drift else "MEDIUM_PRIORITY",
                        "agent_confidence": round(0.6 + 0.3 * rng.random(), 3),
                        "decision_reasoning": texts[rng.randrange(3)] + " planted", "key_factors": []})
        (d / "dither_reference.json").write_text(json.dumps(ref))
        (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in dec))
    return eo


def run(eo):
    return quiet(E.evaluate_h3, eo, eo / "gt.json", eo / "baseline" / "baseline_reference.json")


REAL = E.REAL_SETS
print("S1  strata and multiplicity")
A = build("a", {n: 0.20 for n in REAL})
ra = run(A)
assert ra["primary_stratum"] == "stable" and ra["stratum_sizes"] == {"stable": 400, "boundary": 95, "all": 500}; ok("stable is primary; boundary 95 (lightly + deeply + tied); all 500")
c = ra["difference_in_differences"]["comparisons"]["pair1_churn_nps"]
assert abs(c["raw_did"] - 0.20) < 1e-9 and c["n_customers"] == 400; ok("the top-level DiD is the stable stratum's: 400 customers, the planted 20-point difference-in-differences")
assert set(ra["by_stratum"]) == {"boundary", "all"} and ra["by_stratum"]["all"]["difference_in_differences"]["comparisons"]["pair1_churn_nps"]["n_customers"] == 500; ok("boundary and all carry every analysis")
pv = [ra["difference_in_differences"]["comparisons"][n] for n in REAL]
assert all(x["p_holm"] >= x["p_value"] for x in pv); ok("F2: each DiD test carries a Holm p across the four, never below its raw p")
cf = [ra["confidence_analysis"][n]["primary_unconditional"] for n in REAL]
assert all(("p_holm" in x) and (x["p_value"] is None or x["p_holm"] >= x["p_value"]) for x in cf) and "p_holm" not in ra["confidence_analysis"]["reference_pair"]["primary_unconditional"]
ok("F3: the four real sets' primary confidence tests carry their own Holm p (a separate family; references are not in it)")

print("S2  the decision rule")
r = ra["decision_rule"]
assert r["fires"] and r["direction"] == 1 and r["wording"] == "consistent" and abs(r["median_raw_did"] - 0.20) < 1e-9; ok("four 20-point effects, all significant after Holm, same direction: FIRES ('contradictions cost more')")
rb = run(build("b", {**{n: 0.20 for n in REAL[:3]}, REAL[3]: 0.0}))["decision_rule"]
assert not rb["fires"] and rb["n_significant_same_direction"] == 3 and rb["wording"].startswith("three of four"); ok("three real and one with no effect: does not fire; worded 'three of four', never 'consistent'")
rc = run(build("c", {n: -0.20 for n in REAL}))["decision_rule"]
assert rc["fires"] and rc["direction"] == -1; ok("the opposite direction fires too, and is reported as such ('contradictions cost less')")
rd = run(build("d", {n: 0.07 for n in REAL}))["decision_rule"]
assert not rd["fires"] and rd["median_at_least_M"] is False; ok("four 7-point effects: below M = 10, so the rule does not fire whatever the p-values say")
re_ = run(build("e", {REAL[0]: 0.20, REAL[1]: 0.20, REAL[2]: -0.20, REAL[3]: -0.20}))["decision_rule"]
assert not re_["fires"]; ok("split signs (two up, two down): does not fire")

print("S3  boundary noise cannot make the rule fire")
F = build("f", {n: 0.0 for n in REAL}, boundary_did={n: 0.60 for n in REAL})
rf = run(F)
assert not rf["decision_rule"]["fires"]; ok("no effect among stable customers: the primary rule does not fire")
assert rf["by_stratum"]["boundary"]["decision_rule"]["n_significant_same_direction"] >= 3; ok("the boundary stratum's planted effect shows up there, reported and never used for the verdict")

print("V1  the validator")
for res in (ra, rf):
    good, issues = V.validate_h3_results_schema(json.loads(json.dumps(res, default=str)))
    assert good, issues[:6]
ok("passes on the evaluator's output")
muts = [("primary stratum changed", lambda x: x.__setitem__("primary_stratum", "all")),
        ("a DiD p_holm removed", lambda x: x["difference_in_differences"]["comparisons"]["pair2_spend_ltv"].pop("p_holm")),
        ("fires with three significant", lambda x: (x["decision_rule"].__setitem__("n_significant_same_direction", 3))),
        ("wording 'consistent' without firing", lambda x: (x["decision_rule"].__setitem__("fires", False))),
        ("a stratum missing", lambda x: x["by_stratum"].pop("all")),
        ("a boundary-stratum p-value above 1", lambda x: x["by_stratum"]["boundary"]["difference_in_differences"]["comparisons"]["triplet"].__setitem__("p_value", 1.5)),
        ("stratum size disagrees", lambda x: x["stratum_sizes"].__setitem__("boundary", 3))]
caught = 0
for _, fn in muts:
    x = json.loads(json.dumps(ra, default=str)); fn(x); caught += not V.validate_h3_results_schema(x)[0]
assert caught == len(muts); ok(f"catches all {len(muts)} planted structural faults")

print(f"\n{n_checks} check groups passed")

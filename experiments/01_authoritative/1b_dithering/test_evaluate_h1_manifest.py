#!/usr/bin/env python3
"""
Regression test for evaluate_h1.py's manifest-driven Question A, and for validate_h1_schema.py.

Self-contained: builds its own 200-customer dataset in a temporary directory and never reads or
writes your real experiments_output. Run from 1b_dithering/:

    python3 test_evaluate_h1_manifest.py

It plants baseline citations and per-condition decisions, builds real frozen manifests with
h1_baseline_replication.py, generates the conditional conditions with
generate_h1_replication_conditions.py, runs the evaluator, and recomputes EVERY Question A number
(effective drift rates, the group-level exact Mann-Whitney, all pairwise exact McNemar tests,
win counts, the sensitivity analysis, the stated-vs-revealed rows) with plain loops and SciPy,
sharing no code with the evaluator. It also checks every refusal path and the validator.
"""
import atexit, copy, hashlib, io, contextlib, json, math, os, random, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
from scipy.stats import binomtest, mannwhitneyu
import evaluate_h1 as E
import h1_baseline_replication as H
import validate_h1_schema as V

WORK = Path(tempfile.mkdtemp(prefix="h1_eval_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
BASE = WORK / "base_output"
_g = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "200", "--seed", "7", "--out", str(BASE)], capture_output=True, text=True)
assert _g.returncode == 0 and (BASE / "ground_truth/canonical_customers.json").exists(), _g.stdout[-400:] + _g.stderr[-400:]

CANON = json.load(open(BASE / "ground_truth/canonical_customers.json"))
CUSTS = [c["customer_id"] for c in CANON]
DEC = ["HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"]
_rng = random.Random(11)
GT = {c: {"customer_id": c, "stability_tier": "stable", "final_decision": _rng.choice(DEC), "decision_source": "primary_5run"} for c in CUSTS}
RID_MAP = json.load(open(BASE / "baseline/record_id_map.json"))
BASE_IDS = [json.loads(l)["record_id"] for l in open(BASE / "baseline/agent_input/baseline_customers.jsonl")]
N = len(BASE_IDS)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def eq(a, b, label): assert a == b, f"{label}: evaluator={a!r} independent={b!r}"

# ---------------------------------------------------------------------------------------------
# Building a "world": a copy of the pieces H1 needs, with planted baseline, manifest and decisions
# ---------------------------------------------------------------------------------------------
NOT_TRIGGERED = {"last_purchase_days_ago": .9, "churn_risk_score": .85, "nps_score": .7, "lifetime_value_estimate": .6, "support_tickets_open": .5, "total_spend": .2}
TRIGGERED = {"last_purchase_days_ago": .9, "churn_risk_score": .85, "total_purchases": .7, "email_open_rate": .6, "fraud_risk_score": .5, "nps_score": .2}

def p_drift(cid):    # per-condition planted drift probability, deterministic
    return 0.15 + (int(hashlib.md5(cid.encode()).hexdigest()[:4], 16) % 55) / 100

def plant_decisions(world, cid):
    ref = json.load(open(world / "conditions" / cid / "dither_reference.json")); rng = random.Random(cid)
    with open(world / "conditions" / cid / "decisions.jsonl", "w") as f:
        for r in ref:
            fin = GT[r["customer_id"]]["final_decision"]
            d = rng.choice([x for x in DEC if x != fin]) if rng.random() < p_drift(cid) else fin
            f.write(json.dumps({"record_id": r["record_id"], "business_decision": d, "agent_confidence": 0.8, "decision_reasoning": "x", "key_factors": []}) + "\n")

def build_world(name, rates):
    w = WORK / name; (w / "conditions").mkdir(parents=True); (w / "evaluation").mkdir()
    for sub in ("ground_truth", "baseline"): shutil.copytree(BASE / sub, w / sub)
    needed = list(E.TOP5_FIELDS) + [c for c in E.COMPARISON_FIELDS] + E.CATEGORY_CONDITIONS + ["h1_distributed", "h3_individual_payment_failures", "h1_individual_nps_score"]
    for cid in dict.fromkeys(needed): shutil.copytree(BASE / "conditions" / cid, w / "conditions" / cid)
    json.dump(GT, open(w / "finalized_ground_truth.json", "w"))
    runs = {}
    for run in range(1, 6):
        rows = []
        for i, rid in enumerate(BASE_IDS):
            j = (run - 1) * N + i
            kf = [f for f, r in rates.items() if math.floor((j + 1) * r) > math.floor(j * r)]
            rows.append({"record_id": rid, "business_decision": "HIGH_PRIORITY", "agent_confidence": 0.8, "decision_reasoning": "x", "key_factors": kf, "cost_usd": 0.002})
        runs[run] = rows
        d = w / "baseline/decisions"; d.mkdir(exist_ok=True)
        with open(d / f"run{run}.decisions.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    ref = {"customers": {}}                       # legacy stated-importance source
    for rid, cust in RID_MAP.items(): ref["customers"][cust] = {"run_details": [{"key_factors": next(r for r in runs[k] if r["record_id"] == rid)["key_factors"]} for k in range(1, 6)]}
    json.dump(ref, open(w / "baseline/baseline_reference.json", "w"))
    m = H.analyze(w / "baseline/decisions", w / "baseline/agent_input/baseline_customers.jsonl", w / "ground_truth/canonical_customers.json", bootstrap_b=50)
    mp = w / "evaluation/h1_replication_manifest.json"; json.dump(m, open(mp, "w"), indent=2)
    if m["decision"]["triggered"] and m["decision"]["conditions_to_add"]:
        cp = subprocess.run([sys.executable, str(HERE / "generate_h1_replication_conditions.py"), "--manifest", str(mp), "--out", str(w)], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stdout[-500:] + cp.stderr[-500:]
    for cid in sorted(os.listdir(w / "conditions")): plant_decisions(w, cid)
    return w, mp, m

def run_eval(w, mp, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return E.evaluate_h1(w, w / "finalized_ground_truth.json", w / "baseline/baseline_reference.json", manifest_path=mp, **kw)

# ---------------------------------------------------------------------------------------------
# The independent re-implementation of Question A (plain loops + SciPy)
# ---------------------------------------------------------------------------------------------
def rows_of(w, cid):
    ref = json.load(open(w / "conditions" / cid / "dither_reference.json"))
    dec = {}
    for l in open(w / "conditions" / cid / "decisions.jsonl"): d = json.loads(l); dec[d["record_id"]] = d["business_decision"]
    return {r["customer_id"]: {"drift": dec[r["record_id"]] != GT[r["customer_id"]]["final_decision"], "fields": r["_dither_fields"]} for r in ref}

def expected_qa(w, top, comp, with_pairs=True):
    R = {c: rows_of(w, c) for c in list(top) + list(comp)}
    def stats(cid, f):
        rows = R[cid]; pert = [v for v in rows.values() if f in v["fields"]]; n = len(pert)
        return {"eff": (sum(v["drift"] for v in pert) / n) if n else None, "n": n, "exp": n / len(rows), "raw": sum(v["drift"] for v in rows.values()) / len(rows)}
    S = {c: stats(c, f) for c, f in {**top, **comp}.items()}
    pairs, tw, cw = [], 0, 0
    for tc, tf in top.items():
        for cc, cf in comp.items():
            A = {k: v["drift"] for k, v in R[tc].items() if v["fields"]}; B = {k: v["drift"] for k, v in R[cc].items() if v["fields"]}
            shared = sorted(set(A) & set(B)); b = sum(A[k] and not B[k] for k in shared); c = sum(B[k] and not A[k] for k in shared)
            t, u = S[tc]["eff"] or 0.0, S[cc]["eff"] or 0.0
            tw += t > u; cw += u > t
            pairs.append({"top5_field": tf, "comparison_field": cf, "n_perturbed_in_both": len(shared), "b_x_only": b, "c_y_only": c, "n_discordant": b + c,
                          "p_value": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0, "t": round(t, 4), "u": round(u, 4)})
    te = [S[c]["eff"] for c in top if S[c]["eff"] is not None]; ce = [S[c]["eff"] for c in comp if S[c]["eff"] is not None]
    U, P = mannwhitneyu(te, ce, method="exact", alternative="two-sided")
    return {"S": S, "pairs": pairs, "tw": tw, "cw": cw, "U": float(U), "P": float(P), "nsig": sum(p["p_value"] < 0.05 for p in pairs)}

def check_qa(res, top, comp, exp, label, with_pairs=True):
    for grp, mapping, key in ((top, top, "top5_fields"), (comp, comp, "comparison_fields")):
        eq(list(res[key]), list(mapping.values()), f"{label} {key} order")
        for cid, f in mapping.items():
            s, got = exp["S"][cid], res[key][f]
            eq(got["effective_drift_rate"], None if s["eff"] is None else round(s["eff"], 4), f"{label} {f} effective"); eq(got["raw_drift_rate"], round(s["raw"], 4), f"{label} {f} raw")
            eq(got["exposure"], round(s["exp"], 4), f"{label} {f} exposure"); eq(got["n_perturbed"], s["n"], f"{label} {f} n_perturbed")
    g = res["group_level_check_secondary"]; assert abs(g["u_statistic"] - exp["U"]) < 1e-9 and abs(g["p_value"] - exp["P"]) < 1e-9, (label, g, exp["U"], exp["P"])
    pw = res["pairwise_mcnemar_primary"]
    eq(pw["n_pairs"], len(exp["pairs"]), f"{label} n_pairs"); eq(pw["top5_higher_drift_count"], exp["tw"], f"{label} top wins"); eq(pw["comparison_higher_drift_count"], exp["cw"], f"{label} comp wins")
    eq(pw["n_significant_p05"], exp["nsig"], f"{label} n_significant")
    if with_pairs:
        for got, e in zip(pw["all_pairs"], exp["pairs"]):
            for k in ("top5_field", "comparison_field", "n_perturbed_in_both", "b_x_only", "c_y_only", "n_discordant"): eq(got[k], e[k], f"{label} pair {e['top5_field']}x{e['comparison_field']} {k}")
            assert abs(got["p_value"] - e["p_value"]) < 1e-12, (label, got, e)
            eq(got["top5_effective_drift_rate"], e["t"], f"{label} pair t"); eq(got["comparison_effective_drift_rate"], e["u"], f"{label} pair u")
    else:
        assert "all_pairs" not in pw

LEG_TOP = dict(E.TOP5_FIELDS); LEG_COMP = {c: f for c, (f, _) in E.COMPARISON_FIELDS.items()}

# =============================================================================================
print("W1  NOT TRIGGERED: the legacy list is primary and the 1b section says so")
wN, mpN, mN = build_world("not_triggered", NOT_TRIGGERED); assert not mN["decision"]["triggered"]
rN = run_eval(wN, mpN)
check_qa(rN["question_a_field_importance"], LEG_TOP, LEG_COMP, expected_qa(wN, LEG_TOP, LEG_COMP), "legacy/N"); ok("legacy Question A matches the independent recomputation (5 x 6: effective rates, exact Mann-Whitney, 30 exact McNemar tests, win counts)")
eq(rN["question_a_manifest"], {"status": "used", "decision_sha256": mN["decision_sha256"], "triggered": False, "primary_list": "legacy_1a", "legacy_role": "primary"}, "manifest block"); eq(rN["question_a_1b_list"]["status"], "not_triggered", "1b status"); ok("manifest block: used, not triggered, legacy list primary; 1b section = not_triggered")
ok_, issues = V.validate_h1_results_schema(rN); assert ok_, issues; ok("the validator accepts the output")

print("W2  TRIGGERED: the 1b list is primary; every number recomputed independently")
wT, mpT, mT = build_world("triggered", TRIGGERED); assert mT["decision"]["triggered"]
newc = sorted(os.listdir(wT / "conditions")); assert {"h1_replication_total_purchases", "h1_replication_email_open_rate", "h1_replication_fraud_risk_score"} <= set(newc); ok("the generator produced the 3 new conditions this scenario calls for")
rT = run_eval(wT, mpT)
check_qa(rT["question_a_field_importance"], LEG_TOP, LEG_COMP, expected_qa(wT, LEG_TOP, LEG_COMP), "legacy/T"); ok("the legacy list is still analysed on a triggered run (as the legacy comparison)")
eq(rT["question_a_field_importance"], rN["question_a_field_importance"], "legacy identical across manifests"); ok("the legacy analysis is IDENTICAL whether or not the trigger fired (the manifest never alters it)")
eq(rT["question_a_manifest"]["legacy_role"], "legacy comparison", "legacy role"); eq(rT["question_a_manifest"]["primary_list"], "1b", "primary list")
g = mT["question_a_groups"]["primary_1b"]; top1b = {e["condition_id"]: e["field"] for e in g["top"]}; comp1b = {e["condition_id"]: e["field"] for e in g["comparison"]}
sens = {e["condition_id"]: e["field"] for e in g["comparison_sensitivity_excluding_1b_ranks_6_to_8"]}
sec = rT["question_a_1b_list"]; eq(sec["status"], "ok", "1b status"); eq(sec["role"], "primary", "role"); eq(sec["manifest_decision_sha256"], mT["decision_sha256"], "1b hash")
check_qa(sec["analysis"], top1b, comp1b, expected_qa(wT, top1b, comp1b), "1b"); ok(f"Question A on the 1b list ({len(top1b)} x {len(comp1b)} = {len(top1b) * len(comp1b)} exact McNemar tests, group-level exact Mann-Whitney, win counts) matches the independent recomputation")
assert "h3_individual_payment_failures" in comp1b and not ({"h1_replication_total_purchases"} & set(comp1b)); ok("the 1b comparison group includes payment_failures (not one of the amendment's six) and excludes every top-group field")
assert len(sens) < len(comp1b) and set(sens) <= set(comp1b)
check_qa(sec["sensitivity_excluding_1b_ranks_6_to_8"]["analysis"], top1b, sens, expected_qa(wT, top1b, sens), "1b-sens", with_pairs=False); ok(f"the pre-registered sensitivity analysis ({len(top1b)} x {len(sens)}, comparison fields at raw 1b ranks 6-8 removed) matches")
ok_, issues = V.validate_h1_results_schema(rT); assert ok_, issues; ok("the validator accepts the triggered output")

print("W3  stated vs revealed importance, on both lists")
sv = rT["stated_vs_revealed_importance"]; rate_of = {r["field"]: r["rate"] for r in mT["citation_table"]}
ref = json.load(open(wT / "baseline/baseline_reference.json")); cnt = {}; tot = 0
for e in ref["customers"].values():
    for run in e["run_details"]:
        tot += 1
        for f in run["key_factors"]: cnt[f] = cnt.get(f, 0) + 1
eq(sv["stated_importance_all_fields"], {f: round(c / tot, 4) for f, c in cnt.items()}, "legacy stated importance"); ok("legacy stated importance (raw key_factors strings) is unchanged")
raw = lambda cid: sum(v["drift"] for v in rows_of(wT, cid).values()) / N
for row in sv["top5_comparison"]:
    cid = next(c for c, f in LEG_TOP.items() if f == row["field"])
    eq(row["revealed_drift_rate"], round(raw(cid), 4), "legacy revealed"); eq(row["manifest_citation_rate"], rate_of.get(row["field"], 0.0), "manifest rate beside legacy")
ok("legacy rows keep their values and gain the manifest's pre-registered citation rate beside them (the two instruments are visible side by side)")
b1 = sv["top5_comparison_1b"]; eq(b1["role"], "primary", "1b sv role")
eq({r["field"]: (r["condition_id"], r["stated_citation_rate"], r["revealed_drift_rate"], r["in_1a_top5"]) for r in b1["rows"]},
   {e["field"]: (e["condition_id"], rate_of.get(e["field"], 0.0), round(raw(e["condition_id"]), 4), e["field"] in LEG_TOP.values()) for e in g["top"]}, "1b stated-vs-revealed rows")
assert [r["revealed_drift_rate"] for r in b1["rows"]] == sorted((r["revealed_drift_rate"] for r in b1["rows"]), reverse=True); ok("the 1b rows take their stated rate from the manifest, their revealed rate from the data, and are sorted by revealed drift")
assert "top5_comparison_1b" not in rN["stated_vs_revealed_importance"]; ok("no 1b rows when the trigger did not fire")

print("W4  no manifest at all")
rM = run_eval(wT, wT / "evaluation/does_not_exist.json")
eq(rM["question_a_manifest"]["status"], "no_manifest", "no manifest"); eq(rM["question_a_1b_list"], {"status": "skipped: no manifest"}, "1b skipped")
eq(rM["question_a_field_importance"], rT["question_a_field_importance"], "legacy without manifest"); assert "manifest_citation_rate" not in rM["stated_vs_revealed_importance"]["top5_comparison"][0]
ok("without a manifest: the 1b analysis is skipped LOUDLY (status + warning), the legacy analysis is byte-identical, and nothing else changes")
ok_, issues = V.validate_h1_results_schema(rM); assert ok_, issues; ok("the validator accepts this output too")

print("W5  refusals")
def mutate_manifest(w, fn, rehash=False):
    m = json.load(open(w / "evaluation/h1_replication_manifest.json")); fn(m)
    if rehash: m["decision_sha256"] = H.manifest_decision_sha256(m)
    p = w / "evaluation/tampered.json"; json.dump(m, open(p, "w")); return p
def refuses(fn, needle, w=wT, **kw):
    try: run_eval(w, fn, **kw)
    except (ValueError, SystemExit) as e: assert needle in str(e), (needle, str(e)[:300]); return
    raise AssertionError("should have refused: " + needle)
refuses(mutate_manifest(wT, lambda m: m["decision"].__setitem__("triggered", False)), "edited after it was frozen"); ok("a hand-edited manifest is refused")
refuses(mutate_manifest(wT, lambda m: m["question_a_groups"]["legacy_1a"]["top"].reverse(), rehash=True), "differs from this evaluator's TOP5_FIELDS"); ok("a manifest whose legacy groups disagree with the evaluator's own constants is refused (two sources must not disagree)")
p0 = wT / "baseline/decisions/run2.decisions.jsonl"; saved = p0.read_text(); p0.write_text(saved.replace("HIGH_PRIORITY", "LOW_PRIORITY", 1))
refuses(mpT, "baseline/decisions/run2.decisions.jsonl is missing or changed"); p0.write_text(saved); ok("a baseline run file changed after the manifest was frozen is refused (named)")
cpath = wT / "ground_truth/canonical_customers.json"; csaved = cpath.read_bytes(); d = json.loads(csaved); d[0]["dob"] = "1900-01-01"; cpath.write_text(json.dumps(d, indent=2))
refuses(mpT, "canonical_customers.json is missing or changed"); cpath.write_bytes(csaved); ok("a changed canonical customer file is refused (the date-of-birth hazard)")
hidden = wT / "conditions/h1_replication_fraud_risk_score/decisions.jsonl"; hsaved = hidden.read_text(); hidden.unlink()
refuses(mpT, "h1_replication_fraud_risk_score"); ok("pre-registered 1b conditions with no decisions: the run is refused, naming them")
ri = run_eval(wT, mpT, allow_incomplete_1b=True); eq(ri["question_a_1b_list"]["status"], "incomplete", "incomplete"); eq(ri["question_a_1b_list"]["missing_conditions"], ["h1_replication_fraud_risk_score"], "missing list")
eq(ri["question_a_field_importance"], rT["question_a_field_importance"], "legacy intact when incomplete"); ok_, issues = V.validate_h1_results_schema(ri); assert ok_, issues
ok("--allow_incomplete_1b proceeds knowingly: status 'incomplete' lists what is missing, the legacy analysis is intact, and the validator accepts it"); hidden.write_text(hsaved)

print("W6  command line and validator")
out = WORK / "h1_results.json"
cp = subprocess.run([sys.executable, "evaluate_h1.py", "--experiments_output", str(wT), "--finalized_ground_truth", str(wT / "finalized_ground_truth.json"),
                     "--baseline_reference", str(wT / "baseline/baseline_reference.json"), "--manifest", str(mpT), "--out", str(out)], capture_output=True, text=True)
assert cp.returncode == 0 and out.exists(), cp.stdout[-400:] + cp.stderr[-600:]
eq(json.load(open(out))["question_a_1b_list"]["analysis"]["pairwise_mcnemar_primary"]["n_pairs"], len(top1b) * len(comp1b), "cli n_pairs"); ok("the CLI runs end to end and writes the 1b analysis")
cv = subprocess.run([sys.executable, "validate_h1_schema.py", str(out)], capture_output=True, text=True); assert cv.returncode == 0 and "PASSED" in cv.stdout; ok("the validator CLI passes the written file")
def corrupt(label, fn, needle):
    r = copy.deepcopy(rT); fn(r); good, iss = V.validate_h1_results_schema(r)
    assert (not good) and any(needle in i for i in iss), (label, iss[:3]); print(f"  ✅ validator catches: {label}")
a = lambda r: r["question_a_1b_list"]["analysis"]["pairwise_mcnemar_primary"]
corrupt("n_pairs no longer equals top x comparison", lambda r: a(r).__setitem__("n_pairs", a(r)["n_pairs"] + 1), "n_pairs == n_top x n_comparison")
corrupt("wins exceed the number of pairs", lambda r: a(r).__setitem__("top5_higher_drift_count", a(r)["n_pairs"] + 1), "wins on both sides")
corrupt("a p-value above 1", lambda r: a(r)["all_pairs"][0].__setitem__("p_value", 1.5), "p_value in [0,1]")
corrupt("1b section carries another manifest's hash", lambda r: r["question_a_1b_list"].__setitem__("manifest_decision_sha256", "0" * 64), "carries the manifest's hash")
corrupt("1b section says not_triggered although the trigger fired", lambda r: r["question_a_1b_list"].__setitem__("status", "not_triggered"), "ok or incomplete when the trigger fired")
corrupt("the 1b stated-vs-revealed rows removed", lambda r: r["stated_vs_revealed_importance"].pop("top5_comparison_1b"), "1b rows present")
corrupt("the sensitivity analysis grows a pair list", lambda r: r["question_a_1b_list"]["sensitivity_excluding_1b_ranks_6_to_8"]["analysis"]["pairwise_mcnemar_primary"].__setitem__("all_pairs", []), "sensitivity analysis carries no pair list")
corrupt("an effective rate outside [0,1]", lambda r: r["question_a_field_importance"]["top5_fields"]["nps_score"].__setitem__("effective_drift_rate", 1.4), "effective_drift_rate in [0,1]")
print(f"\n{n_checks} check groups passed")

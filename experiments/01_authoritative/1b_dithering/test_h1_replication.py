#!/usr/bin/env python3
"""
Regression test for h1_baseline_replication.py and generate_h1_replication_conditions.py.

Self-contained: it generates its own small dataset (200 customers) in a temporary
directory and never reads or writes your real experiments_output. Run it from
1b_dithering/ before relying on the rule:

    python3 test_h1_replication.py

Each scenario plants baseline decisions with EXACT citation counts, then
recomputes every expected value with plain loops that share no code with the
scripts: overlap, ranking, ties, skipped fields, coverage by existing
conditions, the measurement rules, the trigger, the frozen manifest (integrity hash,
write-once, machine independence, the pre-registered Question A groups), a bootstrap
replicated independently on Python's RNG, and the generator's safety checks (edited
manifest, changed canonical file, collisions, overwrites, idempotence).
"""
import atexit, hashlib, json, math, os, random, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import h1_baseline_replication as H
from dither_engine import NUMERIC_FIELD_META, CATEGORICAL_FIELD_META, BOOLEAN_FIELDS

WORK = Path(tempfile.mkdtemp(prefix="h1_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)   # cleaned up even if a check fails
OUT = WORK / "experiments_output"
_gen = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "200", "--seed", "7", "--out", str(OUT)],
                      capture_output=True, text=True)
assert _gen.returncode == 0 and (OUT / "ground_truth/canonical_customers.json").exists(), _gen.stdout[-500:] + _gen.stderr[-500:]
BASELINE_INPUT = OUT / "baseline/agent_input/baseline_customers.jsonl"
CANONICAL = OUT / "ground_truth/canonical_customers.json"
SCEN = WORK / "scenarios"; SCEN.mkdir()

A1A = ["last_purchase_days_ago", "churn_risk_score", "nps_score", "lifetime_value_estimate", "support_tickets_open"]
ENGINE_CAN_DITHER = set(NUMERIC_FIELD_META) | set(CATEGORICAL_FIELD_META) | set(BOOLEAN_FIELDS)
EXISTING_IDS = {"last_purchase_days_ago": "h1_individual_last_purchase_days_ago", "churn_risk_score": "h1_individual_churn_risk_score",
                "nps_score": "h1_individual_nps_score", "lifetime_value_estimate": "h1_individual_lifetime_value_estimate",
                "support_tickets_open": "h1_individual_support_tickets_open", "email": "h1_individual_email", "is_vip": "h1_individual_is_vip",
                "total_spend": "h2_total_spend_mag15pct", "tenure_months": "h2_tenure_months_mag15pct",
                "avg_resolution_time_hours": "h3_individual_avg_resolution_time_hours", "refund_rate": "h3_individual_refund_rate",
                "payment_failures": "h3_individual_payment_failures"}
LEGACY6 = ["email", "is_vip", "total_spend", "tenure_months", "avg_resolution_time_hours", "refund_rate"]
EXISTING_12 = {"last_purchase_days_ago", "churn_risk_score", "nps_score", "lifetime_value_estimate", "support_tickets_open",
               "email", "is_vip", "total_spend", "tenure_months", "avg_resolution_time_hours", "refund_rate", "payment_failures"}
SCHEMA = H.schema_fields(BASELINE_INPUT)
BASE_IDS = [json.loads(l)["record_id"] for l in open(BASELINE_INPUT)]
N = len(BASE_IDS)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def eq(a, b, label):
    assert a == b, f"{label}: script={a!r} independent={b!r}"

# ----------------------------------------------------------------------------
def plant(name, rates, noise=True, mutate=None):
    """Exact citation counts: decision j cites field f iff floor((j+1)r) > floor(jr)."""
    d = SCEN / name / "decisions"; d.mkdir(parents=True)
    for run in range(1, 6):
        rows = []
        for i, rid in enumerate(BASE_IDS):
            j = (run - 1) * N + i
            kf = [f for f, r in rates.items() if math.floor((j + 1) * r) > math.floor(j * r)]
            rows.append({"record_id": rid, "business_decision": "HIGH_PRIORITY", "agent_confidence": 0.8,
                         "decision_reasoning": "x", "key_factors": kf, "cost_usd": 0.002})
        if noise and run == 1:
            rows[0]["key_factors"] = ["recent purchase", "Churn_Risk_Score ", "churn risk score", "churn_risk_score"]
        if noise and run == 2:
            rows[1]["business_decision"] = "PARSE_ERROR"; rows[1]["key_factors"] = []
        if noise and run == 3:
            rows[2]["key_factors"] = "churn_risk_score, nps_score"          # not a list
        if mutate: mutate(run, rows)
        with open(d / f"run{run}.decisions.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    return d

def expected(dec_dir):
    """The rule, re-implemented with plain loops (shares no code with the script)."""
    lookup = {f.lower(): f for f in SCHEMA}
    used = excluded = items = unmatched = 0; cnt = Counter()
    for run in range(1, 6):
        for line in open(dec_dir / f"run{run}.decisions.jsonl"):
            r = json.loads(line); kf = r["key_factors"]
            if r["business_decision"] in (None, "PARSE_ERROR") or not isinstance(kf, list):
                excluded += 1; continue
            used += 1; seen = set()
            for it in kf:
                items += 1; f = lookup.get(it.strip().lower()) if isinstance(it, str) else None
                if f is None: unmatched += 1
                else: seen.add(f)
            for f in seen: cnt[f] += 1
    rate = {f: cnt[f] / used for f in SCHEMA}
    ranked = sorted(SCHEMA, key=lambda f: (-rate[f], f))
    top5 = ranked[:5]; overlap = sorted(set(top5) & set(A1A))
    chosen, skipped = [], []
    for f in ranked:
        (chosen if f in ENGINE_CAN_DITHER else skipped).append(f)
        if len(chosen) == 5: break
    trig = len(overlap) <= 2
    return dict(cnt=dict(cnt), used=used, excluded=excluded, items=items, unmatched=unmatched, rate=rate, ranked=ranked, top5=top5,
                overlap=overlap, top8_overlap=sorted(set(ranked[:8]) & set(A1A)), chosen=chosen, skipped=skipped, trig=trig,
                to_add=[f for f in chosen if f not in EXISTING_12] if trig else [], margin=rate[ranked[4]] - rate[ranked[5]])

def check(name, res, exp):
    c, r, d = res["counts"], res["replication"], res["decision"]
    eq(c["n_decisions_used"], exp["used"], f"{name} used"); eq(c["n_decisions_excluded"], exp["excluded"], f"{name} excluded")
    eq(c["n_items"], exp["items"], f"{name} items"); eq(c["n_unmatched_items"], exp["unmatched"], f"{name} unmatched")
    eq([x["field"] for x in res["citation_table"]], exp["ranked"], f"{name} COMPLETE table order")
    for i, x in enumerate(res["citation_table"]):
        eq(x["rank"], i + 1, f"{name} rank"); eq(x["citations"], exp["cnt"].get(x["field"], 0), f"{name} count {x['field']}")
        assert abs(x["rate"] - exp["rate"][x["field"]]) < 1e-12, (name, x)
    eq(r["top5_1b"], exp["top5"], f"{name} top5"); eq(r["top5_overlap_fields"], exp["overlap"], f"{name} overlap"); eq(r["top5_overlap"], len(exp["overlap"]), f"{name} k")
    eq(r["top8_overlap_fields"], exp["top8_overlap"], f"{name} top8 overlap")
    assert abs(r["margin_rank5_rank6"] - exp["margin"]) < 1e-12
    eq(d["triggered"], exp["trig"], f"{name} triggered"); eq(d["dither_list_1b"], exp["chosen"], f"{name} dither list")
    eq(d["skipped_ineligible"], exp["skipped"], f"{name} skipped"); eq([x["field"] for x in d["conditions_to_add"]], exp["to_add"], f"{name} to_add")
    eq(sorted(d["covered_by_existing"]), sorted(f for f in exp["chosen"] if f in EXISTING_12), f"{name} covered")
    # the pre-registered Question A groups, recomputed independently
    g = res["question_a_groups"]; rank_of = {f: i + 1 for i, f in enumerate(exp["ranked"])}
    eq(g["primary_list"], "1b" if exp["trig"] else "legacy_1a", f"{name} primary list")
    eq([(e["field"], e["condition_id"]) for e in g["legacy_1a"]["top"]], [(f, EXISTING_IDS[f]) for f in A1A], f"{name} legacy top")
    eq([(e["field"], e["condition_id"]) for e in g["legacy_1a"]["comparison"]], [(f, EXISTING_IDS[f]) for f in LEGACY6], f"{name} legacy comparison")
    if not exp["trig"]:
        eq(g["primary_1b"], None, f"{name} no 1b group")
    else:
        top = [(f, EXISTING_IDS.get(f, "h1_replication_" + f)) for f in exp["chosen"]]
        comp = [(f, EXISTING_IDS[f]) for f in sorted(EXISTING_IDS) if f not in exp["chosen"]]
        sens = [(f, i) for f, i in comp if rank_of[f] > 8]
        p = g["primary_1b"]
        eq([(e["field"], e["condition_id"]) for e in p["top"]], top, f"{name} 1b top"); eq([(e["field"], e["condition_id"]) for e in p["comparison"]], comp, f"{name} 1b comparison")
        eq([(e["field"], e["condition_id"]) for e in p["comparison_sensitivity_excluding_1b_ranks_6_to_8"]], sens, f"{name} 1b sensitivity")
    eq([(f, v["rank_1a"]) for f, v in r["a1a_landing"].items()], [(f, i + 1) for i, f in enumerate(A1A)], f"{name} rank_1a")
    for f, v in r["a1a_landing"].items(): eq(v["rank_1b"], rank_of[f], f"{name} landing {f}")
    assert H.verify_manifest(res); eq(res["decision_sha256"], indep_hash(res), f"{name} decision hash")

def indep_hash(m):
    body = {k: v for k, v in m.items() if k not in ("context", "decision_sha256")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

def indep_bootstrap(dec_dir, B):
    """The bootstrap re-implemented with plain loops on Python's RNG (same call pattern as the rule)."""
    lookup = {f.lower(): f for f in SCHEMA}; per = {}
    for run in range(1, 6):
        for line in open(dec_dir / f"run{run}.decisions.jsonl"):
            r = json.loads(line); kf = r["key_factors"]
            if r["business_decision"] in (None, "PARSE_ERROR") or not isinstance(kf, list): continue
            seen = {lookup[i.strip().lower()] for i in kf if isinstance(i, str) and i.strip().lower() in lookup}
            per.setdefault(r["record_id"], Counter()).update(seen)
    custs = sorted(per); rng = random.Random(20261009); dist = Counter()
    for _ in range(B):
        idx = [rng.randrange(len(custs)) for _ in range(len(custs))]; tot = Counter()
        for i in idx: tot.update(per[custs[i]])
        top = sorted(SCHEMA, key=lambda f: (-tot[f], f))[:5]; dist[len(set(top) & set(A1A))] += 1
    return {k: dist.get(k, 0) for k in range(6)}

def run(name, **kw): return H.analyze(SCEN / name / "decisions", BASELINE_INPUT, CANONICAL, bootstrap_b=kw.pop("B", 300), **kw)

# ============================================================================
print("T0  the script's notion of 'already has a condition' and 'eligible' match the verified facts")
eq(set(H.existing_individual_fields()), EXISTING_12, "existing individual fields"); ok("12 existing individual fields, exactly the verified set")
eq(set(H.ELIGIBLE_FIELDS), ENGINE_CAN_DITHER, "eligible"); ok("eligible fields == everything the engine can dither (25)")
assert not ({"customer_segment", "is_at_risk", "recently_contacted_support"} & set(H.ELIGIBLE_FIELDS)); ok("protected fields are not eligible")

print("T1  scenarios with exact planted counts, every output recomputed independently")
S = {
 "A_overlap5": {"last_purchase_days_ago": .9, "churn_risk_score": .85, "nps_score": .7, "lifetime_value_estimate": .6, "support_tickets_open": .5, "total_spend": .2, "tenure_months": .1},
 "B_overlap3": {"last_purchase_days_ago": .9, "churn_risk_score": .85, "nps_score": .7, "total_spend": .6, "tenure_months": .5, "lifetime_value_estimate": .2, "support_tickets_open": .1},
 "C_overlap2_new": {"last_purchase_days_ago": .9, "churn_risk_score": .85, "total_purchases": .7, "email_open_rate": .6, "fraud_risk_score": .5, "nps_score": .2},
 "D_overlap2_covered": {"last_purchase_days_ago": .9, "churn_risk_score": .85, "total_spend": .7, "tenure_months": .6, "payment_failures": .5, "nps_score": .2},
 "E_ineligible": {"customer_segment": .95, "is_at_risk": .9, "churn_risk_score": .8, "last_purchase_days_ago": .7, "total_purchases": .6, "fraud_risk_score": .5, "email_open_rate": .4, "nps_score": .1},
 "F_overlap0": {"total_purchases": .9, "email_open_rate": .85, "fraud_risk_score": .8, "avg_order_value": .75, "purchase_frequency_days": .7, "nps_score": .1},
 "I_more_ineligible": {"dob": .95, "account_created_date": .9, "preferred_categories": .85, "churn_risk_score": .8, "total_purchases": .7, "fraud_risk_score": .6, "email_open_rate": .5, "avg_order_value": .4, "nps_score": .1},
 "G_tie": {"last_purchase_days_ago": .9, "churn_risk_score": .8, "nps_score": .7, "total_purchases": .5, "email_open_rate": .5, "fraud_risk_score": .5, "lifetime_value_estimate": .1},
 "K_close": {"last_purchase_days_ago": .9, "churn_risk_score": .85, "email_open_rate": .7, "fraud_risk_score": .6, "nps_score": .5, "total_purchases": .49},
}
RES, EXP = {}, {}
for name, rates in S.items():
    plant(name, rates); EXP[name] = expected(SCEN / name / "decisions"); RES[name] = run(name); check(name, RES[name], EXP[name])
eq([EXP[k]["trig"] for k in S], [False, False, True, True, True, True, True, False, False], "designed triggers")
eq([len(EXP[k]["overlap"]) for k in S], [5, 3, 2, 2, 2, 0, 1, 3, 3], "designed overlaps"); ok("all 9 scenarios match the independent recomputation, and planted overlaps are as designed")
eq(RES["C_overlap2_new"]["decision"]["conditions_to_add"][0]["condition_id"], "h1_replication_total_purchases", "id"); ok("condition ids named h1_replication_<field>")
eq([x["field"] for x in RES["C_overlap2_new"]["decision"]["conditions_to_add"]], ["total_purchases", "email_open_rate", "fraud_risk_score"], "C list"); ok("C: overlap 2 -> 3 new conditions (churn and last_purchase already covered)")
eq(RES["D_overlap2_covered"]["decision"]["conditions_to_add"], [], "D to_add"); ok("D: overlap 2 but every field already has a condition -> triggered, nothing new")
eq(RES["E_ineligible"]["decision"]["skipped_ineligible"], ["customer_segment", "is_at_risk"], "E skipped"); ok("E: customer_segment and is_at_risk skipped; next eligible fields promoted")
eq(RES["I_more_ineligible"]["decision"]["skipped_ineligible"], ["dob", "account_created_date", "preferred_categories"], "I skipped")
eq(RES["I_more_ineligible"]["decision"]["dither_list_1b"], ["churn_risk_score", "total_purchases", "fraud_risk_score", "email_open_rate", "avg_order_value"], "I chosen"); ok("I: dob, account_created_date and preferred_categories (not ditherable) skipped; five eligible fields chosen")
eq(len(RES["F_overlap0"]["decision"]["conditions_to_add"]), 5, "F count"); ok("F: overlap 0 -> the full 5 new conditions (the cap is 5, not 2)")
eq(RES["G_tie"]["replication"]["top5_1b"], ["last_purchase_days_ago", "churn_risk_score", "nps_score", "email_open_rate", "fraud_risk_score"], "G top5")
assert RES["G_tie"]["replication"]["margin_rank5_rank6"] == 0.0; ok("G: three-way tie broken alphabetically (email_open_rate, fraud_risk_score in; total_purchases out), margin 0 reported")

print("T2  measurement rules: normalization, once-per-decision, unmatched share, exclusions")
c = RES["A_overlap5"]["counts"]
assert c["n_decisions_excluded"] == 2 and c["n_unmatched_items"] >= 2; ok("PARSE_ERROR and non-list key_factors excluded (2), counted")
top_un = dict(RES["A_overlap5"]["context"]["top_unmatched"]); assert "recent purchase" in top_un and "churn risk score" in top_un; ok("'recent purchase' and 'churn risk score' (spaces) are NOT assigned to any field")
assert abs(c["unmatched_share"] - c["n_unmatched_items"] / c["n_items"]) < 1e-12; ok("unmatched share = unmatched items / all items")
# the noisy decision cites churn_risk_score under 2 spellings that both normalize to it: must count once
plant("H_dup", {"nps_score": 0.0}, noise=False, mutate=lambda run, rows: rows[0].update(key_factors=["churn_risk_score", "CHURN_RISK_SCORE ", " churn_risk_score"]) if run == 1 else None)
r = run("H_dup"); cr = next(x for x in r["citation_table"] if x["field"] == "churn_risk_score"); eq(cr["citations"], 1, "dup citations"); ok("one decision citing a field three ways counts once")

plant("J_meta", {"nps_score": 0.2}, noise=False, mutate=lambda run, rows: rows[3].update(key_factors=["record_id", "customer_id", "_dither_applied", "churn_risk_score"]) if run == 1 else None)
rj = run("J_meta"); eq(rj["counts"]["n_unmatched_items"], 3, "meta unmatched")
assert not ({"record_id", "customer_id", "_dither_applied"} & {x["field"] for x in rj["citation_table"]}); ok("metadata names (record_id, customer_id, _dither_applied) can never enter the ranking: they are unmatched")
eq(len(RES["A_overlap5"]["citation_table"]), len(SCHEMA), "complete table"); assert any(x["citations"] == 0 for x in RES["A_overlap5"]["citation_table"]); ok("the citation table lists EVERY schema field, including those never cited")
print("T3  the sequencing guard and completeness checks")
agent_dir = SCEN / "agent_results"; agent_dir.mkdir(); (agent_dir / "x.decisions.jsonl").write_text('{"a": 1}\n')
assert run("A_overlap5", agent_results_dir=agent_dir)["context"]["sequencing"]["dithered_decisions_already_present"] is True; ok("warns when dithered-condition decisions already exist")
assert run("A_overlap5", agent_results_dir=SCEN / "nonexistent")["context"]["sequencing"]["dithered_decisions_already_present"] is False; ok("no warning when none exist")
shutil.copytree(SCEN / "A_overlap5", SCEN / "miss"); (SCEN / "miss/decisions/run5.decisions.jsonl").unlink()
try: run("miss"); raise SystemExit("should refuse")
except FileNotFoundError as e: assert "complete baseline" in str(e)
ok("refuses a baseline with a missing run")
shutil.copytree(SCEN / "A_overlap5", SCEN / "badids"); p = SCEN / "badids/decisions/run3.decisions.jsonl"
rows = [json.loads(l) for l in open(p)]; rows[5]["record_id"] = "BASE_NOT_A_REAL_ID"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
try: run("badids"); raise SystemExit("should refuse")
except ValueError as e: assert "differ from run1" in str(e)
ok("refuses runs whose record_ids differ")

print("T4  bootstrap and Spearman")
eq(RES["A_overlap5"]["context"]["bootstrap"]["p_overlap_le_trigger"], 0.0, "A p"); eq(RES["F_overlap0"]["context"]["bootstrap"]["p_overlap_le_trigger"], 1.0, "F p")
ok("clear-cut cases: P(overlap<=2) = 0 for overlap 5, = 1 for overlap 0")
for name in ("A_overlap5", "C_overlap2_new", "F_overlap0", "I_more_ineligible", "K_close"):
    eq(H.analyze(SCEN / name / "decisions", BASELINE_INPUT, CANONICAL, bootstrap_b=60)["context"]["bootstrap"]["overlap_distribution"], indep_bootstrap(SCEN / name / "decisions", 60), f"{name} bootstrap")
ok("the bootstrap distribution equals an independent plain-Python replication, exactly, on 5 scenarios (stable across NumPy versions)")
kd = RES["K_close"]["context"]["bootstrap"]["overlap_distribution"]; assert len([v for v in kd.values() if v]) >= 2, kd; ok("K_close: rank 5 and 6 nearly tied -> the bootstrap genuinely varies, so a wrong seed or method would be visible")
assert H.analyze(SCEN / "C_overlap2_new/decisions", BASELINE_INPUT, CANONICAL, bootstrap_b=300) == H.analyze(SCEN / "C_overlap2_new/decisions", BASELINE_INPUT, CANONICAL, bootstrap_b=300); ok("deterministic: same inputs give identical output")
rk = [x["field"] for x in RES["A_overlap5"]["citation_table"] if x["citations"] > 0]
eq(run("A_overlap5", a1a_ranking=rk)["context"]["spearman"]["value"], 1.0, "rho same"); eq(run("A_overlap5", a1a_ranking=rk[::-1])["context"]["spearman"]["value"], -1.0, "rho reversed")
assert RES["A_overlap5"]["context"]["spearman"]["value"] is None and "full" in RES["A_overlap5"]["context"]["spearman"]["note"]
ok("Spearman = 1.0 for identical order, -1.0 for reversed; reported as not computable (with the reason) when no full 1a ranking is given")
assert run("A_overlap5")["context"]["frozen_defaults_used"] is False and H.analyze(SCEN / "A_overlap5/decisions", BASELINE_INPUT, CANONICAL)["context"]["frozen_defaults_used"] is True; ok("flags when the bootstrap size differs from the frozen default")

print("T5  command line")
rep_path = SCEN / "C.json"
cp = subprocess.run([sys.executable, str(HERE / "h1_baseline_replication.py"), "--decisions_dir", str(SCEN / "C_overlap2_new/decisions"), "--baseline_input", str(BASELINE_INPUT),
                     "--canonical", str(CANONICAL), "--bootstrap", "200", "--out", str(rep_path)], capture_output=True, text=True)
assert cp.returncode == 0 and "FIRED" in cp.stdout and rep_path.exists(), cp.stderr[-400:]; ok("CLI runs, prints the report, writes the JSON (trigger FIRED for scenario C)")
cp = subprocess.run([sys.executable, str(HERE / "h1_baseline_replication.py"), "--decisions_dir", str(SCEN / "A_overlap5/decisions"), "--baseline_input", str(BASELINE_INPUT),
                     "--bootstrap", "100", "--out", str(SCEN / "A.json")], capture_output=True, text=True)
assert "not fired" in cp.stdout and "Nothing to generate" in cp.stdout; ok("CLI says 'not fired ... Nothing to generate' for scenario A")

print("T6  the frozen manifest")
m = RES["C_overlap2_new"]
assert H.verify_manifest(m); ok("verifies as written; its hash equals an independent recomputation")
t = json.loads(json.dumps(m)); t["decision"]["triggered"] = not t["decision"]["triggered"]; assert not H.verify_manifest(t); ok("editing any decision-bearing field breaks the hash")
t = json.loads(json.dumps(m)); t["context"]["bootstrap"]["B"] = 1; t["context"]["spearman"] = {"value": 0.99}; assert H.verify_manifest(t); ok("editing the context block does NOT break the hash (it is explicitly outside it)")
body = json.dumps({k: v for k, v in m.items() if k not in ("context", "decision_sha256")}, sort_keys=True)
assert str(WORK) not in body and str(SCEN) not in body and "/tmp" not in body; ok("no filesystem paths in the decision-bearing content")
cp_dir = WORK / "elsewhere" / "deeper"; (cp_dir / "bl").mkdir(parents=True); (cp_dir / "cn").mkdir(); shutil.copytree(SCEN / "C_overlap2_new/decisions", cp_dir / "dec_copy")
shutil.copy(BASELINE_INPUT, cp_dir / "bl" / BASELINE_INPUT.name); shutil.copy(CANONICAL, cp_dir / "cn" / CANONICAL.name)
m2 = H.analyze(cp_dir / "dec_copy", cp_dir / "bl" / BASELINE_INPUT.name, cp_dir / "cn" / CANONICAL.name, bootstrap_b=300)
eq(m2["decision_sha256"], m["decision_sha256"], "location independence"); ok("same inputs in a different directory tree -> identical decision_sha256 (machine independent)")
outs = []
for hs in ("1", "99991"):
    o = WORK / f"hs_{hs}.json"
    subprocess.run([sys.executable, str(HERE / "h1_baseline_replication.py"), "--decisions_dir", str(SCEN / "G_tie/decisions"), "--baseline_input", str(BASELINE_INPUT), "--canonical", str(CANONICAL),
                    "--bootstrap", "100", "--out", str(o)], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": hs}, check=True); outs.append(o.read_bytes())
assert outs[0] == outs[1]; ok("the tie scenario run under two different PYTHONHASHSEEDs writes byte-identical manifests")
frozen = WORK / "frozen.json"; args = [sys.executable, str(HERE / "h1_baseline_replication.py"), "--baseline_input", str(BASELINE_INPUT), "--canonical", str(CANONICAL), "--bootstrap", "100", "--out", str(frozen)]
cp1 = subprocess.run(args + ["--decisions_dir", str(SCEN / "C_overlap2_new/decisions")], capture_output=True, text=True); assert cp1.returncode == 0 and "Saved (frozen)" in cp1.stdout
first = frozen.read_bytes()
cp2 = subprocess.run(args + ["--decisions_dir", str(SCEN / "C_overlap2_new/decisions")], capture_output=True, text=True); assert cp2.returncode == 0 and "IDENTICAL" in cp2.stdout and frozen.read_bytes() == first
ok("write-once: re-running with the same inputs leaves the frozen manifest untouched")
cp3 = subprocess.run(args + ["--decisions_dir", str(SCEN / "A_overlap5/decisions")], capture_output=True, text=True)
assert cp3.returncode == 1 and "DIFFERENT decision content" in cp3.stdout and frozen.read_bytes() == first; ok("write-once: different inputs are REFUSED (the trigger cannot be re-evaluated); file unchanged")
tam = json.loads(first); tam["decision"]["triggered"] = False; frozen.write_text(json.dumps(tam))
cp4 = subprocess.run(args + ["--decisions_dir", str(SCEN / "C_overlap2_new/decisions")], capture_output=True, text=True)
assert cp4.returncode == 1 and "edited after it was frozen" in cp4.stdout; ok("a hand-edited frozen manifest is detected and refused")

# ============================================================================
print("\nGENERATOR")
G = SCEN / "gen_out"; shutil.copytree(OUT, G)
(G / "evaluation").mkdir(exist_ok=True)
def rehash(m): m["decision_sha256"] = H.manifest_decision_sha256(m); return m
json.dump(H.analyze(SCEN / "C_overlap2_new/decisions", BASELINE_INPUT, G / "ground_truth/canonical_customers.json", bootstrap_b=100), open(G / "evaluation/h1_replication.json", "w"))
json.dump(H.analyze(SCEN / "A_overlap5/decisions", BASELINE_INPUT, G / "ground_truth/canonical_customers.json", bootstrap_b=100), open(G / "evaluation/h1_A.json", "w"))
json.dump(H.analyze(SCEN / "D_overlap2_covered/decisions", BASELINE_INPUT, G / "ground_truth/canonical_customers.json", bootstrap_b=100), open(G / "evaluation/h1_D.json", "w"))
def gen(rep, out=G): return subprocess.run([sys.executable, str(HERE / "generate_h1_replication_conditions.py"), "--manifest", str(rep), "--out", str(out)], capture_output=True, text=True)
before = set(os.listdir(G / "conditions"))

cp = gen(G / "evaluation/h1_A.json"); assert cp.returncode == 0 and "Nothing to generate" in cp.stdout and set(os.listdir(G / "conditions")) == before; ok("G1 not triggered -> generates nothing")
cp = gen(G / "evaluation/h1_D.json"); assert cp.returncode == 0 and "nothing new" in cp.stdout and set(os.listdir(G / "conditions")) == before; ok("G2 triggered but all covered -> generates nothing")

cp = gen(G / "evaluation/h1_replication.json"); assert cp.returncode == 0, cp.stdout[-600:] + cp.stderr[-600:]
new = sorted(set(os.listdir(G / "conditions")) - before); eq(new, ["h1_replication_email_open_rate", "h1_replication_fraud_risk_score", "h1_replication_total_purchases"], "new dirs")
ok("G3 scenario C -> exactly the 3 new condition directories")
canon = {c["customer_id"]: c for c in json.load(open(G / "ground_truth/canonical_customers.json"))}
main_ids = {r["record_id"] for r in json.load(open(G / "conditions/h1_individual_nps_score/dither_reference.json"))}
for cid in new:
    field = cid.replace("h1_replication_", "")
    ref = json.load(open(G / "conditions" / cid / "dither_reference.json")); inp = [json.loads(l) for l in open(G / "conditions" / cid / "agent_input.jsonl")]
    eq({r["record_id"] for r in ref}, main_ids, f"{cid} record ids"); eq(len(inp), N, f"{cid} n")
    assert all("customer_id" not in r and not any(k.startswith("_dither") for k in r) for r in inp)
    changed = 0
    for r in ref:
        base = canon[r["customer_id"]]
        others = {k for k in base if k != field and r[k] != base[k]}
        assert not others, (cid, r["customer_id"], others)
        changed += (r[field] != base[field])
    assert changed / N >= 0.80, (cid, changed)
ok("G4 same 200 record_ids as the main conditions; agent files carry no customer_id or _dither fields; ONLY the target field differs from canonical, in >= 80% of records")

import dataclasses
from dither_engine import build_all_conditions
tmpl = next(c for c in build_all_conditions() if c.condition_id == "h1_individual_nps_score")
import generate_h1_replication_conditions as GEN
for e in json.load(open(G / "evaluation/h1_replication.json"))["decision"]["conditions_to_add"]:
    cfg = GEN.build_config(e)
    diff = {f.name for f in dataclasses.fields(tmpl) if getattr(cfg, f.name) != getattr(tmpl, f.name)}
    assert diff <= {"fields", "seed", "condition_id", "description"}, diff
    assert 500 <= cfg.seed <= 524 and cfg.seed == 500 + sorted(H.DITHERED_FIELDS).index(e["field"])
ok("G5 each new condition differs from h1_individual_nps_score ONLY in field, seed and id; seeds 500-524 are field-based")

snap = {p: p.read_bytes() for cid in new for p in (G / "conditions" / cid).iterdir()}
cp = gen(G / "evaluation/h1_replication.json"); assert cp.returncode == 0 and "unchanged" in cp.stdout
assert all(p.read_bytes() == b for p, b in snap.items()); ok("G6 re-running is idempotent: files reported unchanged and byte-identical")

p = G / "conditions/h1_replication_total_purchases/agent_input.jsonl"; saved = p.read_text(); p.write_text(saved + "\n")
cp = gen(G / "evaluation/h1_replication.json"); assert cp.returncode != 0 and "DIFFERENT content" in (cp.stdout + cp.stderr); p.write_text(saved)
ok("G7 refuses to overwrite a file whose content differs")

cp_path = G / "ground_truth/canonical_customers.json"; orig = cp_path.read_bytes(); data = json.loads(orig); data[0]["dob"] = "1900-01-01"; cp_path.write_text(json.dumps(data, indent=2))
shutil.rmtree(G / "conditions/h1_replication_fraud_risk_score"); cp = gen(G / "evaluation/h1_replication.json")
assert cp.returncode != 0 and "CHANGED since the baseline" in (cp.stdout + cp.stderr) and not (G / "conditions/h1_replication_fraud_risk_score").exists(); cp_path.write_bytes(orig)
ok("G8 a changed canonical file (the date-of-birth hazard) is refused before anything is written")

rep = json.load(open(G / "evaluation/h1_replication.json")); rep["decision"]["conditions_to_add"][0]["condition_id"] = "h1_individual_nps_score"
json.dump(rehash(rep), open(G / "evaluation/bad1.json", "w")); cp = gen(G / "evaluation/bad1.json"); assert cp.returncode != 0 and "collides" in (cp.stdout + cp.stderr); ok("G9 refuses a condition_id that collides with an existing condition")
rep = json.load(open(G / "evaluation/h1_replication.json")); rep["decision"]["conditions_to_add"][0]["field"] = "customer_segment"
json.dump(rehash(rep), open(G / "evaluation/bad2.json", "w")); cp = gen(G / "evaluation/bad2.json"); assert cp.returncode != 0 and "is not a field the engine can dither" in (cp.stdout + cp.stderr); ok("G10 refuses a field the engine cannot dither")
rep = json.load(open(G / "evaluation/h1_replication.json")); rep["inputs"]["canonical"] = None
json.dump(rehash(rep), open(G / "evaluation/nohash.json", "w")); cp = gen(G / "evaluation/nohash.json"); assert cp.returncode == 0 and "did not record" in cp.stdout; ok("G11 warns (but proceeds) when no canonical hash was recorded")
report = json.load(open(G / "validity_report_h1_replication.json")); assert report["n_failed"] == 0 and report["n_conditions"] == 3 and report["canonical_sha256"]; ok("G12 validity report written, 3 conditions, none failed")
rep = json.load(open(G / "evaluation/h1_replication.json")); rep["decision"]["conditions_to_add"][0]["field"] = "fraud_risk_score"   # edited WITHOUT rehashing
json.dump(rep, open(G / "evaluation/edited.json", "w")); cp = gen(G / "evaluation/edited.json"); assert cp.returncode != 0 and "edited after it was frozen" in (cp.stdout + cp.stderr); ok("G13 refuses a manifest edited after it was frozen")
cp = gen(G / "evaluation/h1_replication.json"); assert cp.returncode == 0
gr = json.load(open(G / "evaluation/h1_replication_generated.json")); eq(gr["manifest_decision_sha256"], json.load(open(G / "evaluation/h1_replication.json"))["decision_sha256"], "gen record manifest")
eq(sorted(gr["conditions"]), sorted(new), "gen record conditions")
for cid, h in gr["conditions"].items():
    eq(h["agent_input_sha256"], hashlib.sha256((G / "conditions" / cid / "agent_input.jsonl").read_bytes()).hexdigest(), f"{cid} input hash")
    eq(h["dither_reference_sha256"], hashlib.sha256((G / "conditions" / cid / "dither_reference.json").read_bytes()).hexdigest(), f"{cid} reference hash")
ok("G14 generation record lists the manifest hash and the true hash of every generated file")
print(f"\n{n_checks} check groups passed")

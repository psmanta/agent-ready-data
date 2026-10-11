#!/usr/bin/env python3
"""
Regression test for evaluate_h5.py and validate_h5_schema.py.

Self-contained: builds its own 200-customer dataset in a temporary directory and never touches your real
experiments_output. It plants reasoning texts and baseline runs with exactly known properties (keyword
phrases, which value-echo style each text uses, matching and non-matching baseline runs, judge labels, a
partially-exposed condition) and recomputes EVERY H5 number with plain loops: the three-way detection
tables, the clean-baseline floors, the echo classes, the Jaccard coherence and reasoning-inertia flag
(re-implemented from the documented tokenizer), the judge tables, the confidence-aware 'behavior changed' table, the >=3-text inertia rule, the completeness
refusal, the audit stamp and sample sizes, the explicit class-by-drift count matrices, blocked (selected but unable
to move) customers staying OUT of every table, and the redundancy grouping.

    python3 test_evaluate_h5.py
"""
import atexit, copy, hashlib, io, contextlib, json, os, random, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
from scipy.stats import wilcoxon
import evaluate_core as ec
import evaluate_h5 as E
import validate_h5_schema as V

WORK = Path(tempfile.mkdtemp(prefix="h5_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
OUT = WORK / "out"
_g = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "200", "--seed", "7", "--out", str(OUT)], capture_output=True, text=True)
assert _g.returncode == 0, _g.stdout[-400:] + _g.stderr[-400:]

CANON = {c["customer_id"]: c for c in json.load(open(OUT / "ground_truth/canonical_customers.json"))}
CUSTS = list(CANON)
DEC = ["HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"]
_r = random.Random(3)
def tier_of(i): return "deeply_boundary" if i < 12 else "lightly_boundary" if i < 18 else "unclassified" if i in (190, 191, 192) else "stable"   # 3 customers with an unrecognized tier: "all" only
GT = {c: {"customer_id": c, "stability_tier": tier_of(i), "final_decision": _r.choice(DEC), "decision_source": "primary_5run",
         "confidence_range": None if i % 9 == 0 else [0.70, 0.85]} for i, c in enumerate(CUSTS)}
json.dump(GT, open(WORK / "gt.json", "w"))
STRATUM = {c: ("stable" if tier_of(i) == "stable" else "unknown" if tier_of(i) == "unclassified" else "boundary") for i, c in enumerate(CUSTS)}
IDX = {c: i for i, c in enumerate(CUSTS)}
SPECIAL, SPECIAL2 = CUSTS[150], CUSTS[151]; SPECIALS = {SPECIAL, SPECIAL2}
# engineered so the dithered coherence lands EXACTLY on the boundary of the baseline range (see W3); SPECIAL has five
# matching baseline texts, SPECIAL2 has exactly two (and is "flat" even so: the inertia rule must still exclude it)
UNIQ = ["xenon", "yttrium", "zinc", "argon", "boron"]
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def eq(a, b, label): assert a == b, f"{label}: evaluator={a!r} independent={b!r}"

COND_FIELDS = ["h4_churn_risk_score_plausible", "h4_churn_risk_score_implausible", "h4_churn_risk_score_implausible_isolated",
               "h4_total_spend_implausible", "h4_tenure_months_implausible", "h1_individual_email", "h3_individual_refund_rate", "h1_individual_is_vip", "h1_category_identity"]
H4 = [c for c in COND_FIELDS if c.startswith("h4_")]
EXPECTED_CLASS = {"h4_churn_risk_score_plausible": "soft_only", "h4_churn_risk_score_implausible": "soft_only",
                  "h4_churn_risk_score_implausible_isolated": "strict", "h4_total_spend_implausible": "approximate",
                  "h4_tenure_months_implausible": "strict", "h1_individual_email": "strict", "h3_individual_refund_rate": "none", "h1_individual_is_vip": "bounded"}
P_DRIFT = {c: 0.25 + 0.05 * i for i, c in enumerate(COND_FIELDS)}
P_KW = {c: 0.10 + 0.05 * i for i, c in enumerate(COND_FIELDS)}
VOCAB = ["customer", "account", "engagement", "history", "recent", "purchases", "loyalty", "profile", "value", "retention",
         "priority", "review", "summary", "activity", "support", "standing", "segment", "growth", "balance", "relationship"]
STYLES = {"churn_risk_score": ["echo", "pct", "omit"], "total_spend": ["echo", "omit"], "tenure_months": ["echo", "conv", "omit"], "refund_rate": ["echo", "omit"]}
CLASS_OF_STYLE = {"echo": "echo", "pct": "echo_pct_marker", "conv": "converted", "omit": "omitted"}
KW = " This suggests a data quality issue."
md = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)

def mention(f, shown, style):
    if style == "omit" or style == "none": return ""
    return {"churn_risk_score": {"echo": f" churn risk score {shown}", "pct": f" churn risk score {shown}%"},
            "total_spend": {"echo": f" total spend {shown}"}, "refund_rate": {"echo": f" refund rate {shown}"},
            "tenure_months": {"echo": f" tenure {shown} months", "conv": f" tenure {shown / 12:.2f} years"}}[f][style]

# ------------------------------- plant the world -------------------------------
PLAN = {}          # (cond, customer) -> dict of what was planted
for cond in COND_FIELDS:
    ref = json.load(open(OUT / "conditions" / cond / "dither_reference.json"))
    rows = []
    for r in ref:
        c, fields, rng = r["customer_id"], r["_dither_fields"], random.Random(f"{cond}:{r['customer_id']}")
        drifted = rng.random() < P_DRIFT[cond]
        dec = rng.choice([d for d in DEC if d != GT[c]["final_decision"]]) if drifted else GT[c]["final_decision"]
        text = "customer review " + " ".join(rng.sample(VOCAB, 5)) + "."
        styles = {f: (rng.choice(STYLES[f]) if f in STYLES else "none") for f in fields}
        for f in fields: text += mention(f, r[f], styles[f])
        kw = rng.random() < P_KW[cond]
        if kw: text += KW
        conf = rng.choice([0.60, 0.70, 0.78, 0.85, 0.92])
        if cond == "h4_churn_risk_score_implausible" and c in SPECIALS:
            text, styles, kw = "alpha beta zeta", {f: ("omit" if f in STYLES else "none") for f in fields}, False
        PLAN[(cond, c)] = {"drift": drifted, "kw": kw, "styles": styles, "fields": fields, "text": text, "conf": conf}
        rows.append({"record_id": r["record_id"], "business_decision": dec, "agent_confidence": conf, "decision_reasoning": text, "key_factors": []})
    with open(OUT / "conditions" / cond / "decisions.jsonl", "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
# the premise of every keyword expectation: the phrase flags, nothing else in the planted vocabulary does
assert all(ec.detect_h5_keywords(p["text"])["detected"] == p["kw"] for p in PLAN.values()), "planted texts trip (or miss) the scan by accident"

BASE_FIELDS = {"churn_risk_score": "churn risk score", "total_spend": "total spend", "tenure_months": "tenure", "refund_rate": "refund rate"}
def includes(c, k): return c not in SPECIALS and ((md(c) % 2) + k) % 2 == 0
BASE = {"customers": {}}
for c in CUSTS:
    runs = []
    for k in range(5):
        rng = random.Random(f"base:{c}:{k}")
        text = "customer review " + " ".join(rng.sample(VOCAB, 5)) + "."
        if includes(c, k):
            text += "".join(f" {BASE_FIELDS[f]} {CANON[c][f]}" for f in BASE_FIELDS)
        if md(c) % 5 == 0 and k == 0: text += KW
        if c in SPECIALS: text = "alpha beta " + UNIQ[k]      # five baseline texts, every pair at Jaccard exactly 0.5
        decision = GT[c]["final_decision"]
        if IDX[c] % 7 == 0 and k == 4: decision = [d for d in DEC if d != decision][0]       # one non-matching run
        if IDX[c] % 13 == 0 and k != 0: decision = [d for d in DEC if d != decision][0]      # only one matching run -> skipped by the coherence analysis
        if c == SPECIAL2 and k >= 2: decision = [d for d in DEC if d != GT[c]["final_decision"]][0]
        if IDX[c] % 11 == 0 and k >= 2: decision = [d for d in DEC if d != GT[c]["final_decision"]][0]    # (usually) two matching runs -> coherence yes, inertia no
        runs.append({"business_decision": decision, "decision_reasoning": text, "key_factors": []})
    BASE["customers"][c] = {"run_details": runs}
json.dump(BASE, open(WORK / "baseline_reference.json", "w"))

JUDGE = WORK / "classifier_output"; JUDGE.mkdir()
def jcat(cond, c): return "explicit_concern" if PLAN[(cond, c)]["kw"] or md(cond + c) % 11 == 0 else "narrative_reframing" if md(cond + c) % 4 == 0 else "unremarked_usage"
for cond in H4:
    with open(JUDGE / f"{cond}.jsonl", "w") as f:
        for i, ((cn, c), p) in enumerate((k, v) for k, v in PLAN.items() if k[0] == cond):
            if i in (3, 17): f.write(json.dumps({"id": f"{c}:{p['fields'][0]}", "category": None, "error": "api_failure", "prompt_id": "v3-test"}) + "\n"); continue
            f.write(json.dumps({"id": f"{c}:{p['fields'][0]}", "category": jcat(cond, c), "confidence": "high", "figure_referenced": True, "evidence": "x", "rationale": "x",
                                "evidence_verbatim": True, "error": None, "prompt_id": "v3-test"}) + "\n")
def judged(cond, c): return [c2 for (cn, c2) in PLAN if cn == cond].index(c) not in (3, 17)

def run_eval(judge=True, ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return E.evaluate_h5(OUT, WORK / "gt.json", WORK / "baseline_reference.json", JUDGE if judge else None, ids, allow_incomplete=True)

# ------------------------------- independent expectations -------------------------------
def perturbed(cond): return [c for (cn, c) in PLAN if cn == cond and PLAN[(cn, c)]["fields"]]
def in_s(c, s): return True if s == "all" else STRATUM[c] == s
def conf_out(cond, c):
    rng = GT[c]["confidence_range"]
    return None if rng is None else bool(PLAN[(cond, c)]["conf"] < rng[0] or PLAN[(cond, c)]["conf"] > rng[1])
def wil(rate_k, n):
    from evaluate_core import wilson_interval
    return None if n == 0 else [rate_k / n, list(wilson_interval(rate_k, n))]

R = run_eval()
print("W1  the three-way detection tables (PRIMARY), exposure, and the clean-baseline floor")
for cond in COND_FIELDS:
    t = R["conditions"][cond]; pc = perturbed(cond)
    eq(t["exposure"], {"n_records": 200, "n_perturbed": len(pc), "exposure": len(pc) / 200}, f"{cond} exposure")
    for s in E.STRATA:
        cs = [c for c in pc if in_s(c, s)]; b = t["detection_crosstab"][s]
        dc = sum(PLAN[(cond, c)]["kw"] and PLAN[(cond, c)]["drift"] for c in cs); du = sum(PLAN[(cond, c)]["kw"] and not PLAN[(cond, c)]["drift"] for c in cs)
        nc = sum(not PLAN[(cond, c)]["kw"] and PLAN[(cond, c)]["drift"] for c in cs); nu = sum(not PLAN[(cond, c)]["kw"] and not PLAN[(cond, c)]["drift"] for c in cs)
        eq((b["n"], b["detected_changed"], b["detected_unchanged"], b["not_detected_changed"], b["not_detected_unchanged"]), (len(cs), dc, du, nc, nu), f"{cond}/{s} cells")
        eq(b["three_way_decision_only"], {"detected_and_changed": dc, "detected_no_change": du, "not_detected": nc + nu}, f"{cond}/{s} decision-only three-way")
        cf = {c: conf_out(cond, c) for c in cs}; ch = {c: PLAN[(cond, c)]["drift"] or cf[c] is True for c in cs}
        eq(b["three_way"], {"detected_and_changed": sum(PLAN[(cond, c)]["kw"] and ch[c] for c in cs), "detected_no_change": sum(PLAN[(cond, c)]["kw"] and not ch[c] for c in cs),
                            "not_detected": sum(not PLAN[(cond, c)]["kw"] for c in cs)}, f"{cond}/{s} three-way (decision OR confidence)")
        eq(b["confidence"], {"n_outside_baseline_range": sum(v is True for v in cf.values()), "n_unknown": sum(v is None for v in cf.values())}, f"{cond}/{s} confidence")
        assert b["keyword_detection"]["k"] == dc + du and b["keyword_detection"]["n"] == len(cs)
        if cs:
            lo, hi = ec.wilson_interval(dc + du, len(cs)); assert abs(b["keyword_detection"]["ci"][0] - lo) < 1e-12 and abs(b["keyword_detection"]["ci"][1] - hi) < 1e-12
        hits = sum((md(c) % 5 == 0 and c not in SPECIALS) for c in cs)       # one clean-baseline keyword hit (run 1) for these customers
        eq(b["clean_baseline_floor"], {"n_customers": len(cs), "n_runs": 5 * len(cs), "n_hits": hits, "hit_rate": hits / (5 * len(cs)) if cs else 0.0}, f"{cond}/{s} floor")
ok("every condition x stratum (3 x 9): the 2x2 cells, BOTH three-way tables (decision-only, and decision OR confidence-outside-baseline-range), the confidence counts, the Wilson interval and the clean floor match the independent recomputation")
allb = [R["conditions"][c]["detection_crosstab"]["all"] for c in COND_FIELDS]
assert any(b["three_way"]["detected_and_changed"] > b["three_way_decision_only"]["detected_and_changed"] for b in allb) and any(b["confidence"]["n_unknown"] > 0 for b in allb) and any(b["confidence"]["n_outside_baseline_range"] > 0 for b in allb)
ok("the scenario is non-trivial: confidence adds 'changed' cases the decision alone misses, some customers have no recorded range (counted as unknown), some do")
ev = R["conditions"]["h1_individual_is_vip"]; assert 0 < ev["exposure"]["n_perturbed"] < 100 and ev["detection_crosstab"]["all"]["n"] == ev["exposure"]["n_perturbed"]
assert all(ev["coherence"]["all"]["n"] == ev["exposure"]["n_perturbed"] for _ in [0])
ok(f"a mostly unperturbed condition (h1_individual_is_vip, {ev['exposure']['n_perturbed']} of 200 perturbed): every table covers ONLY the perturbed customers, and exposure is reported beside them")

print("W2  echo classes (exploratory), with their clean floors")
assert ec.classify_value_echo(-10, "tenure -0.83 years")["echo_class"] == "converted" and ec.classify_value_echo(-10, "tenure -0.8 years")["echo_class"] == "omitted"
ok("the documented specificity rule: a rounded conversion needs >= 2 significant digits (-0.83 years is converted; a bare -0.8 is scored omitted by design)")
STYLE_CLASS = lambda cond, c, f: CLASS_OF_STYLE.get(PLAN[(cond, c)]["styles"][f], "not_applicable")
for cond in COND_FIELDS:
    pc = perturbed(cond)
    for f in sorted({f for c in pc for f in PLAN[(cond, c)]["fields"]}):
        for s in E.STRATA:
            cs = [c for c in pc if f in PLAN[(cond, c)]["fields"] and in_s(c, s)]; b = R["conditions"][cond]["echo_classes"][f][s]
            eq(b["n"], len(cs), f"{cond}.{f}/{s} n"); eq(b["class_counts"], dict(sorted(Counter(STYLE_CLASS(cond, c, f) for c in cs).items())), f"{cond}.{f}/{s} classes")
            for cls, blk in b["drift_by_echo_class"].items():
                grp = [c for c in cs if STYLE_CLASS(cond, c, f) == cls]; eq((blk["n"], blk["k"]), (len(grp), sum(PLAN[(cond, c)]["drift"] for c in grp)), f"{cond}.{f}/{s} drift by {cls}")
            fl = Counter()
            for c in cs:
                for k in range(5): fl["echo" if (f in BASE_FIELDS and includes(c, k)) else "omitted" if f in BASE_FIELDS else "not_applicable"] += 1
            eq(b["clean_baseline_floor"]["class_counts"], dict(sorted(fl.items())), f"{cond}.{f}/{s} floor classes"); assert "customers_with_any_run" not in b["clean_baseline_floor"]
ok("class counts, drift by class and the clean-baseline floor classes match for every dithered field, condition and stratum (echo, % marker, /12 conversion, omitted, not applicable)")

print("W3  Jaccard coherence and reasoning inertia (re-implemented from the documented tokenizer)")
def tok(t): return {w for w in t.lower().replace(",", " ").replace(".", " ").split() if w not in ec.STOPWORDS and w.isalpha()}
def jac(a, b):
    A, B = tok(a), tok(b); U = A | B
    return 1.0 if not U else len(A & B) / len(U)
PER = {}
for cond in COND_FIELDS:
    pc = perturbed(cond)
    per = {}
    for c in pc:
        texts = [x["decision_reasoning"] for x in BASE["customers"][c]["run_details"] if x["business_decision"] == GT[c]["final_decision"]]
        if len(texts) < 2: continue
        d = sum(jac(PLAN[(cond, c)]["text"], t) for t in texts) / len(texts)
        b = [sum(jac(texts[i], texts[j]) for j in range(len(texts)) if j != i) / (len(texts) - 1) for i in range(len(texts))]
        pairs = [jac(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        per[c] = (d, sum(b) / len(b), min(pairs) <= d <= max(pairs), PLAN[(cond, c)]["drift"], len(texts))
    PER[cond] = per
    for s in E.STRATA:
        cs = [c for c in pc if in_s(c, s)]; el = [per[c] for c in cs if c in per]; iel = [e for e in el if e[4] >= 3]; blk = R["conditions"][cond]["coherence"][s]
        eq((blk["n"], blk["n_eligible"], blk["n_skipped"]), (len(cs), len(el), len(cs) - len(el)), f"{cond}/{s} eligibility")
        eq((blk["n_inertia_eligible"], blk["n_excluded_from_inertia_fewer_than_3_texts"]), (len(iel), len(el) - len(iel)), f"{cond}/{s} inertia eligibility (>= 3 texts)")
        eq((blk["flat_coherence"], blk["inertia"], blk["n_drifted_inertia_eligible"]), (sum(e[2] for e in iel), sum(e[2] and e[3] for e in iel), sum(e[3] for e in iel)), f"{cond}/{s} flat/inertia")
        if len(el) >= 2:
            nz = [e[0] - e[1] for e in el if e[0] - e[1] != 0]
            if nz: assert abs(blk["condition_level_shift"]["p_value"] - wilcoxon(nz).pvalue) < 1e-9, (cond, s)
            eq(blk["condition_level_shift"]["n_pairs"], len(el), f"{cond}/{s} n_pairs")
ok("per customer: eligibility, flat coherence, the inertia count (only customers with >= 3 matching baseline texts) and the Wilcoxon shift p-value match; customers with fewer than two texts are skipped, those with exactly two feed the coherence shift but are excluded from inertia, both counted")
sp = PER["h4_churn_risk_score_implausible"][SPECIAL]
assert sp[0] == 0.5 and sp[2] is True and sp[4] == 5, sp
sp2 = PER["h4_churn_risk_score_implausible"][SPECIAL2]
assert sp2[0] == 0.5 and sp2[2] is True and sp2[4] == 2, sp2
ok("the engineered boundary customer: dithered coherence is EXACTLY 0.5 and so is every baseline pair, so 'within range' must be INCLUSIVE to count it flat (an exclusive comparison would miss it); a second engineered customer has only TWO baseline texts yet is also exactly on the boundary: it must feed the coherence shift but NOT the inertia counts")
assert any(R["conditions"][c]["coherence"]["all"]["n_skipped"] > 0 for c in COND_FIELDS) and any(R["conditions"][c]["coherence"]["all"]["inertia"] > 0 for c in COND_FIELDS) and any(R["conditions"][c]["coherence"]["all"]["inertia"] < R["conditions"][c]["coherence"]["all"]["n_drifted_inertia_eligible"] for c in COND_FIELDS) and any(R["conditions"][c]["coherence"]["all"]["n_excluded_from_inertia_fewer_than_3_texts"] > 0 for c in COND_FIELDS)
ok("the scenario is non-trivial: skipped customers exist, customers with exactly two baseline texts exist, and inertia is neither always nor never")

print("W4  judge tables (provisional)")
for cond in COND_FIELDS:
    blk = R["conditions"][cond].get("judge")
    if cond not in H4: assert blk is None; continue
    for s in E.STRATA:
        cs = [c for c in perturbed(cond) if in_s(c, s) and judged(cond, c)]; b = blk[s]
        eq(b["n"], len(cs), f"{cond}/{s} judged n"); eq(b["category_counts"], dict(sorted(Counter(jcat(cond, c) for c in cs).items())), f"{cond}/{s} categories")
        eq(b["explicit_concern"]["k"], sum(jcat(cond, c) == "explicit_concern" for c in cs), f"{cond}/{s} explicit")
        for cat, x in b["drift_by_judge_category"].items():
            g = [c for c in cs if jcat(cond, c) == cat]; eq((x["n"], x["k"]), (len(g), sum(PLAN[(cond, c)]["drift"] for c in g)), f"{cond}/{s} drift by {cat}")
        mx = {}
        for c in cs:
            f = PLAN[(cond, c)]["fields"][0]; mx.setdefault(STYLE_CLASS(cond, c, f), Counter())[jcat(cond, c)] += 1
        eq(b["echo_by_judge"], {k: dict(sorted(v.items())) for k, v in sorted(mx.items())}, f"{cond}/{s} echo x judge")
ok("category counts, the explicit_concern rate, drift by category and the echo-by-judge matrix match; errored judge rows are dropped; non-H4 conditions carry no judge section")
R2 = run_eval(judge=False)
assert all("judge" not in t for t in R2["conditions"].values()) and R2["judge_records_joined"] == {}; ok("without --judge_dir there are no judge sections at all")

print("W5  redundancy classes and the negative controls (exploratory)")
rc = R["redundancy_classes"]
eq({r["condition_id"]: r["class"] for r in rc["per_condition"]}, EXPECTED_CLASS, "classes"); ok("each single-field condition gets the right class (churn: soft_only by default but strict when isolated; spend: approximate; tenure and email: strict; is_vip: bounded; refund_rate: none); the multi-field condition is excluded")
pool, grp = {}, {}
for cond, cls in EXPECTED_CLASS.items():
    cs = [c for c in perturbed(cond) if STRATUM[c] == "stable"]; k = sum(PLAN[(cond, c)]["kw"] for c in cs)
    for key in (cls, "has_partner" if cls in ("strict", "bounded", "approximate") else "no_partner"):
        p = pool.setdefault(key, [0, 0]); p[0] += k; p[1] += len(cs)
eq({k: (v["k"], v["n"]) for k, v in rc["pooled_stable"].items()}, {k: tuple(v) for k, v in pool.items()}, "pooled"); ok("pooled stable-stratum detection by class and by group (has_partner = strict/bounded/approximate; no_partner = soft_only/none) matches, with soft_only ALSO reported on its own (the owner's decision)")
assert "soft_only" in rc["pooled_stable"] and rc["grouping"]["soft_only"] == "also reported on its own" and "soft_only" not in rc["grouping"]["has_partner"]
eq([r["condition_id"] for r in rc["negative_controls"]], ["h3_individual_refund_rate"], "negative controls"); ok("negative controls = the no-link fields only (here refund_rate)")
assert "Descriptive only" in rc["caveat"]; ok("the non-independence caveat travels with the table")

print("W7  completeness: H5 reads EVERY dithered condition")
gt_p, base_p = WORK / "gt.json", WORK / "baseline_reference.json"
missing_expected = sorted(d.name for d in (OUT / "conditions").iterdir() if (d / "dither_reference.json").exists() and not (d / "decisions.jsonl").exists())
assert len(missing_expected) == 55 - len(COND_FIELDS), len(missing_expected)
try:
    with contextlib.redirect_stdout(io.StringIO()): E.evaluate_h5(OUT, gt_p, base_p, None)
    raise SystemExit("should have refused")
except SystemExit as e:
    assert "no decisions yet" in str(e) and str(len(missing_expected)) in str(e) and "--allow_incomplete" in str(e), str(e)[:200]
ok(f"with {len(missing_expected)} conditions lacking decisions the evaluator REFUSES (naming the count and the flag), as H1 and H4 do")
Rpart = E.evaluate_h5(OUT, gt_p, base_p, None, None, allow_incomplete=True)
eq(Rpart["conditions_missing_decisions"], missing_expected, "missing list"); eq(Rpart["n_conditions"], len(COND_FIELDS), "n analyzed"); ok("--allow_incomplete analyzes what exists and LISTS exactly what is missing")
Rsel = E.evaluate_h5(OUT, gt_p, base_p, None, COND_FIELDS[:2]); eq(Rsel["n_conditions"], 2, "selected"); eq(Rsel["conditions_missing_decisions"], [], "no missing when ids chosen explicitly"); ok("explicitly chosen conditions are analyzed without the refusal")
cp = subprocess.run([sys.executable, "evaluate_h5.py", "--experiments_output", str(OUT), "--finalized_ground_truth", str(gt_p), "--baseline_reference", str(base_p), "--out", str(WORK / "refused.json")], capture_output=True, text=True)
assert cp.returncode != 0 and "no decisions yet" in (cp.stdout + cp.stderr) and not (WORK / "refused.json").exists(); ok("the command line refuses too, and writes nothing")

print("W8  the audit stamp, per-condition sample sizes, and the explicit [class] x [drift] count matrices")
import classifier_prompt as CP
ast = R["audit_stamp"]
eq(ast["keyword_scan"], {"n_patterns": len(ec.H5_KEYWORD_PATTERNS), "sha256": hashlib.sha256("\n".join(ec.H5_KEYWORD_PATTERNS).encode()).hexdigest()}, "keyword stamp")
assert len(ast["keyword_scan"]["sha256"]) == 64 and sum("(?:ly)?" in p_ for p_ in ec.H5_KEYWORD_PATTERNS) >= 1; ok("the stamp carries the pattern count and the sha256 of the frozen keyword list (which includes the adverb-form patterns)")
eq(ast["echo_check"]["scale_factors"], [[n_, v_] for n_, v_ in ec.ECHO_SCALE_FACTORS], "echo factors"); ok("the stamp carries the frozen echo scale factors")
eq(ast["judge"], {"frozen_prompt_id": CP.PROMPT_ID, "prompt_ids_found": ["v3-test"], "conditions_judged": sorted(H4), "all_match_frozen": False}, "judge stamp")
ok("the judge stamp lists the frozen PROMPT_ID, the ids actually found on the judged conditions, which conditions were judged, and flags a mismatch (the planted files carry a test id)")
for cond in COND_FIELDS:
    pc = perturbed(cond); t = R["conditions"][cond]
    eq(t["sample_sizes"], {"total_n": 200, "n_perturbed": len(pc), "n_stable": sum(STRATUM[c] == "stable" for c in pc), "n_boundary": sum(STRATUM[c] == "boundary" for c in pc), "n_unclassified": sum(STRATUM[c] == "unknown" for c in pc)}, f"{cond} sizes")
    eq(t["judge_prompt_ids"], ["v3-test"] if cond in H4 else None, f"{cond} judge prompt ids")
assert any(R["conditions"][c]["sample_sizes"]["n_unclassified"] > 0 for c in COND_FIELDS)
ok("every condition reports total N, exposure-adjusted N_perturbed, N_stable, N_boundary and N_unclassified (customers with an unrecognized tier appear in 'all' only), and its judge prompt id (null where no judge ran)")
JUDGE2 = WORK / "classifier_output_frozen"; JUDGE2.mkdir()
for f_ in JUDGE.iterdir(): (JUDGE2 / f_.name).write_text(f_.read_text().replace('"v3-test"', json.dumps(CP.PROMPT_ID)))
R3 = E.evaluate_h5(OUT, WORK / "gt.json", WORK / "baseline_reference.json", JUDGE2, None, allow_incomplete=True)
eq(R3["audit_stamp"]["judge"]["all_match_frozen"], True, "match"); eq(R3["audit_stamp"]["judge"]["prompt_ids_found"], [CP.PROMPT_ID], "found"); ok("when the judge files carry the frozen PROMPT_ID the stamp says they match")
JUDGE3 = WORK / "classifier_output_unstamped"; JUDGE3.mkdir()
for f_ in JUDGE.iterdir(): (JUDGE3 / f_.name).write_text("".join(json.dumps({k_: v_ for k_, v_ in json.loads(l_).items() if k_ != "prompt_id"}) + "\n" for l_ in f_.read_text().splitlines()))
R4 = E.evaluate_h5(OUT, WORK / "gt.json", WORK / "baseline_reference.json", JUDGE3, None, allow_incomplete=True)
eq(R4["audit_stamp"]["judge"]["all_match_frozen"], False, "unstamped"); eq(R4["conditions"][H4[0]]["judge_prompt_ids"], [], "empty ids"); ok("judge output that carries NO prompt id is flagged as not matching (never silently accepted)")
eq(R2["audit_stamp"]["judge"], {"frozen_prompt_id": CP.PROMPT_ID, "prompt_ids_found": [], "conditions_judged": [], "all_match_frozen": None}, "no-judge stamp"); assert all(t["judge_prompt_ids"] is None for t in R2["conditions"].values()); ok("with no judge run: every judge_prompt_ids is null and all_match_frozen is null")
for cond in COND_FIELDS:
    pc = perturbed(cond); t = R["conditions"][cond]
    for s_ in E.STRATA:
        cs = [c for c in pc if in_s(c, s_)]; b = t["detection_crosstab"][s_]
        dc = sum(PLAN[(cond, c)]["kw"] and PLAN[(cond, c)]["drift"] for c in cs); du = sum(PLAN[(cond, c)]["kw"] and not PLAN[(cond, c)]["drift"] for c in cs)
        nc = sum(not PLAN[(cond, c)]["kw"] and PLAN[(cond, c)]["drift"] for c in cs); nu = sum(not PLAN[(cond, c)]["kw"] and not PLAN[(cond, c)]["drift"] for c in cs)
        eq(b["detection_by_drift_matrix"], {"detected": {"not_drifted": du, "drifted": dc}, "not_detected": {"not_drifted": nu, "drifted": nc}}, f"{cond}/{s_} detection matrix")
        for f_ in sorted({f for c in pc for f in PLAN[(cond, c)]["fields"]}):
            cf = [c for c in cs if f_ in PLAN[(cond, c)]["fields"]]; exp = {}
            for c in cf:
                m_ = exp.setdefault(STYLE_CLASS(cond, c, f_), {"not_drifted": 0, "drifted": 0}); m_["drifted" if PLAN[(cond, c)]["drift"] else "not_drifted"] += 1
            eq(t["echo_classes"][f_][s_]["echo_by_drift_matrix"], dict(sorted(exp.items())) if False else exp, f"{cond}.{f_}/{s_} echo matrix")
        if cond in H4:
            cj = [c for c in cs if judged(cond, c)]; exp = {}
            for c in cj:
                m_ = exp.setdefault(jcat(cond, c), {"not_drifted": 0, "drifted": 0}); m_["drifted" if PLAN[(cond, c)]["drift"] else "not_drifted"] += 1
            eq(t["judge"][s_]["judge_by_drift_matrix"], exp, f"{cond}/{s_} judge matrix")
ok("explicit [echo class] x [drift 0/1], [judge category] x [drift 0/1] and [keyword detection] x [drift 0/1] count matrices match the independent recomputation for every condition, field and stratum (including stable)")

print("W9  blocked customers are NOT perturbed: selected for dithering but unable to move (e.g. payment_failures = 0 drawn downward)")
pf = "h3_individual_payment_failures"; ref = json.load(open(OUT / "conditions" / pf / "dither_reference.json"))
with open(OUT / "conditions" / pf / "decisions.jsonl", "w") as f_:
    for r_ in ref: f_.write(json.dumps({"record_id": r_["record_id"], "business_decision": GT[r_["customer_id"]]["final_decision"], "agent_confidence": 0.8, "decision_reasoning": "customer review alpha beta.", "key_factors": []}) + "\n")
changed = [r_ for r_ in ref if r_["_dither_fields"]]; blocked = [r_ for r_ in ref if r_["_dither_blocked"] and not r_["_dither_fields"]]; applied_only = [r_ for r_ in ref if r_["_dither_applied"] and not r_["_dither_fields"]]
assert len(blocked) > 0 and len(applied_only) >= len(blocked) and 0 < len(changed) < 200, (len(blocked), len(applied_only), len(changed))
assert all(r_["_dither_original"][f] != r_[f] for r_ in changed for f in r_["_dither_fields"]); ok(f"in the generated data every record the engine lists as perturbed really changed ({len(changed)} of 200), while {len(blocked)} were selected but blocked (and {len(applied_only)} carry _dither_applied=True without any change: that flag must NOT be used)")
Rb = E.evaluate_h5(OUT, WORK / "gt.json", WORK / "baseline_reference.json", None, [pf]); tb = Rb["conditions"][pf]
eq(tb["exposure"], {"n_records": 200, "n_perturbed": len(changed), "exposure": len(changed) / 200}, "pf exposure")
eq((tb["detection_crosstab"]["all"]["n"], tb["sample_sizes"]["n_perturbed"], tb["coherence"]["all"]["n"], tb["echo_classes"]["payment_failures"]["all"]["n"]), (len(changed),) * 4, "pf table sizes")
ok("every H5 table for that condition covers only the customers whose value actually changed: the blocked ones appear in no table, so they cannot inflate 'omitted' or 'not detected'")

print("W10 interval checks tolerate float rounding but still catch real errors (Wilson(0, 183) has a lower bound of 1.7e-18, above the rate 0.0)")
import validate_h4_schema as V4
from refine_boundary_convergence import wilson_interval as _wi
for k_, n_ in ((0, 183), (183, 183), (0, 10), (7, 183), (0, 200), (200, 200)):
    lo_, hi_ = _wi(k_, n_); node_ = {"k": k_, "n": n_, "rate": k_ / n_, "ci": [lo_, hi_]}
    i4, i5 = [], []; V4._walk(dict(node_), "x", i4); V._rate_block(dict(node_), "x", i5)
    assert not i4 and not i5, (k_, n_, i4, i5)
ok("zero-detection and all-detection cells (k = 0 and k = n, at several n) pass the H4 and H5 validators: a real run WILL contain such cells")
bad_ = {"k": 5, "n": 10, "rate": 0.5, "ci": [0.6, 0.9]}; i4, i5 = [], []; V4._walk(dict(bad_), "x", i4); V._rate_block(dict(bad_), "x", i5)
assert i4 and i5; ok("an interval that genuinely excludes its rate is still caught by both")

print("W6  labels, validator, command line")
eq(R["status_by_section"], E.STATUS, "status"); assert [k for k, v in R["status_by_section"].items() if v == "primary"] == ["detection_crosstab"]; ok("only the detection cross-tab is labeled primary")
good, issues = V.validate_h5_results_schema(R); assert good, issues[:5]; ok("the validator accepts the full output")
good, issues = V.validate_h5_results_schema(R2); assert good, issues[:5]; ok("and the output without judge sections")
outp = WORK / "h5.json"
cp = subprocess.run([sys.executable, "evaluate_h5.py", "--experiments_output", str(OUT), "--finalized_ground_truth", str(WORK / "gt.json"),
                     "--baseline_reference", str(WORK / "baseline_reference.json"), "--judge_dir", str(JUDGE), "--allow_incomplete", "--out", str(outp)], capture_output=True, text=True)
assert cp.returncode == 0 and outp.exists(), cp.stdout[-300:] + cp.stderr[-500:]
cv = subprocess.run([sys.executable, "validate_h5_schema.py", str(outp)], capture_output=True, text=True); assert cv.returncode == 0 and "PASSED" in cv.stdout, cv.stdout[-700:]; ok("the CLI writes the file and the validator CLI passes it")
def corrupt(label, fn, needle):
    r = copy.deepcopy(R); fn(r); good, iss = V.validate_h5_results_schema(r)
    assert (not good) and any(needle in i for i in iss), (label, iss[:3]); print(f"  ✅ validator catches: {label}")
cond0 = COND_FIELDS[1]
corrupt("the 2x2 no longer sums to n", lambda r: r["conditions"][cond0]["detection_crosstab"]["stable"].__setitem__("detected_changed", r["conditions"][cond0]["detection_crosstab"]["stable"]["detected_changed"] + 1), "the 2x2 sums to n")
corrupt("an interval that excludes its rate", lambda r: r["conditions"][cond0]["detection_crosstab"]["all"]["keyword_detection"].__setitem__("ci", [0.99, 1.0]), "ci is [lo, hi]")
corrupt("floor hits above floor runs", lambda r: r["conditions"][cond0]["detection_crosstab"]["all"]["clean_baseline_floor"].__setitem__("n_hits", 10 ** 6), "floor hits <= runs")
corrupt("echo class counts that do not sum to n", lambda r: r["conditions"][cond0]["echo_classes"]["churn_risk_score"]["all"]["class_counts"].__setitem__("echo", 999), "class counts sum to n")
corrupt("inertia above the flat count", lambda r: r["conditions"][cond0]["coherence"]["all"].__setitem__("inertia", 10 ** 5), "inertia <= flat")
corrupt("a judge category count that does not sum", lambda r: r["conditions"][cond0]["judge"]["all"]["category_counts"].__setitem__("unremarked_usage", 10 ** 5), "category counts sum to n")
corrupt("a mislabeled section (echo classes called primary)", lambda r: r["status_by_section"].__setitem__("echo_classes", "primary"), "pre-registered ones")
corrupt("pooled redundancy counts that disagree with the conditions", lambda r: r["redundancy_classes"]["pooled_stable"]["has_partner"].__setitem__("n", r["redundancy_classes"]["pooled_stable"]["has_partner"]["n"] + 5), "has_partner + no_partner n")
corrupt("adding confidence SHRINKS 'detected and changed'", lambda r: r["conditions"][cond0]["detection_crosstab"]["stable"]["three_way"].__setitem__("detected_and_changed", r["conditions"][cond0]["detection_crosstab"]["stable"]["three_way_decision_only"]["detected_and_changed"] - 1), "adding confidence can only grow")
corrupt("the confidence-aware three-way table no longer sums to n", lambda r: r["conditions"][cond0]["detection_crosstab"]["all"]["three_way"].__setitem__("not_detected", r["conditions"][cond0]["detection_crosstab"]["all"]["three_way"]["not_detected"] + 1), "three_way sums to n")
corrupt("inertia-eligible plus excluded no longer equals eligible", lambda r: r["conditions"][cond0]["coherence"]["all"].__setitem__("n_excluded_from_inertia_fewer_than_3_texts", r["conditions"][cond0]["coherence"]["all"]["n_excluded_from_inertia_fewer_than_3_texts"] + 3), "inertia-eligible + excluded == eligible")
corrupt("more confidence-outside customers than customers", lambda r: r["conditions"][cond0]["detection_crosstab"]["all"]["confidence"].__setitem__("n_outside_baseline_range", 10 ** 6), "confidence counts")
corrupt("the missing-conditions list removed", lambda r: r.pop("conditions_missing_decisions"), "conditions_missing_decisions present")
corrupt("sample sizes that do not add up", lambda r: r["conditions"][cond0]["sample_sizes"].__setitem__("n_stable", r["conditions"][cond0]["sample_sizes"]["n_stable"] + 4), "sample_sizes stable + boundary + unclassified")
corrupt("a judge prompt id on a condition that has no judge section", lambda r: r["conditions"]["h1_individual_email"].__setitem__("judge_prompt_ids", ["v3-x"]), "judge_prompt_ids is null exactly when")
corrupt("an echo-by-drift matrix that disagrees with its rate table", lambda r: r["conditions"][cond0]["echo_classes"]["churn_risk_score"]["stable"]["echo_by_drift_matrix"].__setitem__("echo", {"not_drifted": 0, "drifted": 999}), "echo-by-drift matrix equals its rate table")
corrupt("a judge-by-drift matrix that disagrees with its rate table", lambda r: r["conditions"][cond0]["judge"]["stable"]["judge_by_drift_matrix"].__setitem__("unremarked_usage", {"not_drifted": 1, "drifted": 1}), "judge-by-drift matrix equals its rate table")
corrupt("a keyword hash that is not a hash", lambda r: r["audit_stamp"]["keyword_scan"].__setitem__("sha256", "not-a-hash"), "64-hex sha256")
corrupt("a stamp that claims the judge prompt ids all match when they do not", lambda r: r["audit_stamp"]["judge"].__setitem__("all_match_frozen", True), "all_match_frozen is consistent")
corrupt("the audit stamp removed", lambda r: r.pop("audit_stamp"), "audit_stamp present")
print(f"\n{n_checks} check groups passed")

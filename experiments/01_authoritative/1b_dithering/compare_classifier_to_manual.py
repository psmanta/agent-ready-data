#!/usr/bin/env python3
"""
Proof-of-concept comparison: does the classifier's call match our own
manual reading, for the subset of the 19 B/C smoke-test customers we
actually classified by hand? NOT a validation test -- that still
requires the real human-audit sample, per the locked sequencing. This
is a cheap early-warning check only.

Honest about scope: only 7 of the 19 qualifying customers were given an
explicit manual label (4 narrative reframing, 3 reformatted-not-
reexamined), straight from the chat discussion where we built the B/C
taxonomy. The other 12 are reported as classifier output only, with NO
comparison, since we never classified them ourselves.

Re-derives the same 19 qualifying customers using the EXACT same
drifted + zero-keyword-hit filter as analyze_h5_bc_smoke.py, rather than
hardcoding a customer list that could silently drift out of sync with
that script's own logic.

Run from experiments/01_authoritative/1b_dithering/, after running the
classifier against all 3 implausible conditions' decisions:
  python3 compare_classifier_to_manual.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from evaluate_core import detect_h5_keywords

# Our own manual labels, straight from the chat discussion where these
# were read by hand and sorted into the B/C taxonomy. Condition is
# included explicitly because some customer_ids appear in more than one
# of the 3 conditions (dithered on different fields) -- the label only
# applies to the SPECIFIC condition named, not the customer in general.
MANUAL_LABELS = {
    ("h4_total_spend_implausible", "CUST_000010"): "narrative_reframing",      # supplies a cause: refund/credit
    ("h4_churn_risk_score_implausible", "CUST_000024"): "narrative_reframing", # negative figure presented as reassuring
    ("h4_churn_risk_score_implausible", "CUST_000026"): "narrative_reframing", # negative figure presented as reassuring
    ("h4_tenure_months_implausible", "CUST_000007"): "narrative_reframing",    # see BORDERLINE below
    # Corrected: the agent echoed (003, 022) or added a % to (009) the value it
    # was shown; there is no reformatting-with-comment to classify, and a blind
    # judge cannot know a field's normal format. Under the three-category
    # judge these are all unremarked_usage; the echo check handles the % / scale.
    ("h4_churn_risk_score_implausible", "CUST_000009"): "unremarked_usage",
    ("h4_churn_risk_score_implausible", "CUST_000022"): "unremarked_usage",
    ("h4_churn_risk_score_implausible", "CUST_000003"): "unremarked_usage",
}

# Labels that rely on knowledge the blind judge does not have. CUST_000007's
# text says "long-tenured customer (47.5 years)": "long-tenured" fits that
# number at face value, and we labeled it reframing because WE know 47.5 years
# is impossible for this field -- the text alone does not show it. A fair
# label for a blind judge must be derivable from the text alone, so this one
# is reported but flagged, and a miss here is not evidence against the judge.
BORDERLINE_BY_TEXT_ALONE = {("h4_tenure_months_implausible", "CUST_000007")}

CONDITIONS = ["h4_churn_risk_score_implausible", "h4_total_spend_implausible",
              "h4_tenure_months_implausible"]
FIELD_BY_CONDITION = {
    "h4_churn_risk_score_implausible": "churn_risk_score",
    "h4_total_spend_implausible": "total_spend",
    "h4_tenure_months_implausible": "tenure_months",
}

with open("experiments_output/baseline/baseline_reference.json") as f:
    baseline = json.load(f)

def matching_baseline_texts(customer_id, majority_decision):
    entry = baseline["customers"].get(customer_id)
    if entry is None:
        return []
    return [r["decision_reasoning"] for r in entry["run_details"]
            if r["business_decision"] == majority_decision and r.get("decision_reasoning")]

# Step 1: re-derive the same 19 qualifying (condition, customer_id) pairs
qualifying = []
for cid in CONDITIONS:
    cond_dir = Path(f"experiments_output/conditions/{cid}")
    with open(cond_dir / "dither_reference.json") as f:
        dither_ref = {r["customer_id"]: r for r in json.load(f)}
    with open(cond_dir / "decisions.jsonl") as f:
        decisions = {json.loads(l)["record_id"]: json.loads(l) for l in f}

    for customer_id, ref in dither_ref.items():
        decision = decisions.get(ref["record_id"])
        if decision is None:
            continue
        baseline_entry = baseline["customers"].get(customer_id)
        if baseline_entry is None:
            continue
        majority = baseline_entry["majority_decision"]
        if decision["business_decision"] == majority:
            continue
        if detect_h5_keywords(decision["decision_reasoning"])["detected"]:
            continue
        if len(matching_baseline_texts(customer_id, majority)) < 2:
            continue
        qualifying.append((cid, customer_id))

print(f"Re-derived {len(qualifying)} qualifying customers (expect 19, matching the original smoke test).\n")

# Step 2: load classifier output for each condition
classifier_results = {}
for cid in CONDITIONS:
    out_path = Path(f"classifier_output/{cid}.jsonl")
    if not out_path.exists():
        print(f"⚠️  Missing classifier_output/{cid}.jsonl -- run the classifier on this condition first.")
        continue
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            classifier_results[r["id"]] = r

# Step 3: split into labeled (compare) vs unlabeled (report only)
print("=" * 70)
print("LABELED SET -- direct comparison against our own manual read")
print("=" * 70)
matches, mismatches, missing = 0, 0, 0
for cid, customer_id in qualifying:
    label = MANUAL_LABELS.get((cid, customer_id))
    if label is None:
        continue
    field = FIELD_BY_CONDITION[cid]
    classifier_id = f"{customer_id}:{field}"
    result = classifier_results.get(classifier_id)
    if result is None:
        print(f"  [{cid}] {customer_id}: manual={label}  classifier=MISSING (not yet classified)")
        missing += 1
        continue
    called = result.get("category")
    match = called == label
    matches += match
    mismatches += not match
    flag = "✅ MATCH" if match else "❌ MISMATCH"
    if (cid, customer_id) in BORDERLINE_BY_TEXT_ALONE:
        flag += "  (borderline: label relies on knowledge the text does not show)"
    print(f"  [{cid}] {customer_id}")
    print(f"    manual={label}   classifier={called}   {flag}")
    print(f"    classifier rationale: {result.get('rationale')!r}")
    print(f"    evidence quoted: {result.get('evidence')!r}  (verbatim in text: {result.get('evidence_verbatim')})")

print()
print(f"Labeled-set result: {matches}/{matches+mismatches} match "
      f"({missing} not yet classified)")
print()
print("=" * 70)
print("UNLABELED SET -- classifier output only, NO manual comparison exists")
print("=" * 70)
for cid, customer_id in qualifying:
    if (cid, customer_id) in MANUAL_LABELS:
        continue
    field = FIELD_BY_CONDITION[cid]
    classifier_id = f"{customer_id}:{field}"
    result = classifier_results.get(classifier_id)
    if result is None:
        print(f"  [{cid}] {customer_id}: MISSING")
        continue
    print(f"  [{cid}] {customer_id}: classifier={result.get('category')}  "
          f"({result.get('rationale')!r})")

print()
print("=" * 70)
print("Reminder: this is a proof-of-concept check on 7 manually-labeled")
print("examples (4 we picked specifically because they were the clearest")
print("confabulation cases, 3 because they were the clearest scale-")
print("insensitive cases) -- NOT the real validation, which still needs")
print("the human-audited zero-hit sample before any full-scale decision.")

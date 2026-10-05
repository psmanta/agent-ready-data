#!/usr/bin/env python3
"""
H5 severity-descriptor check: does the agent's qualitative severity word
(low/moderate/high/significant/critical/etc.) stay consistent between a
customer's baseline reasoning and their dithered reasoning, for the SAME
19 (drifted, zero-keyword-hit) customers the Case B/C smoke test found?

Proper version: checks baseline INTERNAL CONSENSUS first (does the
customer's own 5 baseline runs agree with each other on severity
language?) before checking whether the dithered text overlaps with that
consensus -- same noise-floor principle as the Jaccard baseline check.
Operates directly on the real JSON files, not manually retyped text --
avoids the exact transcription-error risk that showed up when doing this
by hand in chat.

Run from experiments/01_authoritative/1b_dithering/, after the Case B/C
smoke test has already been run (reuses its exact same data, zero new
API cost):
  python3 analyze_h5_severity.py
"""
import json, re
from pathlib import Path

SEVERITY_WORDS = [
    "minimal", "low", "modest", "moderate", "elevated", "significant",
    "substantial", "high", "severe", "critical", "urgent", "serious",
    "concerning", "immediate",
]
_COMPILED = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in SEVERITY_WORDS]

def extract_severity(text):
    if not text:
        return set()
    return {w for w, pat in zip(SEVERITY_WORDS, _COMPILED) if pat.search(text)}

CONDITIONS = ["h4_churn_risk_score_implausible", "h4_total_spend_implausible",
              "h4_tenure_months_implausible"]

with open("experiments_output/baseline/baseline_reference.json") as f:
    baseline = json.load(f)

def matching_baseline_texts(customer_id, majority_decision):
    entry = baseline["customers"].get(customer_id)
    if entry is None:
        return []
    return [r["decision_reasoning"] for r in entry["run_details"]
            if r["business_decision"] == majority_decision and r.get("decision_reasoning")]

# Reuse EXACTLY the same keyword-scan + drift logic as the Case B/C
# smoke test, so this checks the identical 19 customers, not a
# re-derived subset
import sys
sys.path.insert(0, ".")
from evaluate_core import detect_h5_keywords

results = []
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
            continue  # not drifted
        if detect_h5_keywords(decision["decision_reasoning"])["detected"]:
            continue  # explicit detection, not the bucket we're checking

        matching_texts = matching_baseline_texts(customer_id, majority)
        if len(matching_texts) < 2:
            continue

        baseline_sets = [extract_severity(t) for t in matching_texts]
        baseline_consensus = set.intersection(*baseline_sets)
        baseline_union = set.union(*baseline_sets)
        baseline_agrees_internally = len(baseline_consensus) > 0

        dithered_set = extract_severity(decision["decision_reasoning"])
        overlaps_consensus = bool(dithered_set & baseline_consensus) if baseline_consensus else None
        overlaps_any = bool(dithered_set & baseline_union)

        results.append({
            "condition": cid, "customer_id": customer_id,
            "baseline_consensus": sorted(baseline_consensus),
            "baseline_internally_consistent": baseline_agrees_internally,
            "dithered_severity": sorted(dithered_set),
            "overlaps_consensus": overlaps_consensus,
            "overlaps_any_baseline_run": overlaps_any,
        })

print(f"Found {len(results)} qualifying customers.\n")
n_consistent_baseline = sum(1 for r in results if r["baseline_internally_consistent"])
print(f"Customers whose OWN baseline runs agree on severity language: {n_consistent_baseline}/{len(results)}")
print(f"(this is the noise-floor check -- only among these is 'overlaps_consensus' meaningful)\n")

print("=" * 70)
for r in results:
    print(f"\n[{r['condition']}] {r['customer_id']}")
    print(f"  baseline consensus severity words: {r['baseline_consensus']}  "
          f"(internally consistent: {r['baseline_internally_consistent']})")
    print(f"  dithered severity words: {r['dithered_severity']}")
    print(f"  overlaps baseline CONSENSUS: {r['overlaps_consensus']}   "
          f"overlaps ANY baseline run: {r['overlaps_any_baseline_run']}")

n_overlap = sum(1 for r in results if r["overlaps_consensus"])
n_no_overlap = sum(1 for r in results if r["overlaps_consensus"] is False)
n_na = sum(1 for r in results if r["overlaps_consensus"] is None)
print(f"\n{'=' * 70}")
print(f"Overlaps baseline consensus (consistent with scale-insensitive absorption): {n_overlap}")
print(f"No overlap (consistent with confabulation): {n_no_overlap}")
print(f"N/A -- baseline itself had no internal severity consensus: {n_na}")

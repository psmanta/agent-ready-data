#!/usr/bin/env python3
"""
H5 smoke test analysis: for every customer who drifted on an implausible
H4 condition AND showed zero keyword-list hits, surface their Jaccard-to-
baseline score alongside the actual reasoning texts -- checking whether
low Jaccard (novel language) and high/flat Jaccard (unchanged boilerplate)
visibly separate Case B (confabulation) from Case C (silent absorption).

Deliberately NOT a "don't peek at substance" smoke test like Tier 1/2 --
the whole point here is reading a small number of real texts, since this
is testing whether a QUALITATIVE distinction is visible through a
QUANTITATIVE proxy. Treat results as exploratory at this n, not proof.
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, ".")
from evaluate_core import detect_h5_keywords, mean_pairwise_jaccard

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

qualifying = []

for cid in CONDITIONS:
    cond_dir = Path(f"experiments_output/conditions/{cid}")
    with open(cond_dir / "dither_reference.json") as f:
        dither_ref = {r["customer_id"]: r for r in json.load(f)}
    with open(cond_dir / "decisions.jsonl") as f:
        decisions = {json.loads(l)["record_id"]: json.loads(l) for l in f}

    for customer_id, ref in dither_ref.items():
        record_id = ref["record_id"]
        decision = decisions.get(record_id)
        if decision is None:
            continue
        baseline_entry = baseline["customers"].get(customer_id)
        if baseline_entry is None:
            continue
        majority = baseline_entry["majority_decision"]
        drifted = decision["business_decision"] != majority
        if not drifted:
            continue

        kw = detect_h5_keywords(decision["decision_reasoning"])
        if kw["detected"]:
            continue  # explicit detection -- Case A territory, not what we're probing

        matching_texts = matching_baseline_texts(customer_id, majority)
        if len(matching_texts) < 2:
            continue  # need at least 2 for a meaningful self-similarity range

        dithered_score = mean_pairwise_jaccard(decision["decision_reasoning"], matching_texts)
        # Exclude by POSITION, not value -- if 2+ baseline texts are
        # byte-identical (a real outcome at temperature=0, confirmed in
        # this exact run), value-based exclusion wrongly empties the
        # comparison set. See evaluate_h3.py's identical fix.
        self_sim_scores = [mean_pairwise_jaccard(matching_texts[i], matching_texts[:i] + matching_texts[i+1:])
                           for i in range(len(matching_texts))]
        self_sim_scores = [s for s in self_sim_scores if s is not None]

        qualifying.append({
            "condition": cid, "customer_id": customer_id,
            "dithered_jaccard": dithered_score,
            "baseline_self_sim_range": (min(self_sim_scores), max(self_sim_scores)) if self_sim_scores else None,
            "baseline_sample": matching_texts[0],
            "dithered_reasoning": decision["decision_reasoning"],
        })

print(f"Found {len(qualifying)} qualifying customers (drifted, zero keyword hits) "
      f"across {len(CONDITIONS)} conditions at this n.\n")
print("=" * 70)
for q in sorted(qualifying, key=lambda x: x["dithered_jaccard"] or 0):
    lo, hi = q["baseline_self_sim_range"] or (None, None)
    if lo is None:
        flag = "? (no valid baseline self-similarity comparison available)"
        range_str = "[n/a]"
    elif q["dithered_jaccard"] < lo:
        flag = "LOW (below baseline range) -- consistent with Case B, novel/confabulated language"
        range_str = f"[{lo:.3f}, {hi:.3f}]"
    elif q["dithered_jaccard"] > hi:
        flag = "FLAT-OR-ABOVE baseline range -- consistent with Case C, unchanged boilerplate"
        range_str = f"[{lo:.3f}, {hi:.3f}]"
    else:
        flag = "FLAT (within baseline range) -- consistent with Case C, unchanged boilerplate"
        range_str = f"[{lo:.3f}, {hi:.3f}]"
    print(f"\n[{q['condition']}] {q['customer_id']}")
    print(f"  Jaccard-to-baseline: {q['dithered_jaccard']:.3f}   "
          f"baseline self-similarity range: {range_str}   -> {flag}")
    print(f"  BASELINE sample:  {q['baseline_sample']!r}")
    print(f"  DITHERED reasoning: {q['dithered_reasoning']!r}")
print("\n" + "=" * 70)
print("Read each pair above: does LOW Jaccard correspond to novel/confabulated")
print("language (Case B), and FLAT Jaccard to unchanged boilerplate (Case C)?")
print("This is exploratory at this sample size, not a confirmed finding.")

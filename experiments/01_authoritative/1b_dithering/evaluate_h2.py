#!/usr/bin/env python3
"""
Evaluate H2 — Magnitude Effects
====================================
The Agentic Data Contract · Pillar 1: Authoritative · Experiment 1b

Tests whether decision drift scales with dither magnitude, across a
deliberate 3-field spread (churn_risk_score — H4 top-5 anchor,
total_spend — important but not self-reported, tenure_months — expected
lower decision weight), each dithered at 5%, 15%, 40%, and 100%
magnitude.

METHODOLOGY, per 1b_DESIGN_AMENDMENT_1.md and the statistical
methodology note:

1. Cochran's Q (omnibus gate) across all 4 magnitude levels per field —
   NOT full pairwise McNemar's directly. The ladder is ordered (unlike
   H1's unordered field set), so an omnibus test followed by targeted
   adjacent-step follow-ups is both more rigorous (controls family-wise
   error) and more interpretable (localizes WHERE a change occurs).

2. Adjacent-step McNemar's (5->15, 15->40, 40->100) run ONLY if
   Cochran's Q is significant — same conditional discipline as H8b's
   generation rule. If Cochran's Q is not significant, the null finding
   is reported directly; no fishing in post-hoc tests the gate didn't
   earn.

3. Extreme comparison (5% vs. 100%) reported separately as a labeled
   "total range effect" anchor — a single supplementary number for
   stakeholder communication, not part of the same test family and not
   subject to the same multiple-comparisons correction, since it makes
   a different kind of claim (total range, not localized change).

4. Spearman's rank correlation on the 4-point drift-rate curve —
   explicitly DESCRIPTIVE, not confirmatory (see
   descriptive_trend_correlation()'s own attached caveat).

5. Within-segment breakdown for churn_risk_score and total_spend ONLY
   (not tenure_months) — separates "the field intrinsically matters"
   from "the field is a proxy for segment membership," per the
   amendment's analysis enrichment note.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_core import (
    load_condition, compute_drift_rate, mcnemar_paired_test,
    cochrans_q_test, align_drift_by_customer, align_drift_by_customer_multi,
    descriptive_trend_correlation,
)


MAGNITUDE_LEVELS = [5, 15, 40, 100]  # percent
FIELDS = ["churn_risk_score", "total_spend", "tenure_months"]
WITHIN_SEGMENT_FIELDS = {"churn_risk_score", "total_spend"}  # NOT tenure_months


def condition_id_for(field: str, magnitude_pct: int) -> str:
    return f"h2_{field}_mag{magnitude_pct}pct"


def load_all_h2_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Load all 12 H2 conditions, organized as {field: {magnitude_pct: records}}
    for easy per-field ladder access.
    """
    conditions_dir = experiments_output_dir / "conditions"
    loaded: Dict[str, Dict[int, List[Dict[str, Any]]]] = {f: {} for f in FIELDS}

    for field in FIELDS:
        for mag in MAGNITUDE_LEVELS:
            cid = condition_id_for(field, mag)
            cond_dir = conditions_dir / cid
            loaded[field][mag] = load_condition(cond_dir, ground_truth)
            print(f"  Loaded {cid}: {len(loaded[field][mag])} records")

    return loaded


def analyze_field_ladder(
    field: str,
    magnitude_records: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Full magnitude-ladder analysis for one field: Cochran's Q omnibus
    gate, conditional adjacent-step McNemar's, extreme-comparison anchor,
    and the descriptive Spearman's trend curve.
    """
    drift_rates = {mag: compute_drift_rate(magnitude_records[mag]) for mag in MAGNITUDE_LEVELS}

    # --- Cochran's Q omnibus gate ---
    ordered_records = [magnitude_records[mag] for mag in MAGNITUDE_LEVELS]
    binary_matrix, shared_customers = align_drift_by_customer_multi(ordered_records)
    omnibus = cochrans_q_test(binary_matrix)
    omnibus_significant = omnibus["p_value"] is not None and omnibus["p_value"] < 0.05

    # --- Conditional adjacent-step McNemar's ---
    adjacent_results = None
    if omnibus_significant:
        adjacent_results = []
        for i in range(len(MAGNITUDE_LEVELS) - 1):
            mag_a, mag_b = MAGNITUDE_LEVELS[i], MAGNITUDE_LEVELS[i + 1]
            drift_a, drift_b, n_shared = align_drift_by_customer(
                magnitude_records[mag_a], magnitude_records[mag_b])
            mc = mcnemar_paired_test(drift_a, drift_b)
            adjacent_results.append({
                "step": f"{mag_a}pct -> {mag_b}pct",
                "drift_rate_a": round(drift_rates[mag_a], 4),
                "drift_rate_b": round(drift_rates[mag_b], 4),
                "n_shared_customers": n_shared,
                **mc,
            })

    # --- Extreme comparison (5% vs 100%) — separate labeled anchor ---
    drift_5, drift_100, n_shared_extreme = align_drift_by_customer(
        magnitude_records[5], magnitude_records[100])
    extreme_comparison = {
        "drift_rate_5pct":   round(drift_rates[5], 4),
        "drift_rate_100pct": round(drift_rates[100], 4),
        "total_range_effect": round(drift_rates[100] - drift_rates[5], 4),
        "n_shared_customers": n_shared_extreme,
        **mcnemar_paired_test(drift_5, drift_100),
        "note": "Supplementary anchor for stakeholder communication — not "
                "part of the omnibus/adjacent test family, not subject to "
                "the same multiple-comparisons correction, since it makes "
                "a different claim (total range) than a localized change.",
    }

    # --- Descriptive trend curve ---
    trend = descriptive_trend_correlation(
        MAGNITUDE_LEVELS, [drift_rates[m] for m in MAGNITUDE_LEVELS])

    return {
        "field": field,
        "drift_rates_by_magnitude": {str(m): round(drift_rates[m], 4) for m in MAGNITUDE_LEVELS},
        "cochrans_q_omnibus": omnibus,
        "omnibus_significant": omnibus_significant,
        "adjacent_step_mcnemar": adjacent_results,
        "adjacent_tests_note": (
            "Adjacent-step tests run and reported below."
            if omnibus_significant else
            "Cochran's Q was not significant — no adjacent-step tests run. "
            "This null finding is reported as complete on its own, not as "
            "grounds to search for a localized effect the omnibus gate "
            "didn't detect."
        ),
        "extreme_comparison_5v100": extreme_comparison,
        "descriptive_trend": trend,
    }


def within_segment_breakdown(
    magnitude_records: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Drift rate computed separately per customer_segment, at each
    magnitude level — separates "the field intrinsically matters" from
    "the field is a proxy for segment membership." Only called for
    churn_risk_score and total_spend, per the amendment's scope.
    """
    result = {}
    for mag in MAGNITUDE_LEVELS:
        records = magnitude_records[mag]
        by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_segment.setdefault(r["customer_segment"], []).append(r)
        result[str(mag)] = {
            segment: round(compute_drift_rate(recs), 4)
            for segment, recs in by_segment.items()
        }
    return result


def evaluate_h2(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
) -> Dict[str, Any]:
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    print("Loading all 12 H2 conditions...")
    loaded = load_all_h2_conditions(experiments_output_dir, ground_truth)

    results = {"hypothesis": "H2", "fields": {}}

    for field in FIELDS:
        print(f"\nAnalyzing magnitude ladder for {field}...")
        field_result = analyze_field_ladder(field, loaded[field])

        if field in WITHIN_SEGMENT_FIELDS:
            print(f"  Computing within-segment breakdown for {field}...")
            field_result["within_segment_breakdown"] = within_segment_breakdown(loaded[field])
        else:
            field_result["within_segment_breakdown"] = None

        results["fields"][field] = field_result

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_output", type=str, default="experiments_output")
    parser.add_argument("--finalized_ground_truth", type=str,
        default="experiments_output/finalized_ground_truth.json")
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h2_results.json")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("H2 Evaluation — Magnitude Effects")
    print(f"{'='*60}\n")

    results = evaluate_h2(Path(args.experiments_output), Path(args.finalized_ground_truth))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

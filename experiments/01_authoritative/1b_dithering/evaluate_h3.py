#!/usr/bin/env python3
"""
Evaluate H3 — Internal Consistency
=======================================
The Agentic Data Contract · Pillar 1: Authoritative · Experiment 1b

Does dithering a correlated field set coherently (matching its real
relationship) produce different decision drift, confidence, or reasoning
coherence than dithering the SAME fields independently (breaking the
relationship) — beyond what's explained by field-specific leverage or
perturbation volume alone?

Five analyses, per 1b_DESIGN_AMENDMENT_1.md:

1. Core drift comparison — McNemar's, correlated vs. uncorrelated, per set.
2. Confidence — full-population unconditional Wilcoxon (primary) plus the
   "always-drifters" principal stratum Wilcoxon (secondary, explicitly
   scoped). See the amendment for why a mixed-effects interaction model
   was considered and declined.
3. Jaccard reasoning coherence — condition-level shift vs. each
   customer's own baseline wobble (jaccard_condition_level_shift),
   computed per condition, compared correlated vs. uncorrelated per set.
4. Binary difference-in-differences via GEE — does decorrelating a real
   set cost more drift than decorrelating its volume-matched reference?
   4 comparisons: 3 real pairs vs. the 2-field reference, the triplet vs.
   the 3-field reference. Headline finding is whether all four show a
   larger cost than their reference, not any single p-value.
5. Free 2x2 directionality — attribute_by_field() on each real set's
   uncorrelated arm (where fields can move independently; the correlated
   arm's coupling means this breakdown is expected to be dominated by
   "all fields together," reported for contrast, not as the main signal).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_core import (
    load_condition, compute_drift_rate, compute_effective_drift_rate,
    attribute_by_field, mcnemar_paired_test, wilcoxon_signed_rank_test,
    jaccard_condition_level_shift, mean_pairwise_jaccard, binary_did_gee,
    align_drift_by_customer, align_values_by_customer,
)

# Where each H3 field's OWN individual condition lives, at matched 15%
# magnitude -- most already exist in H1/H2, only payment_failures is new
# to H3. Needed to complete the 2x2 directionality analysis: comparing a
# field's "moved alone within the pair" drift rate (from attribute_by_field
# on the pair's own uncorrelated condition) against its TRUE-isolation
# individual drift rate answers whether a field behaves the same when it
# happens to move alone within a joint dither as it does when dithered by
# itself -- this was scoped in the amendment's "complete the 2x2" section
# but the individual conditions were being loaded without ever being used.
FIELD_TO_INDIVIDUAL_CONDITION = {
    "churn_risk_score":        "h1_individual_churn_risk_score",
    "nps_score":               "h1_individual_nps_score",
    "total_spend":             "h2_total_spend_mag15pct",
    "lifetime_value_estimate": "h1_individual_lifetime_value_estimate",
    "support_tickets_open":    "h1_individual_support_tickets_open",
    "payment_failures":        "h3_individual_payment_failures",
}


SETS = {
    "pair1_churn_nps":       {"fields": ["churn_risk_score", "nps_score"],
                              "reference": "reference_pair", "kind": "pair"},
    "pair2_spend_ltv":       {"fields": ["total_spend", "lifetime_value_estimate"],
                              "reference": "reference_pair", "kind": "pair"},
    "pair3_support_payment": {"fields": ["support_tickets_open", "payment_failures"],
                              "reference": "reference_pair", "kind": "pair"},
    "triplet":               {"fields": ["total_spend", "support_tickets_open", "churn_risk_score"],
                              "reference": "reference_triplet", "kind": "triplet"},
    "reference_pair":        {"fields": ["avg_resolution_time_hours", "refund_rate"],
                              "reference": None, "kind": "reference"},
    "reference_triplet":     {"fields": ["avg_resolution_time_hours", "refund_rate", "tenure_months"],
                              "reference": None, "kind": "reference"},
}
REAL_SETS = [name for name, s in SETS.items() if s["kind"] != "reference"]


def load_all_h3_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads all 15 H3 conditions: 6 sets x 2 arms, plus 3 individual fields."""
    conditions_dir = experiments_output_dir / "conditions"
    ids = []
    for name in SETS:
        ids += [f"h3_{name}_correlated", f"h3_{name}_uncorrelated"]
    ids += ["h3_individual_avg_resolution_time_hours", "h3_individual_refund_rate"]
    ids += sorted(set(FIELD_TO_INDIVIDUAL_CONDITION.values()))

    loaded = {}
    for cid in ids:
        loaded[cid] = load_condition(conditions_dir / cid, ground_truth)
        print(f"  Loaded {cid}: {len(loaded[cid])} records")
    return loaded


# ============================================================================
# 1. CORE DRIFT COMPARISON
# ============================================================================

def core_drift_comparison(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """McNemar's, correlated vs. uncorrelated, per set. This is the base
    H3 finding: does decorrelation change drift for THIS set at all,
    same-population paired comparison."""
    results = {}
    for name in SETS:
        corr = loaded[f"h3_{name}_correlated"]
        uncorr = loaded[f"h3_{name}_uncorrelated"]
        drift_corr, drift_uncorr, n_shared = align_drift_by_customer(corr, uncorr)
        results[name] = {
            "drift_rate_correlated":   round(compute_drift_rate(corr), 4),
            "drift_rate_uncorrelated": round(compute_drift_rate(uncorr), 4),
            "n_shared_customers":      n_shared,
            **mcnemar_paired_test(drift_corr, drift_uncorr),
        }
    return results


# ============================================================================
# 2. CONFIDENCE METHODOLOGY
# ============================================================================

def confidence_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Primary: full-population unconditional paired Wilcoxon on confidence
    (uncorrelated minus correlated), regardless of drift status.
    Secondary: same test, restricted to customers who drifted under BOTH
    arms (the "always-drifters" principal stratum) — explicitly scoped
    to that subpopulation, not customers in general.
    """
    results = {}
    for name in SETS:
        corr = loaded[f"h3_{name}_correlated"]
        uncorr = loaded[f"h3_{name}_uncorrelated"]

        conf_corr, conf_uncorr, ids = align_values_by_customer(corr, uncorr, "dithered_confidence")
        diffs = [u - c for u, c in zip(conf_uncorr, conf_corr)]
        primary = wilcoxon_signed_rank_test(diffs)
        primary["interpretation"] = (
            "positive median_diff means confidence is HIGHER under "
            "uncorrelated dithering; negative means lower. Unconditional "
            "on drift status -- see amendment for why."
        )

        drift_corr_map = {r["customer_id"]: r["drifted"] for r in corr}
        drift_uncorr_map = {r["customer_id"]: r["drifted"] for r in uncorr}
        always_drifted = [c for c in ids if drift_corr_map.get(c) and drift_uncorr_map.get(c)]
        idx = {c: i for i, c in enumerate(ids)}
        stratum_diffs = [diffs[idx[c]] for c in always_drifted]
        secondary = wilcoxon_signed_rank_test(stratum_diffs)
        secondary["n_always_drifted"] = len(always_drifted)
        secondary["scope_note"] = (
            "Describes ONLY the always-drifters subpopulation (customers "
            "who drifted under BOTH arms) -- not customers in general. "
            "Holds customer-level susceptibility constant by construction."
        )

        results[name] = {"primary_unconditional": primary, "secondary_always_drifters": secondary}
    return results


# ============================================================================
# 3. JACCARD REASONING COHERENCE
# ============================================================================

def jaccard_coherence_analysis(
    loaded: Dict[str, List[Dict[str, Any]]],
    baseline_reference_path: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Per condition (each of the 12 set x arm conditions): condition-level
    Jaccard coherence shift vs. each customer's own baseline wobble.
    Reported per set as correlated vs. uncorrelated, so a reader can see
    whether uncorrelated dithering degrades reasoning coherence more than
    correlated dithering for the same fields.
    """
    with open(baseline_reference_path) as f:
        baseline = json.load(f)

    def matching_baseline_texts(customer_id: str) -> List[str]:
        entry = baseline["customers"].get(customer_id)
        if entry is None:
            return []
        final_decision = ground_truth[customer_id]["final_decision"]
        return [r["decision_reasoning"] for r in entry["run_details"]
                if r["business_decision"] == final_decision and r.get("decision_reasoning")]

    def condition_shift(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        dithered_coh, baseline_coh = [], []
        for r in records:
            texts = matching_baseline_texts(r["customer_id"])
            if len(texts) < 2 or not r["dithered_reasoning"]:
                continue
            d_score = mean_pairwise_jaccard(r["dithered_reasoning"], texts)
            # Exclude by POSITION, not by value -- found via H5's smoke test,
            # 2026-08. If 2+ of a customer's matching baseline texts are
            # byte-identical (a real, valid outcome at temperature=0, not
            # an error), excluding by value ("!= t") removes every copy,
            # not just the current one -- if ALL texts are identical, this
            # leaves an empty comparison set for every single one, making
            # mean_pairwise_jaccard return None for every element. The old
            # check (`not b_scores`) tested whether the LIST was empty, not
            # whether it was full of Nones -- a list of [None, None, None]
            # is truthy, so execution continued straight into sum() on
            # None values and crashed. Confirmed this is not a rare edge
            # case: it fired in a real 30-customer smoke test.
            b_scores = [mean_pairwise_jaccard(texts[i], texts[:i] + texts[i+1:])
                       for i in range(len(texts))]
            b_scores = [s for s in b_scores if s is not None]
            if d_score is None or not b_scores:
                continue
            dithered_coh.append(d_score)
            baseline_coh.append(sum(b_scores) / len(b_scores))
        if len(dithered_coh) < 2:
            return {"rho": None, "p_value": None, "n_pairs": 0,
                    "note": "Insufficient customers with >=2 matching baseline texts"}
        return jaccard_condition_level_shift(dithered_coh, baseline_coh)

    results = {}
    for name in SETS:
        results[name] = {
            "correlated":   condition_shift(loaded[f"h3_{name}_correlated"]),
            "uncorrelated": condition_shift(loaded[f"h3_{name}_uncorrelated"]),
        }
    return results


# ============================================================================
# 4. BINARY DIFFERENCE-IN-DIFFERENCES VIA GEE
# ============================================================================

def did_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    4 comparisons: each real pair vs. the 2-field reference, the triplet
    vs. the 3-field reference (matched perturbation volume). Headline
    finding is whether all four consistently show a larger decorrelation
    cost than their reference, not any single p-value in isolation.
    """
    results = {}
    larger_than_reference_count = 0
    for name in REAL_SETS:
        ref_name = SETS[name]["reference"]
        rows = []
        for arm, is_unc in [("correlated", 0), ("uncorrelated", 1)]:
            for r in loaded[f"h3_{name}_{arm}"]:
                rows.append({"customer_id": r["customer_id"], "pair_type": name,
                            "is_uncorrelated": is_unc, "drift": int(r["drifted"])})
            for r in loaded[f"h3_{ref_name}_{arm}"]:
                rows.append({"customer_id": r["customer_id"], "pair_type": ref_name,
                            "is_uncorrelated": is_unc, "drift": int(r["drifted"])})
        result = binary_did_gee(rows, treatment_group=name, reference_group=ref_name)
        if result.get("raw_did", 0) > 0:
            larger_than_reference_count += 1
        results[name] = result

    return {
        "comparisons": results,
        "n_sets_tested": len(REAL_SETS),
        "n_larger_than_reference": larger_than_reference_count,
        "consistency_note": (
            f"{larger_than_reference_count}/{len(REAL_SETS)} sets showed a larger "
            f"decorrelation cost than their volume-matched reference. Headline "
            f"finding is this consistency, not any single comparison's p-value."
        ),
    }


# ============================================================================
# 5. FREE 2x2 DIRECTIONALITY
# ============================================================================

def directionality_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Two parts per real set.

    1. attribute_by_field() on the uncorrelated arm (fields can move
       independently there) and correlated arm (reported for contrast --
       coupling means this is expected to be dominated by "all fields
       together," itself a check that coupling worked as designed).

    2. Each field's "moved alone within this pair" drift rate (the
       attribute_by_field subset where only this field changed) against
       its TRUE-ISOLATION individual drift rate from H1/H2/H3 -- completes
       the 2x2 directionality the amendment scoped, using individual
       conditions that were being loaded without ever being compared
       against anything.
    """
    results = {}
    for name in REAL_SETS:
        uncorr_attr = attribute_by_field(loaded[f"h3_{name}_uncorrelated"])
        corr_attr = attribute_by_field(loaded[f"h3_{name}_correlated"])

        individual_comparison = {}
        for field in SETS[name]["fields"]:
            individual_cid = FIELD_TO_INDIVIDUAL_CONDITION.get(field)
            if individual_cid is None or individual_cid not in loaded:
                continue
            individual_rate = compute_effective_drift_rate(loaded[individual_cid], field=field)
            individual_comparison[field] = {
                "moved_alone_within_pair_drift_rate": uncorr_attr["by_field"].get(field, {}).get("drift_rate"),
                "moved_alone_within_pair_n":           uncorr_attr["by_field"].get(field, {}).get("n"),
                "true_isolation_effective_drift_rate": round(individual_rate["effective_drift_rate"], 4)
                    if individual_rate["effective_drift_rate"] is not None else None,
                "true_isolation_n_perturbed":          individual_rate["n_perturbed"],
                "individual_condition_id":             individual_cid,
            }

        results[name] = {
            "uncorrelated": uncorr_attr,
            "correlated":   corr_attr,
            "individual_field_comparison": individual_comparison,
        }
    return results


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def evaluate_h3(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
    baseline_reference_path: Path,
) -> Dict[str, Any]:
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    print("Loading all 15 H3 conditions...")
    loaded = load_all_h3_conditions(experiments_output_dir, ground_truth)

    print("\n1. Core drift comparison (McNemar's, correlated vs. uncorrelated)...")
    drift = core_drift_comparison(loaded)

    print("2. Confidence analysis (primary unconditional + secondary always-drifters)...")
    confidence = confidence_analysis(loaded)

    print("3. Jaccard reasoning coherence...")
    jaccard = jaccard_coherence_analysis(loaded, baseline_reference_path, ground_truth)

    print("4. Binary difference-in-differences via GEE...")
    did = did_analysis(loaded)

    print("5. Free 2x2 directionality...")
    directionality = directionality_analysis(loaded)

    return {
        "hypothesis": "H3",
        "core_drift_comparison": drift,
        "confidence_analysis": confidence,
        "jaccard_coherence": jaccard,
        "difference_in_differences": did,
        "directionality": directionality,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_output", type=str, default="experiments_output")
    parser.add_argument("--finalized_ground_truth", type=str,
        default="experiments_output/finalized_ground_truth.json")
    parser.add_argument("--baseline_reference", type=str,
        default="experiments_output/baseline/baseline_reference.json")
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h3_results.json")
    args = parser.parse_args()

    print(f"\n{'='*60}\nH3 Evaluation — Internal Consistency\n{'='*60}\n")
    results = evaluate_h3(Path(args.experiments_output), Path(args.finalized_ground_truth),
                          Path(args.baseline_reference))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}\nSaved: {out_path}\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

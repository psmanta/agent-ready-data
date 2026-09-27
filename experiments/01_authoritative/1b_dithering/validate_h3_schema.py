#!/usr/bin/env python3
"""
Structural validator for h3_results.json — deliberately blind to
substantive interpretation. Checks that every expected key exists, every
value has the right type, and every numeric value sits in a valid range
(a p-value in [0,1], a drift rate in [0,1], an odds ratio > 0, and so on).

This exists specifically so a smoke-test run's output can be assessed
without looking at what the numbers actually say. Forming an impression
from an underpowered pilot run risks anchoring how the real run gets
interpreted later — the same reasoning that keeps interim clinical trial
data from being read as a result. Every check below reports PASS/FAIL
and, on failure, which field and which structural rule it violated — it
never prints an actual value.
"""

from typing import Any, Dict, List, Tuple


def _check(label: str, condition: bool, issues: List[str]) -> None:
    if not condition:
        issues.append(label)


def _valid_prob_or_none(x: Any) -> bool:
    return x is None or (isinstance(x, (int, float)) and 0.0 <= x <= 1.0)


def _valid_rate(x: Any) -> bool:
    return isinstance(x, (int, float)) and 0.0 <= x <= 1.0


def validate_h3_results_schema(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    SETS = ["pair1_churn_nps", "pair2_spend_ltv", "pair3_support_payment",
            "triplet", "reference_pair", "reference_triplet"]
    REAL_SETS = SETS[:4]

    _check("top-level keys present",
           {"core_drift_comparison", "confidence_analysis", "jaccard_coherence",
            "difference_in_differences", "directionality"} <= set(results),
           issues)

    # --- 1. core_drift_comparison ---
    cdc = results.get("core_drift_comparison", {})
    for name in SETS:
        d = cdc.get(name)
        _check(f"core_drift_comparison.{name}: present", d is not None, issues)
        if d is None:
            continue
        _check(f"core_drift_comparison.{name}.drift_rate_correlated: valid rate [0,1]",
               _valid_rate(d.get("drift_rate_correlated")), issues)
        _check(f"core_drift_comparison.{name}.drift_rate_uncorrelated: valid rate [0,1]",
               _valid_rate(d.get("drift_rate_uncorrelated")), issues)
        _check(f"core_drift_comparison.{name}.n_shared_customers: positive int",
               isinstance(d.get("n_shared_customers"), int) and d["n_shared_customers"] > 0, issues)
        _check(f"core_drift_comparison.{name}.p_value: valid prob or None",
               _valid_prob_or_none(d.get("p_value")), issues)
        _check(f"core_drift_comparison.{name}.b_x_only/c_y_only: non-negative ints",
               isinstance(d.get("b_x_only"), int) and d["b_x_only"] >= 0
               and isinstance(d.get("c_y_only"), int) and d["c_y_only"] >= 0, issues)

    # --- 2. confidence_analysis ---
    ca = results.get("confidence_analysis", {})
    for name in SETS:
        d = ca.get(name)
        _check(f"confidence_analysis.{name}: present", d is not None, issues)
        if d is None:
            continue
        for key in ["primary_unconditional", "secondary_always_drifters"]:
            sub = d.get(key)
            _check(f"confidence_analysis.{name}.{key}: present", sub is not None, issues)
            if sub is None:
                continue
            _check(f"confidence_analysis.{name}.{key}.p_value: valid prob or None",
                   _valid_prob_or_none(sub.get("p_value")), issues)
            _check(f"confidence_analysis.{name}.{key}.median_diff: numeric or None (bounded [-1,1])",
                   sub.get("median_diff") is None or
                   (isinstance(sub.get("median_diff"), (int, float)) and -1.0 <= sub["median_diff"] <= 1.0),
                   issues)
        n_always = d.get("secondary_always_drifters", {}).get("n_always_drifted")
        _check(f"confidence_analysis.{name}.secondary.n_always_drifted: non-negative int",
               isinstance(n_always, int) and n_always >= 0, issues)

    # --- 3. jaccard_coherence ---
    jc = results.get("jaccard_coherence", {})
    for name in SETS:
        d = jc.get(name)
        _check(f"jaccard_coherence.{name}: present", d is not None, issues)
        if d is None:
            continue
        for arm in ["correlated", "uncorrelated"]:
            sub = d.get(arm)
            _check(f"jaccard_coherence.{name}.{arm}: present", sub is not None, issues)
            if sub is not None:
                _check(f"jaccard_coherence.{name}.{arm}.p_value: valid prob or None",
                       _valid_prob_or_none(sub.get("p_value")), issues)

    # --- 4. difference_in_differences ---
    did = results.get("difference_in_differences", {})
    comparisons = did.get("comparisons", {})
    _check("difference_in_differences: exactly 4 comparisons",
           len(comparisons) == 4, issues)
    for name in REAL_SETS:
        d = comparisons.get(name)
        _check(f"difference_in_differences.{name}: present", d is not None, issues)
        if d is None:
            continue
        _check(f"difference_in_differences.{name}.odds_ratio: positive or None",
               d.get("odds_ratio") is None or (isinstance(d["odds_ratio"], (int, float)) and d["odds_ratio"] > 0),
               issues)
        _check(f"difference_in_differences.{name}.p_value: valid prob or None",
               _valid_prob_or_none(d.get("p_value")), issues)
        ci = d.get("odds_ratio_95ci")
        _check(f"difference_in_differences.{name}.odds_ratio_95ci: 2-element, low<=high",
               isinstance(ci, list) and len(ci) == 2 and ci[0] <= ci[1] if ci else True, issues)
        _check(f"difference_in_differences.{name}.n_customers: positive int",
               isinstance(d.get("n_customers"), int) and d["n_customers"] > 0, issues)
    _check("difference_in_differences.n_larger_than_reference: int in [0,4]",
           isinstance(did.get("n_larger_than_reference"), int) and 0 <= did["n_larger_than_reference"] <= 4,
           issues)

    # --- 5. directionality ---
    dir_ = results.get("directionality", {})
    for name in REAL_SETS:
        d = dir_.get(name)
        _check(f"directionality.{name}: present", d is not None, issues)
        if d is None:
            continue
        for arm in ["correlated", "uncorrelated"]:
            _check(f"directionality.{name}.{arm}.by_field: present",
                   "by_field" in d.get(arm, {}), issues)
        ifc = d.get("individual_field_comparison", {})
        for field, comp in ifc.items():
            _check(f"directionality.{name}.individual_field_comparison.{field}: has both drift-rate keys",
                   "moved_alone_within_pair_drift_rate" in comp and "true_isolation_effective_drift_rate" in comp,
                   issues)

    return (len(issues) == 0, issues)


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=str)
    args = parser.parse_args()

    with open(args.results_path) as f:
        results = json.load(f)

    passed, issues = validate_h3_results_schema(results)
    total_checks = "many"  # deliberately not counted precisely -- not the point
    if passed:
        print(f"✅ STRUCTURAL VALIDATION PASSED — schema, types, and ranges all correct.")
        print(f"   (No substantive values inspected or reported by design.)")
    else:
        print(f"❌ STRUCTURAL VALIDATION FAILED — {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Structural validator for h4_results.json — deliberately blind to
substantive interpretation, same discipline as validate_h3_schema.py.
Checks every expected key, type, and value range; never prints or
discusses an actual value.
"""
from typing import Any, Dict, List, Tuple

FIELDS = ["churn_risk_score", "total_spend", "tenure_months"]


def _check(label: str, condition: bool, issues: List[str]) -> None:
    if not condition:
        issues.append(label)


def _valid_prob_or_none(x: Any) -> bool:
    return x is None or (isinstance(x, (int, float)) and 0.0 <= x <= 1.0)


def _valid_rate(x: Any) -> bool:
    return isinstance(x, (int, float)) and 0.0 <= x <= 1.0


def validate_h4_results_schema(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    _check("top-level keys present",
           {"core_mcnemar_comparisons", "style_plausibility_contrasts",
            "perturbation_characteristics", "total_spend_bounds_escape",
            "garbage_filter_analysis"} <= set(results), issues)

    # --- 1. core_mcnemar_comparisons ---
    core = results.get("core_mcnemar_comparisons", {})
    for f in FIELDS:
        d = core.get(f)
        _check(f"core.{f}: present", d is not None, issues)
        if d is None:
            continue
        for label in ["drift_vs_plausible", "drift_vs_implausible"]:
            sub = d.get(label)
            _check(f"core.{f}.{label}: present", sub is not None, issues)
            if sub is None:
                continue
            _check(f"core.{f}.{label}.drift_rate_drift: valid rate",
                   _valid_rate(sub.get("drift_rate_drift")), issues)
            _check(f"core.{f}.{label}.drift_rate_other: valid rate",
                   _valid_rate(sub.get("drift_rate_other")), issues)
            _check(f"core.{f}.{label}.p_value: valid prob or None",
                   _valid_prob_or_none(sub.get("p_value")), issues)
            _check(f"core.{f}.{label}.n_shared_customers: positive int",
                   isinstance(sub.get("n_shared_customers"), int) and sub["n_shared_customers"] > 0,
                   issues)

    # --- 2. style_plausibility_contrasts ---
    sp = results.get("style_plausibility_contrasts", {})
    _check("contrasts.headline_source valid",
           sp.get("headline_source") in ("per_field", "pooled"), issues)
    gate = sp.get("interaction_gate", {})
    _check("contrasts.gate.plausibility_interaction_p_value: valid prob or None",
           _valid_prob_or_none(gate.get("plausibility_interaction_p_value")), issues)
    _check("contrasts.gate.style_interaction_p_value: valid prob or None",
           _valid_prob_or_none(gate.get("style_interaction_p_value")), issues)
    _check("contrasts.gate.safe_to_pool_plausibility: bool",
           isinstance(gate.get("safe_to_pool_plausibility"), bool), issues)
    per_field = sp.get("per_field", {})
    for f in FIELDS:
        pf = per_field.get(f)
        _check(f"contrasts.per_field.{f}: present", pf is not None, issues)
        if pf is not None:
            _check(f"contrasts.per_field.{f}.plausibility_p_value: valid prob or None",
                   _valid_prob_or_none(pf.get("plausibility_p_value")), issues)
            _check(f"contrasts.per_field.{f}.plausibility_odds_ratio: positive or None",
                   pf.get("plausibility_odds_ratio") is None or pf["plausibility_odds_ratio"] > 0, issues)
    pooled = sp.get("pooled", {})
    _check("contrasts.pooled.plausibility_p_value: valid prob or None",
           _valid_prob_or_none(pooled.get("plausibility_p_value")), issues)

    # --- 3. perturbation_characteristics ---
    pc = results.get("perturbation_characteristics", {})
    for f in FIELDS:
        d = pc.get(f)
        _check(f"perturbation.{f}: present", d is not None, issues)
        if d is None:
            continue
        for plaus in ["plausible", "implausible"]:
            sub = d.get(plaus)
            _check(f"perturbation.{f}.{plaus}: present", sub is not None, issues)
            if sub is None:
                continue
            _check(f"perturbation.{f}.{plaus}.n: non-negative int",
                   isinstance(sub.get("n"), int) and sub["n"] >= 0, issues)
            for key in ["operator_breakdown", "direction_breakdown", "mechanism_style_breakdown"]:
                _check(f"perturbation.{f}.{plaus}.{key}: is a dict",
                       isinstance(sub.get(key), dict), issues)

    # --- 4. total_spend_bounds_escape ---
    bounds = results.get("total_spend_bounds_escape")
    if bounds is not None:
        _check("bounds_escape.n_escaped_bounds: non-negative int",
               isinstance(bounds.get("n_escaped_bounds"), int) and bounds["n_escaped_bounds"] >= 0, issues)
        _check("bounds_escape.drift_rate_escaped: valid rate or None",
               _valid_prob_or_none(bounds.get("drift_rate_escaped")), issues)
        _check("bounds_escape.drift_rate_in_bounds: valid rate or None",
               _valid_prob_or_none(bounds.get("drift_rate_in_bounds")), issues)

    # --- 5. garbage_filter_analysis (expected stub for now) ---
    gf = results.get("garbage_filter_analysis", {})
    _check("garbage_filter.status present", "status" in gf, issues)

    return (len(issues) == 0, issues)


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=str)
    args = parser.parse_args()
    with open(args.results_path) as f:
        results = json.load(f)
    passed, issues = validate_h4_results_schema(results)
    if passed:
        print("✅ STRUCTURAL VALIDATION PASSED — schema, types, and ranges all correct.")
        print("   (No substantive values inspected or reported by design.)")
    else:
        print(f"❌ STRUCTURAL VALIDATION FAILED — {len(issues)} issue(s):")
        for i in issues:
            print(f"   - {i}")
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())

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


STRATA = ("stable", "boundary", "all")


def _walk(node, path, issues):
    """Recursively check every comparison / rate dict in the garbage-filter
    block for internal consistency: counts are non-negative integers, rates
    and p-values lie in [0, 1], intervals contain their rate, and the paired
    2x2 adds up. Never inspects what a value MEANS."""
    if isinstance(node, dict):
        if "n_pairs" in node and node["n_pairs"] > 0:
            for k in ("a_only", "b_only", "both", "neither", "n_discordant"):
                _check(f"{path}.{k}: non-negative int", isinstance(node.get(k), int) and node[k] >= 0, issues)
            if all(isinstance(node.get(k), int) for k in ("a_only", "b_only", "both", "neither")):
                _check(f"{path}: 2x2 sums to n_pairs",
                       node["a_only"] + node["b_only"] + node["both"] + node["neither"] == node["n_pairs"], issues)
                _check(f"{path}: n_discordant == a_only + b_only",
                       node.get("n_discordant") == node["a_only"] + node["b_only"], issues)
            _check(f"{path}.p_value in [0,1]", _valid_rate(node.get("p_value")), issues)
            _check(f"{path}.rate_a / rate_b in [0,1]", _valid_rate(node.get("rate_a")) and _valid_rate(node.get("rate_b")), issues)
            _check(f"{path}.low_power is bool", isinstance(node.get("low_power"), bool), issues)
            _check(f"{path}.primary is bool", isinstance(node.get("primary"), bool), issues)
        if "k" in node and "n" in node and "ci" in node and node["n"] is not None:
            _check(f"{path}: 0 <= k <= n", isinstance(node["k"], int) and 0 <= node["k"] <= node["n"], issues)
            if node["n"] > 0:
                lo, hi = node["ci"]
                _check(f"{path}: rate within its interval within [0,1]",
                       0.0 <= lo <= node["rate"] <= hi <= 1.0, issues)
        for k, v in node.items():
            _walk(v, f"{path}.{k}", issues)


def _validate_garbage_filter(gf, issues):
    for key in ("strata", "arms", "detection_spike", "isolation", "caveats", "judge_records_joined"):
        _check(f"garbage_filter.{key}: present", key in gf, issues)
    _check("garbage_filter.caveats: non-empty list of strings",
           isinstance(gf.get("caveats"), list) and gf["caveats"] and all(isinstance(c, str) for c in gf["caveats"]), issues)

    for field in FIELDS:
        arms = gf.get("arms", {}).get(field)
        _check(f"garbage_filter.arms.{field}: present", arms is not None, issues)
        if arms is None:
            continue
        for arm in ("drift", "plausible", "implausible"):
            _check(f"garbage_filter.arms.{field}.{arm}: present", arm in arms, issues)
        for arm, by_stratum in arms.items():
            for s in STRATA:
                summ = by_stratum.get(s)
                _check(f"garbage_filter.arms.{field}.{arm}.{s}: present", summ is not None, issues)
                if summ is not None:
                    _check(f"garbage_filter.arms.{field}.{arm}.{s}.n: non-negative int",
                           isinstance(summ.get("n"), int) and summ["n"] >= 0, issues)
        _check(f"garbage_filter.detection_spike.{field}: has all strata",
               all(s in gf.get("detection_spike", {}).get(field, {}) for s in STRATA), issues)

    # exactly one family of pre-specified primary tests: tagged, and only on the stable stratum
    spike = gf.get("detection_spike", {})
    for field in FIELDS:
        for s in STRATA:
            for test, c in spike.get(field, {}).get(s, {}).items():
                if isinstance(c, dict) and "primary" in c and test != "drift":
                    _check(f"detection_spike.{field}.{s}.{test}: primary only on stable",
                           c["primary"] == (s == "stable"), issues)

    for p in ("plausible", "implausible"):
        iso = gf.get("isolation", {}).get(p)
        _check(f"garbage_filter.isolation.{p}: present", iso is not None, issues)
        if iso and iso.get("status") == "ok":
            dc = iso.get("design_check", {})
            _check(f"isolation.{p}.design_check: counts consistent",
                   dc.get("n_contradiction_carrying", -1) + dc.get("n_input_identical", -1) == dc.get("n_shared_customers"), issues)
            _check(f"isolation.{p}.design_check.design_ok: bool", isinstance(dc.get("design_ok"), bool), issues)
    for fam, tests in gf.get("primary_tests", {}).items():
        _check(f"primary_tests.{fam}: non-empty list", isinstance(tests, list) and len(tests) > 0, issues)
        for t in tests if isinstance(tests, list) else []:
            _check(f"primary_tests.{fam}.{t.get('test')}: 0 <= p_value <= p_holm <= 1",
                   _valid_rate(t.get("p_value")) and _valid_rate(t.get("p_holm")) and t["p_holm"] >= t["p_value"] - 1e-12, issues)
    _walk(gf, "garbage_filter", issues)


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

    # --- 5. garbage_filter_analysis ---
    gf = results.get("garbage_filter_analysis", {})
    _check("garbage_filter.status present", "status" in gf, issues)
    if gf.get("status") == "implemented":
        _validate_garbage_filter(gf, issues)

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

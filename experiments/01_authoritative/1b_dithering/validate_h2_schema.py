#!/usr/bin/env python3
"""
Structural validator for h2_results.json — deliberately blind to
substantive interpretation. Checks that every expected key exists, every
value has the right type, and every numeric value sits in a valid range
(a p-value in [0,1], a drift rate in [0,1], a net drift in [-1,1], a
classification from the allowed set, and so on), and that the stratified
layout and the pre-registered shape test are present.

Like validate_h3_schema.py and validate_h4_schema.py, it exists so a
smoke-test run's output can be checked without looking at what the
numbers say: on failure it names the field and the rule it broke; it never
prints a value.

    python3 validate_h2_schema.py experiments_output/evaluation/h2_results.json
"""

from typing import Any, Dict, List, Tuple

FIELDS = ["churn_risk_score", "total_spend", "tenure_months"]
MAGS = ["5", "15", "40", "100"]
STRATA = ["stable", "boundary", "all"]
CLASSES = {"cliff", "smooth", "mixed", "below_floor", "no_population"}
WITHIN_SEGMENT_FIELDS = {"churn_risk_score", "total_spend"}


def _check(label: str, condition: bool, issues: List[str]) -> None:
    if not condition:
        issues.append(label)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rate(x: Any) -> bool:
    return _num(x) and 0.0 <= x <= 1.0


def _prob_or_none(x: Any) -> bool:
    return x is None or _rate(x)


def _signed(x: Any) -> bool:
    return _num(x) and -1.0 <= x <= 1.0


def _nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _ladder(where: str, d: Dict[str, Any], issues: List[str], expect_segments: bool) -> None:
    _check(f"{where}.n_customers: non-negative int", _nonneg_int(d.get("n_customers")), issues)
    if not d.get("n_customers"):
        return
    rates = d.get("drift_rates_by_magnitude", {})
    _check(f"{where}.drift_rates_by_magnitude: all four rungs, valid rates",
           set(rates) == set(MAGS) and all(_rate(v) for v in rates.values()), issues)
    om = d.get("cochrans_q_omnibus", {})
    _check(f"{where}.cochrans_q_omnibus.p_value: valid prob or None", _prob_or_none(om.get("p_value")), issues)
    _check(f"{where}.omnibus_significant: bool", isinstance(d.get("omnibus_significant"), bool), issues)
    adj = d.get("adjacent_step_mcnemar")
    if d.get("omnibus_significant"):
        _check(f"{where}.adjacent_step_mcnemar: three steps when the omnibus is significant",
               isinstance(adj, list) and len(adj) == 3 and all(_prob_or_none(a.get("p_value")) for a in adj), issues)
    else:
        _check(f"{where}.adjacent_step_mcnemar: None when the omnibus is not significant", adj is None, issues)
    ex = d.get("extreme_comparison_5v100", {})
    _check(f"{where}.extreme_comparison_5v100: rates valid, effect in [-1,1], p valid",
           _rate(ex.get("drift_rate_5pct")) and _rate(ex.get("drift_rate_100pct")) and _signed(ex.get("total_range_effect"))
           and _prob_or_none(ex.get("p_value")), issues)
    tr = d.get("descriptive_trend", {})
    _check(f"{where}.descriptive_trend: rho in [-1,1] or None, with its caveat",
           (tr.get("rho") is None or _signed(tr.get("rho"))) and ("interpretation" in tr or "note" in tr), issues)
    net = d.get("net_of_control")
    _check(f"{where}.net_of_control: control rate valid, four net values in [-1,1]",
           net is None or (_rate(net.get("control_drift_rate")) and set(net.get("net_drift_by_magnitude", {})) == set(MAGS)
                           and all(_signed(v) for v in net["net_drift_by_magnitude"].values())), issues)
    seg = d.get("within_segment_breakdown")
    if expect_segments:
        _check(f"{where}.within_segment_breakdown: present, rungs x segments, valid rates",
               isinstance(seg, dict) and set(seg) == set(MAGS) and all(_rate(v) for m in seg.values() for v in m.values()), issues)
    else:
        _check(f"{where}.within_segment_breakdown: absent outside its scope", seg is None, issues)


def _shape_field(where: str, d: Dict[str, Any], issues: List[str]) -> None:
    _check(f"{where}.classification: one of {sorted(CLASSES)}", d.get("classification") in CLASSES, issues)
    if d.get("classification") == "no_population":
        return
    _check(f"{where}.n_customers: positive int", _nonneg_int(d.get("n_customers")) and d["n_customers"] > 0, issues)
    _check(f"{where}.control_drift_rate: valid rate", _rate(d.get("control_drift_rate")), issues)
    dm = d.get("drift_by_magnitude", {})
    _check(f"{where}.drift_by_magnitude: four valid rates", set(dm) == set(MAGS) and all(_rate(v) for v in dm.values()), issues)
    _check(f"{where}.s and F: in [-1,1]", _signed(d.get("s")) and _signed(d.get("F")), issues)
    _check(f"{where}.F_rung_pct: 15, 40 or 100", d.get("F_rung_pct") in (15, 40, 100), issues)
    _check(f"{where}.s_over_F: numeric or None", d.get("s_over_F") is None or _num(d.get("s_over_F")), issues)
    ci = d.get("s_over_F_ci95_bootstrap")
    _check(f"{where}.s_over_F_ci95_bootstrap: None or an ordered pair",
           ci is None or (isinstance(ci, list) and len(ci) == 2 and all(_num(x) for x in ci) and ci[0] <= ci[1]), issues)
    _check(f"{where}.non_monotone: bool", isinstance(d.get("non_monotone"), bool), issues)
    mc = d.get("mcnemar_5pct_vs_control", {})
    _check(f"{where}.mcnemar_5pct_vs_control: counts and p-values valid, p_holm >= p",
           _nonneg_int(mc.get("a_only_5pct")) and _nonneg_int(mc.get("b_only_control")) and _prob_or_none(mc.get("p_value"))
           and _prob_or_none(mc.get("p_holm")) and (mc.get("p_holm") is None or mc["p_holm"] >= mc["p_value"] * (1 - 1e-9)), issues)
    h = d.get("halves", {})
    _check(f"{where}.halves: both halves, sizes adding up to the population",
           set(h) == {"0", "1"} and all(_nonneg_int(x.get("n_customers")) for x in h.values())
           and sum(x["n_customers"] for x in h.values()) == d.get("n_customers"), issues)
    if d.get("classification") == "cliff":
        _check(f"{where}: a cliff is significant after Holm", (mc.get("p_holm") or 1) < 0.05, issues)


def _shape(where: str, st: Dict[str, Any], issues: List[str], top: bool) -> None:
    _check(f"{where}.M: in (0,1)", _num(st.get("M")) and 0 < st["M"] < 1, issues)
    pf = st.get("per_field", {})
    _check(f"{where}.per_field: all three fields", set(pf) == set(FIELDS), issues)
    for f in FIELDS:
        if f in pf:
            _shape_field(f"{where}.per_field.{f}", pf[f], issues)
    cliffs = st.get("cliff_fields")
    _check(f"{where}.cliff_fields: the fields classified cliff",
           isinstance(cliffs, list) and set(cliffs) == {f for f in FIELDS if pf.get(f, {}).get("classification") == "cliff"}, issues)
    _check(f"{where}.n_cliff_fields matches", st.get("n_cliff_fields") == len(cliffs or []), issues)
    eff = st.get("candidate_effect_median_s")
    _check(f"{where}.candidate_effect_median_s: None without cliffs, else in [-1,1]",
           (eff is None) == (not cliffs) and (eff is None or _signed(eff)), issues)
    _check(f"{where}.qualified: bool, and only with at least two cliffs replicated in both halves",
           isinstance(st.get("qualified"), bool) and (not st["qualified"] or (len(cliffs or []) >= 2 and st.get("replicated_in_both_halves") is True)),
           issues)
    _check(f"{where}.effect_at_least_M: bool", isinstance(st.get("effect_at_least_M"), bool), issues)
    if top:
        sens = st.get("sensitivity", {})
        _check(f"{where}.sensitivity: M = 5 and 15 tabulated", set(sens) == {"M_5", "M_15"}
               and all(set(v.get("classification", {})) == set(FIELDS) and all(c in CLASSES for c in v["classification"].values())
                       and isinstance(v.get("qualified"), bool) for v in sens.values()), issues)
        _check(f"{where}.rules: recorded", isinstance(st.get("rules"), dict) and st["rules"].get("stratum") == "stable", issues)


def validate_h2_results_schema(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    _check("top-level keys present",
           {"hypothesis", "primary_stratum", "strata_definitions", "stratum_sizes", "control_condition", "fields", "shape_test"} <= set(results),
           issues)
    _check("hypothesis is H2 and the primary stratum is stable",
           results.get("hypothesis") == "H2" and results.get("primary_stratum") == "stable", issues)
    sizes = results.get("stratum_sizes", {})
    _check("stratum_sizes: three strata, stable + boundary <= all",
           set(sizes) == set(STRATA) and all(_nonneg_int(v) for v in sizes.values())
           and sizes.get("stable", 0) + sizes.get("boundary", 0) <= sizes.get("all", 0), issues)
    fields = results.get("fields", {})
    _check("fields: all three", set(fields) == set(FIELDS), issues)
    for f in FIELDS:
        per = fields.get(f, {})
        _check(f"fields.{f}: all three strata", set(per) == set(STRATA), issues)
        for s in STRATA:
            if s in per:
                _ladder(f"fields.{f}.{s}", per[s], issues,
                        expect_segments=(f in WITHIN_SEGMENT_FIELDS and s in ("stable", "all") and bool(per[s].get("n_customers"))))
    _shape("shape_test", results.get("shape_test", {}), issues, top=True)
    return (len(issues) == 0, issues)


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_path", type=str)
    args = parser.parse_args()
    with open(args.results_path) as f:
        results = json.load(f)
    passed, issues = validate_h2_results_schema(results)
    if passed:
        print("✅ STRUCTURAL VALIDATION PASSED — schema, types, and ranges all correct.")
        print("   (No substantive values inspected or reported by design.)")
    else:
        print(f"❌ STRUCTURAL VALIDATION FAILED — {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())

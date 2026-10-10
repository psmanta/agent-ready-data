#!/usr/bin/env python3
"""
Structural validator for h1_results.json — deliberately blind to substantive
interpretation, same discipline as validate_h3_schema.py / validate_h4_schema.py.
Checks every expected key, type, and value range, and internal consistency
(group sizes against pair counts, win counts against pair counts, the manifest
status against the 1b section it governs); never prints or discusses an
actual drift rate.

Usage:
    python validate_h1_schema.py experiments_output/evaluation/h1_results.json
"""
import json
import re
import sys
from typing import Any, Dict, List, Tuple

TOP_LEVEL = ("hypothesis", "question_a_field_importance", "question_a_manifest", "question_a_1b_list",
             "question_b_category_ranking", "question_c_field_attribution", "question_d_identity_null_check",
             "stated_vs_revealed_importance", "h1_distributed")
LEGACY_SIZES = (5, 6)          # the amendment: 5 top fields vs 6 comparison fields
MANIFEST_STATUSES = ("used", "no_manifest")
SECTION_1B_STATUSES = ("ok", "not_triggered", "incomplete", "skipped: no manifest")


def _check(label: str, condition: bool, issues: List[str]) -> None:
    if not condition:
        issues.append(label)


def _rate(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0


def _rate_or_none(x: Any) -> bool:
    return x is None or _rate(x)


def _nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _check_question_a(qa: Dict[str, Any], label: str, n_top: int, n_comp: int, require_pairs: bool,
                      issues: List[str]) -> None:
    for key in ("adjustment_note", "top5_fields", "comparison_fields", "group_level_check_secondary",
                "pairwise_mcnemar_primary"):
        _check(f"{label}.{key}: present", key in qa, issues)
    if not all(k in qa for k in ("top5_fields", "comparison_fields", "pairwise_mcnemar_primary")):
        return
    _check(f"{label}: top group has {n_top} fields", len(qa["top5_fields"]) == n_top, issues)
    _check(f"{label}: comparison group has {n_comp} fields", len(qa["comparison_fields"]) == n_comp, issues)
    for grp in ("top5_fields", "comparison_fields"):
        for field, r in qa[grp].items():
            p = f"{label}.{grp}.{field}"
            _check(f"{p}: effective_drift_rate in [0,1] or null", _rate_or_none(r.get("effective_drift_rate")), issues)
            _check(f"{p}: raw_drift_rate in [0,1]", _rate(r.get("raw_drift_rate")), issues)
            _check(f"{p}: exposure in [0,1]", _rate(r.get("exposure")), issues)
            _check(f"{p}: n_perturbed non-negative int", _nonneg_int(r.get("n_perturbed")), issues)
    g = qa["group_level_check_secondary"]
    _check(f"{label}: group-level p_value in [0,1] or null", _rate_or_none(g.get("p_value")), issues)
    _check(f"{label}: group-level carries its low-power caveat", isinstance(g.get("power_caveat"), str) and "LOW POWER" in g["power_caveat"], issues)
    pw = qa["pairwise_mcnemar_primary"]
    for k in ("n_pairs", "n_significant_p05", "top5_higher_drift_count", "comparison_higher_drift_count"):
        _check(f"{label}.pairwise.{k}: non-negative int", _nonneg_int(pw.get(k)), issues)
    if all(_nonneg_int(pw.get(k)) for k in ("n_pairs", "n_significant_p05", "top5_higher_drift_count", "comparison_higher_drift_count")):
        _check(f"{label}: n_pairs == n_top x n_comparison", pw["n_pairs"] == n_top * n_comp, issues)
        _check(f"{label}: n_significant <= n_pairs", pw["n_significant_p05"] <= pw["n_pairs"], issues)
        _check(f"{label}: wins on both sides <= n_pairs", pw["top5_higher_drift_count"] + pw["comparison_higher_drift_count"] <= pw["n_pairs"], issues)
    if require_pairs:
        pairs = pw.get("all_pairs")
        _check(f"{label}: all_pairs present", isinstance(pairs, list), issues)
        if isinstance(pairs, list):
            _check(f"{label}: all_pairs has n_pairs entries", len(pairs) == pw.get("n_pairs"), issues)
            for i, pr in enumerate(pairs):
                p = f"{label}.all_pairs[{i}]"
                _check(f"{p}: p_value in [0,1]", _rate(pr.get("p_value")), issues)
                for k in ("b_x_only", "c_y_only", "n_discordant", "n_perturbed_in_both"):
                    _check(f"{p}.{k}: non-negative int", _nonneg_int(pr.get(k)), issues)
                if _nonneg_int(pr.get("b_x_only")) and _nonneg_int(pr.get("c_y_only")):
                    _check(f"{p}: n_discordant == b_x_only + c_y_only", pr.get("n_discordant") == pr["b_x_only"] + pr["c_y_only"], issues)
                if _nonneg_int(pr.get("n_discordant")) and _nonneg_int(pr.get("n_perturbed_in_both")):
                    _check(f"{p}: n_discordant <= n_perturbed_in_both", pr["n_discordant"] <= pr["n_perturbed_in_both"], issues)
                _check(f"{p}: effective rates in [0,1]", _rate(pr.get("top5_effective_drift_rate")) and _rate(pr.get("comparison_effective_drift_rate")), issues)
    else:
        _check(f"{label}: sensitivity analysis carries no pair list", "all_pairs" not in pw, issues)


def validate_h1_results_schema(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for key in TOP_LEVEL:
        _check(f"top level: {key} present", key in results, issues)
    if issues:
        return False, issues
    _check("hypothesis == H1", results["hypothesis"] == "H1", issues)

    _check_question_a(results["question_a_field_importance"], "question_a (legacy list)", *LEGACY_SIZES, True, issues)

    qm = results["question_a_manifest"]
    _check("question_a_manifest.status is a known value", qm.get("status") in MANIFEST_STATUSES, issues)
    sec = results["question_a_1b_list"]
    _check("question_a_1b_list.status is a known value", sec.get("status") in SECTION_1B_STATUSES, issues)
    triggered = None
    if qm.get("status") == "used":
        _check("manifest: decision_sha256 is 64 hex characters", isinstance(qm.get("decision_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", qm["decision_sha256"]) is not None, issues)
        _check("manifest: triggered is bool", isinstance(qm.get("triggered"), bool), issues)
        triggered = qm.get("triggered")
        _check("manifest: primary_list consistent with the trigger", qm.get("primary_list") == ("1b" if triggered else "legacy_1a"), issues)
        _check("manifest: legacy_role consistent with the trigger", qm.get("legacy_role") == ("legacy comparison" if triggered else "primary"), issues)
        if triggered is False:
            _check("1b section: not_triggered when the trigger did not fire", sec.get("status") == "not_triggered", issues)
        if triggered is True:
            _check("1b section: ok or incomplete when the trigger fired", sec.get("status") in ("ok", "incomplete"), issues)
    else:
        _check("1b section: skipped when there is no manifest", sec.get("status") == "skipped: no manifest", issues)

    if sec.get("status") == "ok":
        _check("1b section: role is primary", sec.get("role") == "primary", issues)
        _check("1b section: carries the manifest's hash", sec.get("manifest_decision_sha256") == qm.get("decision_sha256"), issues)
        top, comp = sec.get("top_group"), sec.get("comparison_group")
        _check("1b section: top_group is a non-empty list of {field, condition_id}",
               isinstance(top, list) and len(top) >= 1 and all({"field", "condition_id"} <= set(e) for e in top), issues)
        _check("1b section: comparison_group is a non-empty list", isinstance(comp, list) and len(comp) >= 1, issues)
        if isinstance(top, list) and isinstance(comp, list) and top and comp:
            _check("1b section: top and comparison groups are disjoint",
                   not ({e["condition_id"] for e in top} & {e["condition_id"] for e in comp}), issues)
            _check_question_a(sec["analysis"], "question_a (1b list)", len(top), len(comp), True, issues)
            sens = sec.get("sensitivity_excluding_1b_ranks_6_to_8", {})
            sg = sens.get("comparison_group")
            _check("1b section: sensitivity comparison group is a subset of the comparison group",
                   isinstance(sg, list) and {e["condition_id"] for e in sg} <= {e["condition_id"] for e in comp}, issues)
            if isinstance(sg, list) and sg:
                _check_question_a(sens["analysis"], "question_a (1b sensitivity)", len(top), len(sg), False, issues)
    if sec.get("status") == "incomplete":
        _check("1b section: incomplete lists the missing conditions", isinstance(sec.get("missing_conditions"), list) and len(sec["missing_conditions"]) >= 1, issues)

    sv = results["stated_vs_revealed_importance"]
    _check("stated_vs_revealed: top5_comparison has 5 rows", isinstance(sv.get("top5_comparison"), list) and len(sv["top5_comparison"]) == 5, issues)
    for i, row in enumerate(sv.get("top5_comparison", [])):
        _check(f"stated_vs_revealed.top5_comparison[{i}]: rates in [0,1]", _rate(row.get("stated_citation_rate")) and _rate(row.get("revealed_drift_rate")), issues)
        if qm.get("status") == "used":
            _check(f"stated_vs_revealed.top5_comparison[{i}]: manifest_citation_rate present and in [0,1]", _rate(row.get("manifest_citation_rate")), issues)
    if triggered is True and sec.get("status") == "ok":
        s1b = sv.get("top5_comparison_1b")
        _check("stated_vs_revealed: 1b rows present when the trigger fired", isinstance(s1b, dict) and s1b.get("role") == "primary", issues)
        if isinstance(s1b, dict):
            _check("stated_vs_revealed: 1b rows match the 1b top group in number",
                   len(s1b.get("rows", [])) == len(sec.get("top_group", [])), issues)
            for i, row in enumerate(s1b.get("rows", [])):
                _check(f"stated_vs_revealed.top5_comparison_1b[{i}]: rates in [0,1]", _rate(row.get("stated_citation_rate")) and _rate(row.get("revealed_drift_rate")), issues)
                _check(f"stated_vs_revealed.top5_comparison_1b[{i}]: in_1a_top5 is bool", isinstance(row.get("in_1a_top5"), bool), issues)
    elif sec.get("status") == "incomplete":
        s1b = sv.get("top5_comparison_1b")
        _check("stated_vs_revealed: an incomplete 1b section reports its 1b rows as incomplete",
               s1b is None or (isinstance(s1b, dict) and s1b.get("status") == "incomplete" and isinstance(s1b.get("missing_conditions"), list) and len(s1b["missing_conditions"]) >= 1), issues)
    elif triggered is False or qm.get("status") != "used":
        _check("stated_vs_revealed: no 1b rows unless the trigger fired", "top5_comparison_1b" not in sv, issues)
    return (not issues), issues


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python validate_h1_schema.py <h1_results.json>")
        return 2
    results = json.load(open(sys.argv[1]))
    ok, issues = validate_h1_results_schema(results)
    if ok:
        print("✅ STRUCTURAL VALIDATION PASSED — schema, types, ranges and internal consistency all correct.")
        return 0
    print(f"❌ STRUCTURAL VALIDATION FAILED — {len(issues)} issue(s):")
    for i in issues:
        print(f"  - {i}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Structural validator for h5_results.json -- deliberately blind to substantive interpretation, same
discipline as the H1/H3/H4 validators. Checks keys, types, ranges and internal consistency (every
2x2 sums to its n, every interval contains its rate, sub-counts never exceed totals, section labels are
the pre-registered ones); never interprets a detection rate.

Usage:  python validate_h5_schema.py experiments_output/evaluation/h5_results.json
"""
import json
import sys
from typing import Any, Dict, List, Tuple

STRATA = ("stable", "boundary", "all")
EXPECTED_STATUS = {"detection_crosstab": "primary", "echo_classes": "exploratory", "coherence": "exploratory",
                   "redundancy_classes": "exploratory", "judge": "provisional"}
ECHO_CLASSES = {"echo", "echo_pct_marker", "converted", "magnitude_only", "omitted", "not_applicable"}
REDUNDANCY_CLASSES = {"strict", "bounded", "approximate", "soft_only", "none"}


def _check(label: str, cond: bool, issues: List[str]) -> None:
    if not cond:
        issues.append(label)


def _rate(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0


def _nn(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _matrix_matches(m: Any, table: Any) -> bool:
    """An explicit [group] x [drift status] matrix must equal what its rate table implies (k drifted, n-k not)."""
    return (isinstance(m, dict) and isinstance(table, dict) and set(m) == set(table) and
            all(isinstance(table[g], dict) and m[g] == {"not_drifted": table[g]["n"] - table[g]["k"], "drifted": table[g]["k"]} for g in table))


def _rate_block(b: Any, label: str, issues: List[str]) -> None:
    if not isinstance(b, dict) or not _nn(b.get("k")) or not _nn(b.get("n")):
        issues.append(f"{label}: rate block needs non-negative integer k and n"); return
    _check(f"{label}: k <= n", b["k"] <= b["n"], issues)
    if b["n"] == 0:
        _check(f"{label}: an empty block has rate and ci null", b.get("rate") is None and b.get("ci") is None, issues)
    else:
        _check(f"{label}: rate in [0,1] and equals k/n", _rate(b.get("rate")) and abs(b["rate"] - b["k"] / b["n"]) < 1e-9, issues)
        ci = b.get("ci")
        _check(f"{label}: ci is [lo, hi] in [0,1] containing the rate",
               isinstance(ci, list) and len(ci) == 2 and -1e-9 <= ci[0] <= b.get("rate", -1) + 1e-9 and b.get("rate", 2) - 1e-9 <= ci[1] <= 1 + 1e-9, issues)


def validate_h5_results_schema(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for key in ("hypothesis", "status_by_section", "n_conditions", "conditions_missing_decisions", "audit_stamp", "conditions", "redundancy_classes", "strata_note", "floor_note"):
        _check(f"top level: {key} present", key in results, issues)
    if issues:
        return False, issues
    _check("hypothesis == H5", results["hypothesis"] == "H5", issues)
    _check("section labels are the pre-registered ones (only detection_crosstab is primary)", results["status_by_section"] == EXPECTED_STATUS, issues)
    conds = results["conditions"]
    _check("n_conditions equals the number of conditions", results["n_conditions"] == len(conds), issues)

    n_by_cond: Dict[str, int] = {}
    for cid, t in conds.items():
        p = f"{cid}"
        for key in ("exposure", "detection_crosstab", "echo_classes", "coherence"):
            _check(f"{p}: {key} present", key in t, issues)
        if not all(k in t for k in ("exposure", "detection_crosstab", "echo_classes", "coherence")):
            continue
        e = t["exposure"]
        _check(f"{p}: exposure n_perturbed <= n_records", _nn(e.get("n_records")) and _nn(e.get("n_perturbed")) and e["n_perturbed"] <= e["n_records"], issues)
        ss = t.get("sample_sizes", {})
        ssk = ("total_n", "n_perturbed", "n_stable", "n_boundary", "n_unclassified")
        _check(f"{p}: sample_sizes are non-negative integers", all(_nn(ss.get(k)) for k in ssk), issues)
        if all(_nn(ss.get(k)) for k in ssk):
            _check(f"{p}: sample_sizes match the exposure counts", ss["total_n"] == t["exposure"].get("n_records") and ss["n_perturbed"] == t["exposure"].get("n_perturbed"), issues)
            _check(f"{p}: sample_sizes stable + boundary + unclassified == perturbed", ss["n_stable"] + ss["n_boundary"] + ss["n_unclassified"] == ss["n_perturbed"], issues)
            _check(f"{p}: sample_sizes agree with the stratum tables", ss["n_stable"] == t["detection_crosstab"].get("stable", {}).get("n") and ss["n_boundary"] == t["detection_crosstab"].get("boundary", {}).get("n"), issues)
        jp = t.get("judge_prompt_ids", "MISSING")
        _check(f"{p}: judge_prompt_ids is null exactly when there is no judge section", (jp is None) == ("judge" not in t) and jp != "MISSING", issues)
        _check(f"{p}: judge_prompt_ids is null or a list of strings", jp is None or (isinstance(jp, list) and all(isinstance(x, str) for x in jp)), issues)
        ct = t["detection_crosstab"]
        for s in STRATA:
            c = ct.get(s)
            if c is None:
                issues.append(f"{p}.detection_crosstab: stratum {s} missing"); continue
            q = f"{p}.detection_crosstab.{s}"
            cells = [c.get(k) for k in ("detected_changed", "detected_unchanged", "not_detected_changed", "not_detected_unchanged")]
            _check(f"{q}: the four cells are non-negative integers", all(_nn(x) for x in cells) and _nn(c.get("n")), issues)
            if all(_nn(x) for x in cells) and _nn(c.get("n")):
                _check(f"{q}: the 2x2 sums to n", sum(cells) == c["n"], issues)
                dm = c.get("detection_by_drift_matrix")
                _check(f"{q}: detection-by-drift matrix equals the 2x2 cells", dm == {"detected": {"not_drifted": cells[1], "drifted": cells[0]}, "not_detected": {"not_drifted": cells[3], "drifted": cells[2]}}, issues)
                tw = c.get("three_way", {})
                td, tw = c.get("three_way_decision_only", {}), c.get("three_way", {})
                _check(f"{q}: three_way_decision_only agrees with the 2x2 cells", td == {"detected_and_changed": cells[0], "detected_no_change": cells[1], "not_detected": cells[2] + cells[3]}, issues)
                _check(f"{q}: three_way sums to n", all(_nn(tw.get(k)) for k in ("detected_and_changed", "detected_no_change", "not_detected")) and tw.get("detected_and_changed", 0) + tw.get("detected_no_change", 0) + tw.get("not_detected", 0) == c["n"], issues)
                if all(_nn(tw.get(k)) for k in ("detected_and_changed", "not_detected")):
                    _check(f"{q}: adding confidence can only grow 'detected and changed'", tw["detected_and_changed"] >= td.get("detected_and_changed", 0), issues)
                    _check(f"{q}: 'not detected' is identical in both three-way tables", tw["not_detected"] == td.get("not_detected"), issues)
                cf = c.get("confidence", {})
                _check(f"{q}: confidence counts are non-negative and at most n", _nn(cf.get("n_outside_baseline_range")) and _nn(cf.get("n_unknown")) and cf["n_outside_baseline_range"] <= c["n"] and cf["n_unknown"] <= c["n"], issues)
                _check(f"{q}: carries the definition of 'behavior changed'", isinstance(c.get("behavior_changed_definition"), str) and "confidence" in c["behavior_changed_definition"], issues)
                _check_k = c.get("keyword_detection", {})
                _check(f"{q}: keyword_detection.k equals detected_changed + detected_unchanged and n matches", _check_k.get("k") == cells[0] + cells[1] and _check_k.get("n") == c["n"], issues)
            _rate_block(c.get("keyword_detection"), f"{q}.keyword_detection", issues)
            f = c.get("clean_baseline_floor", {})
            _check(f"{q}: floor hits <= runs, hit_rate in [0,1]", _nn(f.get("n_hits")) and _nn(f.get("n_runs")) and f["n_hits"] <= f["n_runs"] and _rate(f.get("hit_rate")), issues)
        if all(ct.get(s) for s in STRATA):
            _check(f"{p}: 'all' holds at least stable + boundary", ct["all"]["n"] >= ct["stable"]["n"] + ct["boundary"]["n"], issues)
            n_by_cond[cid] = ct["stable"]["n"]
        for field, per in t["echo_classes"].items():
            for s in STRATA:
                b = per.get(s)
                if b is None:
                    issues.append(f"{p}.echo_classes.{field}: stratum {s} missing"); continue
                q = f"{p}.echo_classes.{field}.{s}"
                _check(f"{q}: class counts sum to n", sum(b["class_counts"].values()) == b["n"], issues)
                _check(f"{q}: only known echo classes", set(b["class_counts"]) <= ECHO_CLASSES, issues)
                _check(f"{q}: drift-by-class n's sum to n", sum(v["n"] for v in b["drift_by_echo_class"].values()) == b["n"], issues)
                _check(f"{q}: echo-by-drift matrix equals its rate table", _matrix_matches(b.get("echo_by_drift_matrix"), b.get("drift_by_echo_class")), issues)
                _check(f"{q}: floor class counts sum to its runs", sum(b["clean_baseline_floor"].get("class_counts", {}).values()) == b["clean_baseline_floor"].get("n_runs", -1), issues)
        co = t["coherence"]
        for s in STRATA:
            b = co.get(s)
            if b is None:
                issues.append(f"{p}.coherence: stratum {s} missing"); continue
            q = f"{p}.coherence.{s}"
            _check(f"{q}: eligible + skipped == n", b["n_eligible"] + b["n_skipped"] == b["n"], issues)
            keys = ("n_inertia_eligible", "n_excluded_from_inertia_fewer_than_3_texts", "n_drifted_inertia_eligible", "flat_coherence", "inertia")
            _check(f"{q}: inertia counts are non-negative integers", all(_nn(b.get(k)) for k in keys), issues)
            if all(_nn(b.get(k)) for k in keys):
                _check(f"{q}: inertia-eligible + excluded == eligible", b["n_inertia_eligible"] + b["n_excluded_from_inertia_fewer_than_3_texts"] == b["n_eligible"], issues)
                _check(f"{q}: inertia <= flat <= inertia-eligible and inertia <= drifted <= inertia-eligible",
                       b["inertia"] <= b["flat_coherence"] <= b["n_inertia_eligible"] and b["inertia"] <= b["n_drifted_inertia_eligible"] <= b["n_inertia_eligible"], issues)
            pv = b["condition_level_shift"].get("p_value")
            _check(f"{q}: shift p-value in [0,1] or null", pv is None or _rate(pv), issues)
        j = t.get("judge")
        if j is not None:
            for s in STRATA:
                b = j.get(s, {})
                q = f"{p}.judge.{s}"
                _check(f"{q}: category counts sum to n", sum(b.get("category_counts", {}).values()) == b.get("n"), issues)
                _check(f"{q}: explicit_concern.k equals its category count", b.get("explicit_concern", {}).get("k") == b.get("category_counts", {}).get("explicit_concern", 0), issues)
                _rate_block(b.get("explicit_concern"), f"{q}.explicit_concern", issues)
                _check(f"{q}: drift-by-category n's sum to n", sum(v["n"] for v in b.get("drift_by_judge_category", {}).values()) == b.get("n"), issues)
                _check(f"{q}: judge-by-drift matrix equals its rate table", _matrix_matches(b.get("judge_by_drift_matrix"), b.get("drift_by_judge_category")), issues)
                _check(f"{q}: echo-by-judge cells sum to n", sum(sum(v.values()) for v in b.get("echo_by_judge", {}).values()) == b.get("n"), issues)

    rc = results["redundancy_classes"]
    for key in ("grouping", "caveat", "per_condition", "pooled_stable", "negative_controls"):
        _check(f"redundancy_classes: {key} present", key in rc, issues)
    if all(k in rc for k in ("per_condition", "pooled_stable", "negative_controls")):
        rows = rc["per_condition"]
        for r in rows:
            _check(f"redundancy_classes.{r['condition_id']}: class is a known class", r["class"] in REDUNDANCY_CLASSES, issues)
            _check(f"redundancy_classes.{r['condition_id']}: n_stable matches the condition's stable cross-tab", n_by_cond.get(r["condition_id"]) == r["n_stable"], issues)
            _check(f"redundancy_classes.{r['condition_id']}: detected_stable <= n_stable", r["detected_stable"] <= r["n_stable"], issues)
        pooled = rc["pooled_stable"]
        hp, npr = pooled.get("has_partner", {"n": 0, "k": 0}), pooled.get("no_partner", {"n": 0, "k": 0})
        _check("redundancy_classes: has_partner + no_partner n equals the sum over conditions", hp["n"] + npr["n"] == sum(r["n_stable"] for r in rows), issues)
        _check("redundancy_classes: has_partner + no_partner k equals the sum over conditions", hp["k"] + npr["k"] == sum(r["detected_stable"] for r in rows), issues)
        for k, b in pooled.items():
            _rate_block(b, f"redundancy_classes.pooled_stable.{k}", issues)
        ids = {r["condition_id"] for r in rows}
        _check("redundancy_classes: negative controls are a subset of the per-condition rows", {r["condition_id"] for r in rc["negative_controls"]} <= ids, issues)
    st = results["audit_stamp"]
    ks, jd, ec_ = st.get("keyword_scan", {}), st.get("judge", {}), st.get("echo_check", {})
    _check("audit_stamp: keyword scan carries a count and a 64-hex sha256", _nn(ks.get("n_patterns")) and ks["n_patterns"] >= 1 and isinstance(ks.get("sha256"), str) and len(ks["sha256"]) == 64 and all(ch in "0123456789abcdef" for ch in ks["sha256"]), issues)
    _check("audit_stamp: echo scale factors are [name, number] pairs", isinstance(ec_.get("scale_factors"), list) and len(ec_["scale_factors"]) >= 1 and all(isinstance(x, list) and len(x) == 2 for x in ec_["scale_factors"]), issues)
    judged = sorted(c for c, t in conds.items() if "judge" in t)
    _check("audit_stamp: conditions_judged equals the conditions that have a judge section", jd.get("conditions_judged") == judged, issues)
    found = sorted({i for t in conds.values() for i in (t.get("judge_prompt_ids") or [])})
    _check("audit_stamp: prompt_ids_found equals the ids on the judged conditions", jd.get("prompt_ids_found") == found, issues)
    frozen = jd.get("frozen_prompt_id")
    expect_match = None if not judged else (frozen is not None and found == [frozen])
    _check("audit_stamp: all_match_frozen is consistent with the frozen id and the ids found (null when no judge ran)", jd.get("all_match_frozen") == expect_match, issues)
    _check("audit_stamp: frozen_prompt_id is a string or null", frozen is None or isinstance(frozen, str), issues)
    return (not issues), issues


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python validate_h5_schema.py <h5_results.json>"); return 2
    ok, issues = validate_h5_results_schema(json.load(open(sys.argv[1])))
    if ok:
        print("✅ STRUCTURAL VALIDATION PASSED — schema, types, ranges and internal consistency all correct."); return 0
    print(f"❌ STRUCTURAL VALIDATION FAILED — {len(issues)} issue(s):")
    for i in issues: print(f"  - {i}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

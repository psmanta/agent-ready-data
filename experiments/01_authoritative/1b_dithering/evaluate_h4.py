#!/usr/bin/env python3
"""
Evaluate H4 — Dither Type Effects
======================================
The Agentic Data Contract · Pillar 1: Authoritative · Experiment 1b

Does the MECHANISM by which a field becomes wrong change how much it
disrupts the agent's decision, independent of how far the value moved?
Full design history and every locked decision: h4_working_doc.md.

Four analyses:

1. Core comparison — McNemar's, drift vs. plausible and drift vs.
   implausible, per field. Same-population paired comparison, same
   shape as every other such comparison in this project.

2. Style and plausibility contrasts — gee_style_plausibility_test() per
   field, plus the pooled version, gated by
   gee_field_mechanism_interaction_gate(). If the gate says pooling
   isn't safe for a contrast, per-field results are the primary report
   for that contrast; the pooled number is a footnote only.

3. Perturbation characteristics — raw_delta, operator, direction, and
   mechanism_style breakdowns per condition, plus (total_spend
   specifically) whether the implausible-up operator actually escaped
   the field's own bounds, since it only does so for ~26% of customers
   (verified; see h4_working_doc.md).

4. Garbage-filter analysis -- WHY drift differs between plausible and
   implausible values: did the agent notice (explicit detection), silently
   repair (echo class "converted"), or blindly absorb? Per field and arm
   (drift, plausible, implausible, and for churn the two isolated-
   corruption arms), reported for the "stable" stratum (clean attribution:
   a boundary customer drifts at its own clean rate), the "boundary"
   stratum (reported separately, never dropped) and "all":
     - drift rate; keyword-scan detection rate; drift by detected /
       not-detected (the four-way table); drift by echo class; and, when
       judge outputs are supplied, drift by the judge's category.
     - the detection spike: paired McNemar's, implausible vs. plausible,
       on keyword (and judge) detection, with the raw 2x2.
     - the isolation analysis for churn: paired McNemar's, isolated vs.
       propagated, split into contradiction-carrying records (propagation
       would have flipped is_at_risk) and input-identical records (a
       built-in negative control: both arms saw the same input).
   Pre-specified primary tests are tagged "primary": the detection spike
   on stable customers, and the isolation detection comparison on
   contradiction-carrying stable customers. Everything else is exploratory.
   Judge-derived numbers are provisional until the human audit calibrates
   the judge; the keyword scan is anywhere-in-text and cannot say which
   field was doubted. Most comparisons will be underpowered on rare
   events: read the raw 2x2 counts, not just the p-values.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Also find shared/data_generation/, where dither_engine.py actually
# lives (NUMERIC_FIELD_META is needed for the bounds-escape analysis).
# The earlier graceful ImportError fallback silently degraded to None
# when this path was missing -- found by a Tier 1 smoke test against
# the real repo layout, not by any synthetic test, since synthetic
# testing never exercises real directory structure. Project root is
# three levels up from this file (1b_dithering -> 01_authoritative ->
# experiments -> project root).
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "shared" / "data_generation"))

from evaluate_core import (
    load_condition, compute_drift_rate, mcnemar_paired_test,
    align_drift_by_customer, gee_style_plausibility_test,
    gee_field_mechanism_interaction_gate,
    stability_stratum, outcome_rate_by_group, paired_binary_comparison,
    load_judge_results,
)

# Deliberately NOT wrapped in try/except ImportError -- an earlier
# version silently degraded to None here, which masked a real path bug
# (dither_engine.py lives in shared/data_generation/, not alongside this
# file) rather than surfacing it. Found by a Tier 1 smoke test against
# the real repo layout. If this import fails now, it fails loudly.
from dither_engine import NUMERIC_FIELD_META


FIELDS = ["churn_risk_score", "total_spend", "tenure_months"]


def drift_condition_id(field: str) -> str:
    return f"h2_{field}_mag15pct"


def plausible_condition_id(field: str) -> str:
    return f"h4_{field}_plausible"


ISOLATED_PLAUSIBILITIES = ("plausible", "implausible")


def isolated_condition_id(plausibility: str) -> str:
    """Churn with recompute_derived=False: is_at_risk keeps its ORIGINAL value
    instead of following the corrupted score."""
    return f"h4_churn_risk_score_{plausibility}_isolated"


def implausible_condition_id(field: str) -> str:
    return f"h4_{field}_implausible"


def load_all_h4_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads the 9 conditions H4's core analysis needs (3 reused H2 drift
    conditions plus the 6 plausible/implausible conditions) and, when their
    folders exist, the 2 isolated-corruption churn conditions."""
    conditions_dir = experiments_output_dir / "conditions"
    ids = []
    for f in FIELDS:
        ids += [drift_condition_id(f), plausible_condition_id(f), implausible_condition_id(f)]

    loaded = {}
    for cid in ids:
        loaded[cid] = load_condition(conditions_dir / cid, ground_truth)
        print(f"  Loaded {cid}: {len(loaded[cid])} records")
    for p in ISOLATED_PLAUSIBILITIES:
        iso = isolated_condition_id(p)
        if (conditions_dir / iso).exists():
            loaded[iso] = load_condition(conditions_dir / iso, ground_truth)
            print(f"  Loaded {iso}: {len(loaded[iso])} records")
        else:
            print(f"  (not generated: {iso}; the isolation analysis will be skipped)")
    return loaded


# ============================================================================
# 1. CORE COMPARISON — McNemar's, per field
# ============================================================================

def core_mcnemar_comparisons(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """drift-vs-plausible and drift-vs-implausible, per field — the base
    H4 finding before any pooling or GEE modeling."""
    results = {}
    for f in FIELDS:
        drift_recs = loaded[drift_condition_id(f)]
        plaus_recs = loaded[plausible_condition_id(f)]
        implaus_recs = loaded[implausible_condition_id(f)]

        results[f] = {}
        for label, other_recs in [("drift_vs_plausible", plaus_recs),
                                   ("drift_vs_implausible", implaus_recs)]:
            drift_a, drift_b, n_shared = align_drift_by_customer(drift_recs, other_recs)
            results[f][label] = {
                "drift_rate_drift":  round(compute_drift_rate(drift_recs), 4),
                "drift_rate_other":  round(compute_drift_rate(other_recs), 4),
                "n_shared_customers": n_shared,
                **mcnemar_paired_test(drift_a, drift_b),
            }
    return results


# ============================================================================
# 2. STYLE / PLAUSIBILITY CONTRASTS — per field, pooled, interaction-gated
# ============================================================================

def build_mechanism_rows(
    loaded: Dict[str, List[Dict[str, Any]]],
    field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Long-format rows for gee_style_plausibility_test() /
    gee_field_mechanism_interaction_gate(): one row per
    (customer, field, mechanism). field=None pools all 3 fields."""
    target_fields = [field] if field else FIELDS
    rows = []
    for f in target_fields:
        for mech, cid in [("drift", drift_condition_id(f)),
                          ("plausible", plausible_condition_id(f)),
                          ("implausible", implausible_condition_id(f))]:
            for r in loaded[cid]:
                rows.append({"customer_id": r["customer_id"], "field": f,
                            "mechanism": mech, "drift": int(r["drifted"])})
    return rows


def style_plausibility_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Per-field contrasts (always computed and reported), the pooled
    contrast (computed but only a footnote unless the gate clears it),
    and the gate's own result, which decides which one is the headline.
    """
    per_field = {f: gee_style_plausibility_test(build_mechanism_rows(loaded, field=f))
                 for f in FIELDS}

    pooled_rows = build_mechanism_rows(loaded, field=None)
    pooled = gee_style_plausibility_test(pooled_rows, field_col="field")
    gate = gee_field_mechanism_interaction_gate(pooled_rows)

    return {
        "per_field": per_field,
        "pooled": pooled,
        "interaction_gate": gate,
        "headline_source": (
            "per_field"
            if not (gate["safe_to_pool_style"] and gate["safe_to_pool_plausibility"])
            else "pooled"
        ),
    }


# ============================================================================
# 3. PERTURBATION CHARACTERISTICS — raw_delta, operator, direction, style
# ============================================================================

def perturbation_characteristics(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    raw_delta on the field's own native scale (bounded, comparable within
    a field across plausible/implausible -- NOT percentage_delta, which
    was verified unstable; see h4_working_doc.md), plus operator and
    direction breakdowns from the metadata built into the engine.
    """
    results = {}
    for f in FIELDS:
        results[f] = {}
        for plaus, cid in [("plausible", plausible_condition_id(f)),
                           ("implausible", implausible_condition_id(f))]:
            recs = loaded[cid]
            deltas, operators, directions, styles = [], {}, {}, {}
            for r in recs:
                if f not in r["dither_fields"]:
                    continue
                orig = r["dither_original"].get(f)
                curr = r["dither_current_values"].get(f)
                if orig is None or curr is None:
                    continue
                deltas.append(abs(curr - orig))
                op = r["dither_operator"].get(f)
                di = r["dither_direction"].get(f)
                ms = r["dither_mechanism_style"].get(f)
                if op: operators[op] = operators.get(op, 0) + 1
                if di: directions[di] = directions.get(di, 0) + 1
                if ms: styles[ms] = styles.get(ms, 0) + 1

            results[f][plaus] = {
                "n": len(deltas),
                "raw_delta_median": round(sorted(deltas)[len(deltas)//2], 4) if deltas else None,
                "raw_delta_min":    round(min(deltas), 4) if deltas else None,
                "raw_delta_max":    round(max(deltas), 4) if deltas else None,
                "operator_breakdown":  operators,
                "direction_breakdown": directions,
                "mechanism_style_breakdown": styles,
            }
    return results


def total_spend_bounds_escape(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    total_spend's implausible-up operator (x100) only escapes the
    field's own 500,000 ceiling for ~26% of customers (verified; see
    h4_working_doc.md) -- stratifies the implausible condition's drift
    rate by whether the dithered value actually escaped bounds, so a
    "garbage filter" reading isn't confounded with "this value didn't
    even look unusual in the first place."
    """
    meta = NUMERIC_FIELD_META["total_spend"]
    recs = loaded[implausible_condition_id("total_spend")]

    escaped, not_escaped = [], []
    for r in recs:
        if "total_spend" not in r["dither_fields"]:
            continue
        curr = r["dither_current_values"].get("total_spend")
        if curr is None:
            continue
        (escaped if not (meta["min"] <= curr <= meta["max"]) else not_escaped).append(r["drifted"])

    def rate(flags):
        return round(sum(flags) / len(flags), 4) if flags else None

    return {
        "n_escaped_bounds":     len(escaped),
        "n_stayed_in_bounds":   len(not_escaped),
        "drift_rate_escaped":   rate(escaped),
        "drift_rate_in_bounds": rate(not_escaped),
    }


# ============================================================================
# 4. GARBAGE-FILTER ANALYSIS
# ============================================================================

STRATA = ("stable", "boundary", "all")


def _in_stratum(record: Dict[str, Any], stratum: str) -> bool:
    return True if stratum == "all" else stability_stratum(record) == stratum


def _keyword(r):
    return bool(r["h5_keyword_detected"])


def _drift(r):
    return bool(r["drifted"])


def _judge_explicit(r):
    cat = r.get("judge_category")
    return None if cat is None else (cat == "explicit_concern")


def _has_judge(records) -> bool:
    return any(r.get("judge_category") is not None for r in records)


def _rate(records, outcome_fn) -> Dict[str, Any]:
    d = outcome_rate_by_group(records, lambda r: "all", outcome_fn)
    return d.get("all", {"n": 0, "k": 0, "rate": None, "ci": None})


def attach_judge_results(loaded: Dict[str, List[Dict[str, Any]]],
                         judge_dir: Path) -> Dict[str, int]:
    """Join the blind judge's output (classifier_output/<condition>.jsonl, ids
    'customer_id:field') onto each condition's records as judge_category and
    judge_figure_referenced. Returns records joined per condition. Conditions
    with no judge file are left untouched (judge fields simply absent)."""
    joined = {}
    for cid, recs in loaded.items():
        path = Path(judge_dir) / f"{cid}.jsonl"
        if not path.exists():
            continue
        by_id = load_judge_results(path)
        n = 0
        for r in recs:
            if not r["dither_fields"]:
                continue
            row = by_id.get(f'{r["customer_id"]}:{r["dither_fields"][0]}')
            if row:
                r["judge_category"] = row["category"]
                r["judge_figure_referenced"] = row["figure_referenced"]
                n += 1
        joined[cid] = n
    return joined


def _arm_summary(records: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    summary = {
        "n": len(records),
        "drift": _rate(records, _drift),
        "keyword_detected": _rate(records, _keyword),
        "drift_by_keyword": outcome_rate_by_group(
            records, lambda r: "detected" if r["h5_keyword_detected"] else "not_detected"),
        "drift_by_echo_class": outcome_rate_by_group(
            records, lambda r: r["value_echo"].get(field, {}).get("echo_class")),
    }
    if _has_judge(records):
        summary["judge_explicit_concern"] = _rate(records, _judge_explicit)
        summary["drift_by_judge_category"] = outcome_rate_by_group(
            records, lambda r: r.get("judge_category"))
    return summary


def _arm_ids(field: str) -> Dict[str, str]:
    ids = {"drift": drift_condition_id(field), "plausible": plausible_condition_id(field),
           "implausible": implausible_condition_id(field)}
    if field == "churn_risk_score":
        for p in ISOLATED_PLAUSIBILITIES:
            ids[f"{p}_isolated"] = isolated_condition_id(p)
    return ids


def _compare(records_a, records_b, customers, primary_keyword: bool) -> Dict[str, Any]:
    """Paired comparisons (a vs. b) on keyword detection, drift, and -- when
    both arms carry judge results -- judge explicit_concern."""
    entry = {
        "keyword_detection": {**paired_binary_comparison(records_a, records_b, _keyword, customers),
                              "primary": primary_keyword},
        "drift": {**paired_binary_comparison(records_a, records_b, _drift, customers),
                  "primary": False},
    }
    if _has_judge(records_a) and _has_judge(records_b):
        entry["judge_explicit_concern"] = {
            **paired_binary_comparison(records_a, records_b, _judge_explicit, customers),
            "primary": primary_keyword}
    return entry


def _detection_spike(loaded) -> Dict[str, Any]:
    """Implausible vs. plausible, same customers. Garbage-filter reading:
    detection should spike on implausible values (a_only > b_only)."""
    out = {"orientation": "a = implausible, b = plausible; a_only = outcome under implausible only"}
    for field in FIELDS:
        plaus, implaus = loaded[plausible_condition_id(field)], loaded[implausible_condition_id(field)]
        out[field] = {}
        for s in STRATA:
            customers = {r["customer_id"] for r in plaus if _in_stratum(r, s)}
            out[field][s] = _compare(implaus, plaus, customers, primary_keyword=(s == "stable"))
    return out


def _isolation_design(output_dir: Path, prop_id: str, iso_id: str) -> Dict[str, Any]:
    """Reads both arms' dither_reference.json. Verifies the matched-pair
    assumption (identical corrupted churn value per customer) and splits
    customers into contradiction-carrying (propagation flipped is_at_risk) and
    input-identical (both arms saw exactly the same record)."""
    def ref(cid):
        with open(output_dir / "conditions" / cid / "dither_reference.json") as f:
            return {r["customer_id"]: r for r in json.load(f)}
    a, b = ref(prop_id), ref(iso_id)
    shared = sorted(set(a) & set(b))
    same_value = sum(a[c]["churn_risk_score"] == b[c]["churn_risk_score"] for c in shared)
    cc = {c for c in shared if a[c]["is_at_risk"] != b[c]["is_at_risk"]}
    ii = {c for c in shared if a[c]["is_at_risk"] == b[c]["is_at_risk"]}
    return {"n_shared": len(shared), "n_identical_value": same_value, "cc": cc, "ii": ii,
            "design_ok": bool(shared) and same_value == len(shared)}


def isolation_analysis(loaded, experiments_output_dir: Path) -> Dict[str, Any]:
    """Isolated vs. propagated churn corruption. The redundancy hypothesis
    predicts higher detection in the isolated arm among contradiction-carrying
    records and NO difference among input-identical ones (a negative control:
    identical inputs, so any difference there is agent run-to-run noise)."""
    out = {"orientation": ("a = isolated, b = propagated; a_only = outcome in the isolated "
                           "arm only. Hypothesis: keyword/judge detection a_only > b_only among "
                           "contradiction_carrying, ~equal among input_identical. Drift is exploratory.")}
    for p in ISOLATED_PLAUSIBILITIES:
        prop_id = plausible_condition_id("churn_risk_score") if p == "plausible" \
            else implausible_condition_id("churn_risk_score")
        iso_id = isolated_condition_id(p)
        if iso_id not in loaded or prop_id not in loaded:
            out[p] = {"status": "not_loaded"}
            continue
        design = _isolation_design(Path(experiments_output_dir), prop_id, iso_id)
        prop, iso = loaded[prop_id], loaded[iso_id]
        stable_customers = {r["customer_id"] for r in prop if _in_stratum(r, "stable")}
        block = {"status": "ok",
                 "design_check": {"n_shared_customers": design["n_shared"],
                                  "n_identical_corrupted_value": design["n_identical_value"],
                                  "n_contradiction_carrying": len(design["cc"]),
                                  "n_input_identical": len(design["ii"]),
                                  "design_ok": design["design_ok"]}}
        for name, subset in (("contradiction_carrying", design["cc"]), ("input_identical", design["ii"])):
            block[name] = {}
            for s in ("all", "stable"):
                customers = subset if s == "all" else subset & stable_customers
                block[name][s] = _compare(iso, prop, customers,
                                          primary_keyword=(name == "contradiction_carrying" and s == "stable"))
        out[p] = block
    return out


def _primary_tests(spike: Dict[str, Any], isolation: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-specified primary tests, gathered into families with Holm
    adjustment so no one reads five raw p-values as five independent
    findings. Two families, each corrected separately: keyword detection and
    (once validated) judge explicit_concern. Each family holds the 3 detection
    spikes (implausible vs. plausible, stable customers) plus the 2 isolation
    comparisons (isolated vs. propagated, contradiction-carrying stable
    customers). All five predict the same direction: a_only > b_only."""
    from statsmodels.stats.multitest import multipletests
    families: Dict[str, List] = {"keyword_detection": [], "judge_explicit_concern": []}
    for field in FIELDS:
        for test in families:
            c = spike.get(field, {}).get("stable", {}).get(test)
            if c and c.get("primary") and c.get("n_pairs"):
                families[test].append((f"detection_spike.{field}.stable.{test}", c))
    for p in ISOLATED_PLAUSIBILITIES:
        block = isolation.get(p, {})
        if block.get("status") != "ok":
            continue
        for test in families:
            c = block["contradiction_carrying"]["stable"].get(test)
            if c and c.get("primary") and c.get("n_pairs"):
                families[test].append((f"isolation.{p}.contradiction_carrying.stable.{test}", c))
    out = {}
    for fam, items in families.items():
        if not items:
            continue
        adj = multipletests([c["p_value"] for _, c in items], method="holm")[1]
        out[fam] = [{"test": path, "a_only": c["a_only"], "b_only": c["b_only"],
                     "n_discordant": c["n_discordant"], "p_value": c["p_value"],
                     "p_holm": float(a), "predicted_direction_observed": c["a_only"] > c["b_only"],
                     "low_power": c["low_power"]} for (path, c), a in zip(items, adj)]
    return out


def garbage_filter_analysis(
    loaded: Dict[str, List[Dict[str, Any]]],
    experiments_output_dir: Path,
    judge_joined: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    arms = {}
    for field in FIELDS:
        arms[field] = {}
        for arm, cid in _arm_ids(field).items():
            if cid not in loaded:
                continue
            arms[field][arm] = {s: _arm_summary([r for r in loaded[cid] if _in_stratum(r, s)], field)
                                for s in STRATA}
    spike = _detection_spike(loaded)
    isolation = isolation_analysis(loaded, experiments_output_dir)
    return {
        "status": "implemented",
        "strata": {"stable": "stability_tier == stable (clean attribution)",
                   "boundary": "lightly_boundary, deeply_boundary, tied_no_majority (reported "
                               "separately, never dropped)",
                   "all": "every customer"},
        "judge_records_joined": judge_joined or {},
        "arms": arms,
        "detection_spike": spike,
        "isolation": isolation,
        "primary_tests": _primary_tests(spike, isolation),
        "multiplicity": "Holm adjustment within each family (keyword; judge); two-sided exact McNemar.",
        "caveats": [
            "Keyword detection is anywhere-in-text: it cannot say which field was doubted.",
            "Judge-derived fields are provisional until the human audit calibrates the judge.",
            "Rare events: most paired comparisons are low_power; read the raw 2x2 counts.",
            "Echo class 'omitted' means the NUMBER is absent, not the field.",
            "Only tests tagged primary were pre-specified; all others are exploratory.",
        ],
    }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def evaluate_h4(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
    judge_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    print("Loading H4 conditions (3 reused H2 drift + 6 plausible/implausible + 2 isolated churn)...")
    loaded = load_all_h4_conditions(experiments_output_dir, ground_truth)
    judge_joined = attach_judge_results(loaded, judge_dir) if judge_dir else None
    if judge_dir:
        print(f"  Judge results joined: {judge_joined or 'none found in ' + str(judge_dir)}")

    print("\n1. Core comparison (McNemar's: drift vs. plausible, drift vs. implausible)...")
    core = core_mcnemar_comparisons(loaded)

    print("2. Style and plausibility contrasts (per-field, pooled, interaction-gated)...")
    contrasts = style_plausibility_analysis(loaded)

    print("3. Perturbation characteristics (raw_delta, operator, direction, mechanism_style)...")
    characteristics = perturbation_characteristics(loaded)
    bounds_escape = total_spend_bounds_escape(loaded)

    print("4. Garbage-filter analysis...")
    garbage_filter = garbage_filter_analysis(loaded, experiments_output_dir, judge_joined)

    return {
        "hypothesis": "H4",
        "core_mcnemar_comparisons": core,
        "style_plausibility_contrasts": contrasts,
        "perturbation_characteristics": characteristics,
        "total_spend_bounds_escape": bounds_escape,
        "garbage_filter_analysis": garbage_filter,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_output", type=str, default="experiments_output")
    parser.add_argument("--finalized_ground_truth", type=str,
        default="experiments_output/finalized_ground_truth.json")
    parser.add_argument("--judge_dir", type=str, default=None,
        help="directory of blind-judge outputs (classifier_output/); optional")
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h4_results.json")
    args = parser.parse_args()

    print(f"\n{'='*60}\nH4 Evaluation — Dither Type Effects\n{'='*60}\n")
    results = evaluate_h4(Path(args.experiments_output), Path(args.finalized_ground_truth),
                          Path(args.judge_dir) if args.judge_dir else None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}\nSaved: {out_path}\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

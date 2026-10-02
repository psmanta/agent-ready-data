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

4. Garbage-filter analysis — NOT YET IMPLEMENTED. Depends on H5's
   detection regex, which was fully designed in conversation but never
   actually committed to code anywhere in this repo (confirmed by
   search before writing this file) -- my own recollection of its exact
   pattern count doesn't even internally reconcile (recalled as "17
   patterns" but enumerating the categories from memory gives ~25), so
   reconstructing it from memory and presenting it as "the frozen list"
   would be exactly the overconfident-recall mistake this project has
   repeatedly caught elsewhere. Needs the authoritative list restored or
   explicitly rebuilt and reviewed before this analysis can run.
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


def implausible_condition_id(field: str) -> str:
    return f"h4_{field}_implausible"


def load_all_h4_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads all 9 conditions H4's analysis needs: 3 reused H2 drift
    conditions plus the 6 new H4 plausible/implausible conditions."""
    conditions_dir = experiments_output_dir / "conditions"
    ids = []
    for f in FIELDS:
        ids += [drift_condition_id(f), plausible_condition_id(f), implausible_condition_id(f)]

    loaded = {}
    for cid in ids:
        loaded[cid] = load_condition(conditions_dir / cid, ground_truth)
        print(f"  Loaded {cid}: {len(loaded[cid])} records")
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
# 4. GARBAGE-FILTER ANALYSIS — NOT YET IMPLEMENTED
# ============================================================================

def garbage_filter_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Intentionally not implemented. See module docstring and
    h4_working_doc.md: this needs H5's detection regex, which was
    designed in conversation but never committed to code, and my own
    recollection of it doesn't internally reconcile on pattern count.
    Reconstructing and presenting a guessed list as "the frozen list"
    would risk corrupting H5's own eventual build with an unreviewed
    pattern set. Blocked pending that list being restored or rebuilt
    and reviewed as its own step.
    """
    return {
        "status": "not_implemented",
        "reason": ("Depends on H5's detection regex, which exists only in "
                  "design history, not in committed code. See module "
                  "docstring."),
    }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def evaluate_h4(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
) -> Dict[str, Any]:
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    print("Loading all 9 H4 conditions (3 reused H2 drift + 6 new plausible/implausible)...")
    loaded = load_all_h4_conditions(experiments_output_dir, ground_truth)

    print("\n1. Core comparison (McNemar's: drift vs. plausible, drift vs. implausible)...")
    core = core_mcnemar_comparisons(loaded)

    print("2. Style and plausibility contrasts (per-field, pooled, interaction-gated)...")
    contrasts = style_plausibility_analysis(loaded)

    print("3. Perturbation characteristics (raw_delta, operator, direction, mechanism_style)...")
    characteristics = perturbation_characteristics(loaded)
    bounds_escape = total_spend_bounds_escape(loaded)

    print("4. Garbage-filter analysis...")
    garbage_filter = garbage_filter_analysis(loaded)

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
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h4_results.json")
    args = parser.parse_args()

    print(f"\n{'='*60}\nH4 Evaluation — Dither Type Effects\n{'='*60}\n")
    results = evaluate_h4(Path(args.experiments_output), Path(args.finalized_ground_truth))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}\nSaved: {out_path}\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

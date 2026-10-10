#!/usr/bin/env python3
"""
Evaluate H1 — Field Importance Validation
=============================================
The Agentic Data Contract · Pillar 1: Authoritative · Experiment 1b

Answers H1's four sub-questions from 1b_DESIGN_AMENDMENT_1.md:
  A. Does the self-reported top-5 field set (from 1a's H4) produce more
     decision drift than fields the agent did NOT self-report as
     important?
  B. Does dithering behave differently depending on which conceptual
     category of data it hits?
  C. Within a category, does one field carry disproportionate weight,
     or is the category's effect evenly distributed?
  D. Does identity data — assumed decision-irrelevant — actually behave
     as inert as assumed?

Plus two free analyses: category impact ranking, and stated-vs-revealed
field importance.

CROSS-HYPOTHESIS DATA REUSE: Question A's comparison group draws 4 of
its 6 fields from H2 and H3's condition folders (total_spend,
tenure_months, avg_resolution_time_hours, refund_rate) rather than
duplicating data generation — consistent with the pattern already
established for H2/H4's field reuse and H3's free 2x2 directionality.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_core import (
    load_condition, compute_drift_rate, compute_effective_drift_rate,
    attribute_by_field, mcnemar_paired_test, mann_whitney_test,
    align_drift_by_customer,
)


# ============================================================================
# QUESTION A — FIELD GROUPS
# ============================================================================
# Corrected 2026-08: email and is_vip were added specifically as
# predicted-null COMPARISON fields, not top-5 fields. An earlier version
# of this design mistakenly grouped them with the top-5 set (7v4 instead
# of 5v6) — caught before this file was built against the wrong grouping.

TOP5_FIELDS = {
    "h1_individual_last_purchase_days_ago":  "last_purchase_days_ago",
    "h1_individual_churn_risk_score":        "churn_risk_score",
    "h1_individual_nps_score":               "nps_score",
    "h1_individual_lifetime_value_estimate": "lifetime_value_estimate",
    "h1_individual_support_tickets_open":    "support_tickets_open",
}

# Comparison fields — 2 native to H1 (added specifically for this
# purpose), 4 reused from H2/H3's condition folders (cross-hypothesis
# data reuse, not duplicated generation).
COMPARISON_FIELDS = {
    "h1_individual_email":                     ("email", "1b_dithering"),
    "h1_individual_is_vip":                    ("is_vip", "1b_dithering"),
    "h2_total_spend_mag15pct":                 ("total_spend", "1b_dithering"),
    "h2_tenure_months_mag15pct":               ("tenure_months", "1b_dithering"),
    "h3_individual_avg_resolution_time_hours": ("avg_resolution_time_hours", "1b_dithering"),
    "h3_individual_refund_rate":               ("refund_rate", "1b_dithering"),
}

CATEGORY_CONDITIONS = [
    "h1_category_identity",
    "h1_category_purchase_behavior",
    "h1_category_engagement",
    "h1_category_risk_factors",
    "h1_category_segmentation",
    "h1_category_account_status",
]


def load_all_h1_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
    extra_condition_ids: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load every condition H1's analysis needs — its own 14 conditions,
    plus reaching into H2 and H3's folders for Question A's 4 reused
    comparison fields. All conditions live under the same
    experiments_output/conditions/ directory regardless of which
    hypothesis file originally defined them, so this is a normal
    load_condition() call per condition_id, not a special cross-
    hypothesis mechanism.

    extra_condition_ids: further conditions named by the frozen replication
    manifest (Question A on the 1b list): H3's payment_failures, and any
    h1_replication_* conditions. Loaded only when the trigger fired.

    Returns {condition_id: joined_records} for every condition loaded.
    """
    conditions_dir = experiments_output_dir / "conditions"
    all_condition_ids = (
        list(TOP5_FIELDS.keys())
        + list(COMPARISON_FIELDS.keys())
        + CATEGORY_CONDITIONS
        + ["h1_distributed"]
    )
    for cid in (extra_condition_ids or []):
        if cid not in all_condition_ids:
            all_condition_ids.append(cid)

    loaded = {}
    for cid in all_condition_ids:
        cond_dir = conditions_dir / cid
        loaded[cid] = load_condition(cond_dir, ground_truth)
        print(f"  Loaded {cid}: {len(loaded[cid])} records")

    return loaded


# ============================================================================
# QUESTION A — self-reported top-5 vs. everything else
# ============================================================================

def _question_a(
    loaded: Dict[str, List[Dict[str, Any]]],
    top: Dict[str, str],
    comparison: Dict[str, str],
    include_pairs: bool = True,
) -> Dict[str, Any]:
    """
    Two complementary analyses, per 1b_DESIGN_AMENDMENT_1.md — both on
    EXPOSURE-ADJUSTED drift (drift among customers actually perturbed).

    Why exposure-adjusted: an exposure audit (2026-08) showed fields
    differ enormously in how many customers a condition touches — is_vip
    flips for ~15% of customers by design, versus ~100% for
    churn_risk_score. Raw drift rates would partly measure how many
    customers were touched, not how much the agent cares, which would
    bias the top-5 vs. comparison contrast (is_vip sits in the comparison
    group). Raw rates and exposure are still reported alongside, so the
    adjustment is visible rather than silent.

    1. Group-level check (SECONDARY): Mann-Whitney comparing 5 top-5
       effective drift rates against 6 comparison effective drift rates —
       independent summary statistics, one per field. Explicitly
       low-powered (n=5 vs n=6 fields); caveat attached to the result.

    2. Per-field check (PRIMARY): 30 pairwise McNemar's tests (5x6), each
       restricted to customers perturbed in BOTH conditions — the paired,
       exposure-adjusted comparison. Headline finding is whether the
       top-5 advantage holds CONSISTENTLY across pairs, not any single
       p-value in isolation.
    """
    def rates_for(cid, field):
        r = compute_effective_drift_rate(loaded[cid], field=field)
        return r

    top5 = {cid: rates_for(cid, f) for cid, f in top.items()}
    comparison_rates = {cid: rates_for(cid, f) for cid, f in comparison.items()}

    def summarize(rate_dict, name_of):
        return {
            name_of(cid): {
                "effective_drift_rate": round(r["effective_drift_rate"], 4) if r["effective_drift_rate"] is not None else None,
                "raw_drift_rate":       round(r["raw_drift_rate"], 4),
                "exposure":             round(r["exposure"], 4),
                "n_perturbed":          r["n_perturbed"],
            } for cid, r in rate_dict.items()
        }

    top5_eff = [r["effective_drift_rate"] for r in top5.values() if r["effective_drift_rate"] is not None]
    comp_eff = [r["effective_drift_rate"] for r in comparison_rates.values() if r["effective_drift_rate"] is not None]

    # --- Secondary: group-level Mann-Whitney on effective rates ---
    group_result = mann_whitney_test(top5_eff, comp_eff, label_a="top5", label_b="comparison")
    group_result["power_caveat"] = (
        f"LOW POWER: n={len(top)} vs n={len(comparison)} fields. True sample size for this "
        "group-level question is the number of FIELDS tested, not the "
        "number of customers per field (pseudo-replication) — a null "
        "result here means 'not enough fields tested to detect an "
        "effect at this sample size,' not 'no effect exists.'"
    )

    # --- Primary: pairwise McNemar's, perturbed-in-both customers only ---
    pairwise_results = []
    top5_wins = comparison_wins = 0
    for top5_cid, top5_field in top.items():
        for comp_cid, comp_field in comparison.items():
            drift_top5, drift_comp, n_shared = align_drift_by_customer(
                loaded[top5_cid], loaded[comp_cid], perturbed_only=True)
            mcnemar_result = mcnemar_paired_test(drift_top5, drift_comp)

            t_rate = top5[top5_cid]["effective_drift_rate"] or 0.0
            c_rate = comparison_rates[comp_cid]["effective_drift_rate"] or 0.0
            if t_rate > c_rate:
                top5_wins += 1
            elif c_rate > t_rate:
                comparison_wins += 1

            pairwise_results.append({
                "top5_field":                  top5_field,
                "comparison_field":            comp_field,
                "top5_effective_drift_rate":   round(t_rate, 4),
                "comparison_effective_drift_rate": round(c_rate, 4),
                "n_perturbed_in_both":         n_shared,
                **mcnemar_result,
            })

    n_pairs = len(pairwise_results)
    n_significant = sum(1 for r in pairwise_results if r["p_value"] < 0.05)

    pairwise_block = {
        "n_pairs":                       n_pairs,
        "n_significant_p05":             n_significant,
        "top5_higher_drift_count":       top5_wins,
        "comparison_higher_drift_count": comparison_wins,
        "consistency_note": (
            f"Top-5 field showed higher exposure-adjusted drift in "
            f"{top5_wins}/{n_pairs} pairwise comparisons. Headline finding "
            f"is this consistency across comparisons, not any single "
            f"pair's p-value in isolation."
        ),
    }
    if include_pairs:
        pairwise_block["all_pairs"] = pairwise_results

    return {
        "adjustment_note": (
            "All comparisons use EXPOSURE-ADJUSTED drift (drift among "
            "customers actually perturbed). Raw rates and exposure shown "
            "per field for transparency."
        ),
        "top5_fields":       summarize(top5, lambda cid: top[cid]),
        "comparison_fields": summarize(comparison_rates, lambda cid: comparison[cid]),
        "group_level_check_secondary": group_result,
        "pairwise_mcnemar_primary": pairwise_block,
    }


def question_a_analysis(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Question A on the LEGACY list (1a's top 5 against the amendment's six comparison fields).
    Output is unchanged from before the manifest existed; see _question_a for the method."""
    return _question_a(loaded, TOP5_FIELDS, {cid: f for cid, (f, _) in COMPARISON_FIELDS.items()})


# ============================================================================
# QUESTION A on the 1b list — driven ENTIRELY by the frozen replication manifest
# ============================================================================
# The evaluator never re-derives the trigger or the groups: it verifies the
# manifest's decision_sha256, checks that the files the manifest was computed
# from are unchanged, and reads the pre-registered groups from it.

def load_and_check_manifest(manifest_path: Path, experiments_output_dir: Path) -> Dict[str, Any]:
    """Load the manifest; refuse one that was edited, or whose inputs have changed since it was frozen,
    or whose legacy groups disagree with this evaluator's own constants."""
    import hashlib
    import h1_baseline_replication as H
    manifest = H.load_manifest(manifest_path)          # ValueError if edited after it was frozen
    problems: List[str] = []

    legacy = manifest["question_a_groups"]["legacy_1a"]
    if [(e["condition_id"], e["field"]) for e in legacy["top"]] != list(TOP5_FIELDS.items()):
        problems.append("the manifest's legacy top group differs from this evaluator's TOP5_FIELDS")
    if [(e["condition_id"], e["field"]) for e in legacy["comparison"]] != [(c, f) for c, (f, _) in COMPARISON_FIELDS.items()]:
        problems.append("the manifest's legacy comparison group differs from this evaluator's COMPARISON_FIELDS")

    def sha(p: Path) -> Optional[str]:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    base = experiments_output_dir / "baseline"
    inputs = manifest["inputs"]
    for name, recorded in inputs["run_files"].items():
        if sha(base / "decisions" / name) != recorded:
            problems.append(f"baseline/decisions/{name} is missing or changed since the manifest was frozen")
    bi = inputs["baseline_input"]
    if sha(base / "agent_input" / bi["name"]) != bi["sha256"]:
        problems.append(f"baseline/agent_input/{bi['name']} is missing or changed since the manifest was frozen")
    if inputs.get("canonical"):
        c = inputs["canonical"]
        if sha(experiments_output_dir / "ground_truth" / c["name"]) != c["sha256"]:
            problems.append(f"ground_truth/{c['name']} is missing or changed since the manifest was frozen")
    if problems:
        raise ValueError("replication manifest does not match the data it governs:\n  - " + "\n  - ".join(problems))
    return manifest


def manifest_required_conditions(manifest: Dict[str, Any]) -> List[str]:
    """Condition ids the 1b analysis needs (empty unless the trigger fired)."""
    p = manifest["question_a_groups"]["primary_1b"]
    if not manifest["decision"]["triggered"] or p is None:
        return []
    return [e["condition_id"] for e in p["top"]] + [e["condition_id"] for e in p["comparison"]]


def question_a_1b_analysis(loaded: Dict[str, List[Dict[str, Any]]], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Question A on the 1b list (primary when the trigger fired), plus the pre-registered sensitivity
    analysis that drops comparison fields at raw 1b citation ranks 6-8."""
    p = manifest["question_a_groups"]["primary_1b"]
    to_map = lambda entries: {e["condition_id"]: e["field"] for e in entries}
    top, comp = to_map(p["top"]), to_map(p["comparison"])
    sens = to_map(p["comparison_sensitivity_excluding_1b_ranks_6_to_8"])
    return {
        "status": "ok",
        "role": "primary",
        "manifest_decision_sha256": manifest["decision_sha256"],
        "top_group": p["top"],
        "comparison_group": p["comparison"],
        "analysis": _question_a(loaded, top, comp),
        "sensitivity_excluding_1b_ranks_6_to_8": {
            "comparison_group": p["comparison_sensitivity_excluding_1b_ranks_6_to_8"],
            "analysis": _question_a(loaded, top, sens, include_pairs=False) if sens else
                        {"status": "no comparison fields remain after the exclusion"},
        },
    }


# ============================================================================
# QUESTION B — category impact ranking (free analysis)
# ============================================================================

def question_b_category_ranking(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Which conceptual category of data matters most when dithered?
    Directly analogous to 1a's decision-cliff framing, but for category
    TYPE rather than duplication volume. Zero additional API cost — the
    6 category conditions already exist for Question B's own sake.
    """
    rates = {cid: compute_drift_rate(loaded[cid]) for cid in CATEGORY_CONDITIONS}
    ranked = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "ranked_categories": [
            {"condition_id": cid, "drift_rate": round(rate, 4)}
            for cid, rate in ranked
        ],
    }


# ============================================================================
# QUESTION C — per-field attribution within category conditions
# ============================================================================

def question_c_attribution(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Within a category, does one field carry disproportionate weight, or
    is the effect evenly distributed? Uses attribute_by_field() (built
    in evaluate_core.py) against every category condition — this is
    where the is_vip-style asymmetry, if present in h1_category_
    account_status, would surface directly, rather than being hidden
    inside one bundled drift rate.
    """
    return {
        cid: attribute_by_field(loaded[cid])
        for cid in CATEGORY_CONDITIONS
    }


# ============================================================================
# QUESTION D — identity data null-hypothesis check
# ============================================================================

def question_d_null_check(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Does identity data — assumed decision-irrelevant — actually behave
    as inert as assumed? Reports h1_category_identity (bundled) and
    h1_individual_email (isolated) together, since a surprising result
    on either would be a genuinely notable finding, analogous to H8a's
    negative-control framing.
    """
    category_rate = compute_drift_rate(loaded["h1_category_identity"])
    email_rate = compute_drift_rate(loaded["h1_individual_email"])
    return {
        "h1_category_identity_drift_rate": round(category_rate, 4),
        "h1_individual_email_drift_rate":  round(email_rate, 4),
        "note": (
            "Both predicted near-zero. A meaningfully elevated rate on "
            "either is a notable, surprising finding worth headline "
            "treatment, not a footnote — the same framing used for "
            "H8a's negative-control pair."
        ),
    }


# ============================================================================
# STATED VS. REVEALED FIELD IMPORTANCE (free analysis)
# ============================================================================

def stated_vs_revealed_importance(
    baseline_reference_path: Path,
    loaded: Dict[str, List[Dict[str, Any]]],
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Does the agent's self-reported "key_factors" predict what actually
    changes its decisions? Stated importance: how often each field is
    cited in key_factors across all baseline decisions (from
    baseline_reference.json's run_details — NOT carried into the
    finalized ground truth file, so this reads baseline_reference.json
    directly). Revealed importance: actual drift rate when that field is
    individually dithered (from the 5 top-5 conditions only — the only
    fields with clean individual-condition drift rates that also appear
    meaningfully in key_factors citations).
    """
    with open(baseline_reference_path) as f:
        baseline = json.load(f)

    citation_counts: Dict[str, int] = {}
    total_runs = 0
    for customer_id, entry in baseline["customers"].items():
        for run in entry["run_details"]:
            total_runs += 1
            for field in run.get("key_factors", []):
                citation_counts[field] = citation_counts.get(field, 0) + 1

    stated_importance = {
        field: round(count / total_runs, 4)
        for field, count in citation_counts.items()
    }

    revealed_importance = {
        TOP5_FIELDS[cid]: round(compute_drift_rate(loaded[cid]), 4)
        for cid in TOP5_FIELDS
    }

    comparison = []
    for cid, field in TOP5_FIELDS.items():
        comparison.append({
            "field": field,
            "stated_citation_rate": stated_importance.get(field, 0.0),
            "revealed_drift_rate":  revealed_importance[field],
        })
    comparison.sort(key=lambda r: r["revealed_drift_rate"], reverse=True)

    result = {
        "stated_importance_all_fields": stated_importance,
        "top5_comparison": comparison,
    }

    if manifest is not None:
        # The manifest's citation table is the PRE-REGISTERED measurement of stated importance (exact
        # schema matching, once per decision). The raw-string rates above are kept unchanged; the
        # manifest rate is added beside them so any disagreement between the two instruments is visible.
        rate_of = {row["field"]: row["rate"] for row in manifest["citation_table"]}
        for row in comparison:
            row["manifest_citation_rate"] = rate_of.get(row["field"], 0.0)
        p = manifest["question_a_groups"]["primary_1b"]
        if manifest["decision"]["triggered"] and p is not None and any(e["condition_id"] not in loaded for e in p["top"]):
            # --allow_incomplete_1b: some 1b conditions have no decisions yet, so no rows can be built
            result["top5_comparison_1b"] = {"role": "primary", "status": "incomplete",
                                            "missing_conditions": [e["condition_id"] for e in p["top"] if e["condition_id"] not in loaded]}
        elif manifest["decision"]["triggered"] and p is not None:
            rows = []
            for e in p["top"]:
                rows.append({
                    "field": e["field"],
                    "condition_id": e["condition_id"],
                    "in_1a_top5": e["field"] in TOP5_FIELDS.values(),
                    "stated_citation_rate": rate_of.get(e["field"], 0.0),
                    "revealed_drift_rate": round(compute_drift_rate(loaded[e["condition_id"]]), 4),
                })
            rows.sort(key=lambda r: r["revealed_drift_rate"], reverse=True)
            result["top5_comparison_1b"] = {
                "role": "primary",
                "stated_source": "manifest citation_table (pre-registered measurement)",
                "rows": rows,
            }
    return result


# ============================================================================
# H1_DISTRIBUTED — investigatory, not exhaustive
# ============================================================================

def h1_distributed_report(loaded: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    One distributed combination cannot characterize the full space of
    possible distributed dither patterns — reported with its scope note
    attached directly to the result, not left to documentation alone.
    """
    return {
        "drift_rate": round(compute_drift_rate(loaded["h1_distributed"]), 4),
        "scope_note": (
            "Investigatory, not exhaustive. A null or positive result "
            "scopes deeper combinatorial work into Phase 2 rather than "
            "being treated as conclusive alone."
        ),
    }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def evaluate_h1(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
    baseline_reference_path: Path,
    manifest_path: Optional[Path] = None,
    allow_incomplete_1b: bool = False,
) -> Dict[str, Any]:
    """
    Run the full H1 evaluation: all four sub-questions plus the two free
    analyses, against real generated data.

    manifest_path: the frozen replication manifest (h1_replication_manifest.json). When it exists, its
    hash and inputs are verified and Question A is also run on the 1b list if the trigger fired.
    """
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    manifest: Optional[Dict[str, Any]] = None
    qa_manifest: Dict[str, Any]
    extra_ids: List[str] = []
    missing: List[str] = []
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = load_and_check_manifest(Path(manifest_path), experiments_output_dir)
        triggered = manifest["decision"]["triggered"]
        qa_manifest = {"status": "used", "decision_sha256": manifest["decision_sha256"], "triggered": triggered,
                       "primary_list": manifest["question_a_groups"]["primary_list"],
                       "legacy_role": "legacy comparison" if triggered else "primary"}
        extra_ids = manifest_required_conditions(manifest)
        missing = [c for c in dict.fromkeys(extra_ids)
                   if not (experiments_output_dir / "conditions" / c / "decisions.jsonl").exists()]
        if missing and not allow_incomplete_1b:
            raise SystemExit(
                "Question A on the 1b list is PRE-REGISTERED as primary (the replication trigger fired) but these "
                f"conditions have no decisions yet: {', '.join(missing)}. Generate them "
                "(generate_h1_replication_conditions.py) and run them through the agent, or pass "
                "--allow_incomplete_1b to proceed knowingly without the primary analysis.")
    else:
        qa_manifest = {"status": "no_manifest",
                       "note": f"No replication manifest at {manifest_path}: Question A on the 1b list was NOT run. "
                               "The legacy-list analysis below is unchanged."}
        print(f"  WARNING: {qa_manifest['note']}")

    print("Loading all 18 conditions (14 native + 4 reused from H2/H3)...")
    loaded = load_all_h1_conditions(experiments_output_dir, ground_truth,
                                    extra_condition_ids=[c for c in extra_ids if c not in missing])

    print("\nRunning Question A (field importance: top-5 vs. comparison)...")
    question_a = question_a_analysis(loaded)

    if manifest is None:
        question_a_1b: Dict[str, Any] = {"status": "skipped: no manifest"}
    elif not manifest["decision"]["triggered"]:
        question_a_1b = {"status": "not_triggered",
                         "note": "Top-5 overlap was above the pre-registered trigger; Question A stands on the legacy list."}
    elif missing:
        question_a_1b = {"status": "incomplete", "missing_conditions": missing}
    else:
        print("Running Question A on the 1b list (primary: the replication trigger fired)...")
        question_a_1b = question_a_1b_analysis(loaded, manifest)

    print("Running Question B (category impact ranking)...")
    question_b = question_b_category_ranking(loaded)

    print("Running Question C (per-field attribution within categories)...")
    question_c = question_c_attribution(loaded)

    print("Running Question D (identity null-hypothesis check)...")
    question_d = question_d_null_check(loaded)

    print("Running stated-vs-revealed importance...")
    stated_revealed = stated_vs_revealed_importance(baseline_reference_path, loaded, manifest)

    print("Compiling h1_distributed report...")
    distributed = h1_distributed_report(loaded)

    return {
        "hypothesis": "H1",
        "question_a_field_importance": question_a,
        "question_a_manifest": qa_manifest,
        "question_a_1b_list": question_a_1b,
        "question_b_category_ranking": question_b,
        "question_c_field_attribution": question_c,
        "question_d_identity_null_check": question_d,
        "stated_vs_revealed_importance": stated_revealed,
        "h1_distributed": distributed,
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
    parser.add_argument("--manifest", type=str,
        default="experiments_output/evaluation/h1_replication_manifest.json",
        help="frozen replication manifest; enables Question A on the 1b list")
    parser.add_argument("--allow_incomplete_1b", action="store_true",
        help="proceed even if conditions the manifest requires have no decisions yet")
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h1_results.json")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("H1 Evaluation — Field Importance Validation")
    print(f"{'='*60}\n")

    results = evaluate_h1(
        Path(args.experiments_output),
        Path(args.finalized_ground_truth),
        Path(args.baseline_reference),
        manifest_path=Path(args.manifest),
        allow_incomplete_1b=args.allow_incomplete_1b,
    )

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

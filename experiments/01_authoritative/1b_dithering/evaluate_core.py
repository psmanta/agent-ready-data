#!/usr/bin/env python3
"""
Evaluate Core — Experiment 1b: Dithering
============================================
The Agentic Data Contract · Pillar 1: Authoritative

Shared primitives used by every evaluate_h*.py hypothesis file. "Bard
Hall in one file" — every statistical tool used anywhere in 1b's
evaluation lives here, implemented and verified once, rather than
re-derived per hypothesis file where a subtle divergence could hide.

This module does three distinct jobs:

1. GROUND TRUTH FINALIZATION — merges the primary 5-run baseline vote
   with refined boundary-expansion data into one clean per-customer
   reference decision. Pure file merging, no new agent calls, no new
   prompt design (see module docstring in aggregate_baseline.py and
   run_boundary_expansion.py for where the actual agent calls already
   happened).

2. CONDITION LOADING AND JOINING — 1b's skeleton key mechanism. Unlike
   1a (one uniform treatment, one cluster_map for the whole experiment),
   1b has 44 conditions each with their own subset of customers and
   fields touched, so the record_id -> customer_id join already lives
   correctly inside each condition's own dither_reference.json (see
   dither_engine.py's DitherEngine.apply() and save_dither_reference()).
   This module's job is the SECOND join every hypothesis needs on top
   of that: customer_id -> finalized ground truth decision. Implemented
   once here, not re-derived in every evaluate_h*.py file.

3. STATISTICAL PRIMITIVES — verified, not just implemented, before any
   hypothesis file depends on them:
   - Wilson interval: imported from refine_boundary_convergence.py
     directly rather than duplicated (same function, same verified math)
   - Prediction-interval t-statistic: for comparing a SINGLE new
     confidence observation against a small reference sample — a
     different formula than a standard one-sample t-test, which tests a
     sample MEAN against a hypothesized value. Verified 2026-08.
   - Exact Mann-Whitney U: for comparing Jaccard dispersion sets at
     small sample sizes, where the normal approximation breaks down.
     Verified with method='exact' against both a null and a clearly
     positive scenario, 2026-08.
   - Fisher's exact test: for H6's per-condition tier ordering check,
     better suited than chi-square for small cell counts.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Reuse the verified Wilson interval implementation directly rather than
# duplicating it — same function, same math, one source of truth.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_boundary_convergence import wilson_interval, interval_width_pp


# ============================================================================
# 1. GROUND TRUTH FINALIZATION
# ============================================================================

def finalize_ground_truth(
    baseline_reference_path: Path,
    refined_classification_path: Optional[Path],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Merge the primary 5-run baseline vote with refined boundary-expansion
    data into one clean per-customer ground truth record.

    For stable / lightly_boundary / deeply_boundary customers: the final
    reference decision is the primary 5-run majority_decision, UNTOUCHED.
    Refined data (if this customer went through boundary expansion) rides
    along as supplementary color — n_runs, plurality_rate, interval_width,
    converged — but never overrides the primary vote. This is the "equal
    computational footing" principle: every customer's ground truth costs
    the same 5 runs, regardless of how much extra effort went into
    refining uncertain cases.

    For tied_no_majority customers: there IS no primary vote to protect —
    Counter.most_common() would otherwise silently pick whichever tied
    decision was inserted first, an artifact of file processing order,
    not a real result. For these customers ONLY, the refined plurality
    (once converged, or the best available estimate at the 60-run cap)
    becomes the final reference decision. This is a documented exception,
    not a violation of the "primary vote never changes" principle — see
    aggregate_baseline.py's module docstring for the full reasoning.

    refined_classification_path may be None if no customers required
    boundary expansion at all (a valid outcome — see
    extract_boundary_subset.py's handling of an empty boundary population).

    Returns the finalized ground truth dict and also writes it to
    output_path. Every customer gets:
        customer_id, stability_tier, final_decision, decision_source
        ("primary_5run" or "refined_boundary_expansion"), plus inline
        refined stats (n_runs, plurality_rate, interval_width_pp,
        converged) for any customer who went through expansion, else None.
    """
    with open(baseline_reference_path) as f:
        baseline = json.load(f)

    refined = {}
    if refined_classification_path is not None and refined_classification_path.exists():
        with open(refined_classification_path) as f:
            refined = json.load(f)

    finalized = {}
    n_primary = 0
    n_refined_exception = 0

    for customer_id, entry in baseline["customers"].items():
        stability = entry["stability"]
        primary_decision = entry["majority_decision"]
        refined_entry = refined.get(customer_id)

        if stability == "tied_no_majority":
            if refined_entry is None:
                raise ValueError(
                    f"Customer {customer_id} is tied_no_majority but has no "
                    f"refined_classification entry — boundary expansion must "
                    f"run for every tied_no_majority customer before ground "
                    f"truth can be finalized. This customer has NO valid "
                    f"reference decision without it."
                )
            final_decision = refined_entry["plurality_decision"]
            decision_source = "refined_boundary_expansion"
            n_refined_exception += 1
        else:
            final_decision = primary_decision
            decision_source = "primary_5run"
            n_primary += 1

        refined_stats = None
        if refined_entry is not None:
            refined_stats = {
                "n_runs":           refined_entry["n_runs"],
                "plurality_rate":   refined_entry["plurality_rate"],
                "interval_width_pp": refined_entry["interval_width_pp"],
                "converged":        refined_entry["converged"],
                "status":           refined_entry["status"],
            }

        finalized[customer_id] = {
            "customer_id":       customer_id,
            "stability_tier":    stability,
            "final_decision":    final_decision,
            "decision_source":   decision_source,
            "primary_decision":  primary_decision,  # kept for transparency,
                                                       # even for the tied
                                                       # exception case, where
                                                       # it will read
                                                       # "TIED_NO_MAJORITY"
            "avg_confidence":    entry.get("avg_confidence"),
            "confidence_range":  entry.get("confidence_range"),
            "refined_stats":     refined_stats,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(finalized, f, indent=2, default=str)

    print(f"  Finalized ground truth for {len(finalized)} customers")
    print(f"    primary_5run:              {n_primary}")
    print(f"    refined_boundary_expansion: {n_refined_exception} (tied_no_majority exception)")
    print(f"  Saved: {output_path}")

    return finalized


# ============================================================================
# 2. CONDITION LOADING AND JOINING — 1b's skeleton key, second join
# ============================================================================

def load_condition(
    condition_dir: Path,
    ground_truth: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Load a single condition's decisions and dither_reference.json, join
    to finalized ground truth by customer_id, and return one record per
    customer with everything a hypothesis file needs.

    This is 1b's skeleton key mechanism, second half. The first join
    (record_id -> customer_id) already lives inside dither_reference.json,
    built by dither_engine.py at generation time. This function performs
    the second join (customer_id -> finalized ground truth decision) and
    hands back one flat, ready-to-use record per customer — so every
    evaluate_h*.py file calls this once rather than re-deriving the join.

    Expects condition_dir to contain:
        agent_input.jsonl       (not read here — the agent already
                                  consumed this to produce decisions)
        dither_reference.json   (customer_id, record_id, _dither_applied,
                                  _dither_fields, _dither_original,
                                  customer_segment, and all other
                                  customer fields)
        decisions.jsonl         (business_decision, agent_confidence,
                                  decision_reasoning, key_factors per
                                  record_id — produced by the agent run)

    Returns a list of dicts, each with:
        customer_id, record_id, condition_id,
        dithered_decision, dithered_confidence, dithered_reasoning,
        dithered_key_factors,
        dither_applied (bool), dither_fields (list), dither_original (dict),
        customer_segment,
        ground_truth_decision, stability_tier, decision_source,
        drifted (bool) — dithered_decision != ground_truth_decision
    """
    dither_ref_path = condition_dir / "dither_reference.json"
    decisions_path = condition_dir / "decisions.jsonl"

    if not dither_ref_path.exists():
        raise FileNotFoundError(f"Missing dither_reference.json in {condition_dir}")
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Missing decisions.jsonl in {condition_dir} — has the agent "
            f"been run against this condition's agent_input.jsonl yet?"
        )

    with open(dither_ref_path) as f:
        dither_ref_records = json.load(f)
    dither_ref_by_record_id = {r["record_id"]: r for r in dither_ref_records}

    decisions_by_record_id = {}
    with open(decisions_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            decisions_by_record_id[d["record_id"]] = d

    condition_id = condition_dir.name
    joined = []
    missing_decisions = []
    missing_ground_truth = []

    for record_id, ref in dither_ref_by_record_id.items():
        customer_id = ref["customer_id"]

        decision = decisions_by_record_id.get(record_id)
        if decision is None:
            missing_decisions.append(record_id)
            continue

        gt = ground_truth.get(customer_id)
        if gt is None:
            missing_ground_truth.append(customer_id)
            continue

        # Capture the post-dither value for each field this customer had
        # dithered, alongside dither_original's pre-dither values. Needed
        # for per-field/per-direction attribution — without both sides,
        # we can know WHICH fields changed but not the direction of a
        # boolean flip or the magnitude of a numeric shift.
        dither_fields = ref.get("_dither_fields", [])
        dither_current_values = {f: ref.get(f) for f in dither_fields}

        joined.append({
            "customer_id":           customer_id,
            "record_id":             record_id,
            "condition_id":          condition_id,
            "dithered_decision":     decision.get("business_decision"),
            "dithered_confidence":   decision.get("agent_confidence"),
            "dithered_reasoning":    decision.get("decision_reasoning"),
            "dithered_key_factors":  decision.get("key_factors", []),
            "dither_applied":        ref.get("_dither_applied", False),
            "dither_fields":         dither_fields,
            "dither_original":       ref.get("_dither_original", {}),
            "dither_current_values": dither_current_values,
            "dither_operator":       ref.get("_dither_operator", {}),
            "dither_direction":      ref.get("_dither_direction", {}),
            "dither_mechanism_style": ref.get("_dither_mechanism_style", {}),
            "customer_segment":      ref.get("customer_segment"),
            "ground_truth_decision": gt["final_decision"],
            "stability_tier":        gt["stability_tier"],
            "decision_source":       gt["decision_source"],
            "drifted":               decision.get("business_decision") != gt["final_decision"],
        })

    if missing_decisions:
        raise ValueError(
            f"{condition_id}: {len(missing_decisions)} record(s) in "
            f"dither_reference.json have no matching decision — agent run "
            f"may be incomplete. First few: {missing_decisions[:5]}"
        )
    if missing_ground_truth:
        raise ValueError(
            f"{condition_id}: {len(missing_ground_truth)} customer(s) have "
            f"no entry in finalized ground truth. First few: "
            f"{missing_ground_truth[:5]}. Was finalize_ground_truth() run "
            f"against the same baseline population as this condition?"
        )

    return joined


def compute_drift_rate(condition_records: List[Dict[str, Any]]) -> float:
    """
    Binary drift rate for a condition: fraction of customers whose
    dithered decision differs from their finalized ground truth decision.
    The single most-reused number in the entire evaluator — every
    hypothesis file's headline metric traces back to this.
    """
    if not condition_records:
        raise ValueError("No records to compute drift rate from")
    n_drifted = sum(1 for r in condition_records if r["drifted"])
    return n_drifted / len(condition_records)


def compute_effective_drift_rate(
    condition_records: List[Dict[str, Any]],
    field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    EXPOSURE-ADJUSTED drift rate: drift among only the customers whose
    data was actually perturbed — every customer listing `field` in
    dither_fields if a field is given, otherwise every customer with at
    least one changed field. (dither_fields lists only fields that
    ACTUALLY changed, so no new metadata is needed.)

    Why this exists: an exposure audit (2026-08) found fields differ
    enormously in how many customers a condition actually touches —
    booleans by design (~15% flip at 15% magnitude), and small-integer
    counts by an engine limitation since fixed. Raw drift rate averages
    over untouched customers, so comparing it ACROSS fields partly
    measures how many customers were touched, not how much the agent
    cares. Any cross-field comparison (H1's Question A in particular)
    must use this.

    Conditioning on "was perturbed" is valid, unlike the rejected
    confidence-given-drift analysis: perturbation is set by the dither's
    random draw (treatment assignment), never by the agent's outcome, so
    no post-treatment selection bias is introduced.
    """
    if not condition_records:
        raise ValueError("No records to compute drift rate from")
    if field is None:
        perturbed = [r for r in condition_records if r["dither_fields"]]
    else:
        perturbed = [r for r in condition_records if field in r["dither_fields"]]
    n = len(perturbed)
    return {
        "effective_drift_rate": (sum(1 for r in perturbed if r["drifted"]) / n) if n else None,
        "n_perturbed":          n,
        "exposure":             n / len(condition_records),
        "raw_drift_rate":       compute_drift_rate(condition_records),
    }


# ============================================================================
# 3. PER-FIELD / PER-DIRECTION ATTRIBUTION
# ============================================================================
#
# CORRELATIONAL, NOT CAUSAL. When a multi-field condition dithers several
# fields simultaneously for the same customer (e.g. h1_category_account_status,
# any H7 breadth step, h8a's category pairs), observing that "customers
# where field X changed drifted more often" does not isolate X's individual
# causal contribution — other fields moved for those same customers too.
# This is a stated limitation of every multi-field condition, not a flaw
# specific to any one analysis that uses it.

def attribute_by_field(condition_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Break out drift rate by which specific field(s) actually changed for
    each customer, using dither_fields (which is per-customer — in an
    uncorrelated multi-field condition, not every targeted field
    necessarily changes for every customer, since each field independently
    rolls its own perturbation).

    For boolean fields specifically, also break out by DIRECTION of
    change (False->True vs True->False), using dither_original vs
    dither_current_values. This is what let us disentangle the is_vip
    finding — a rare boolean's flip-probability dithering produces a
    dramatically asymmetric False->True vs True->False split purely from
    base rate, not from the mechanism doing anything wrong.

    Returns:
        {
          "by_field": {field_name: {"n": int, "drift_rate": float}, ...},
          "by_field_direction": {
              "field_name": {
                  "False_to_True": {"n": int, "drift_rate": float},
                  "True_to_False": {"n": int, "drift_rate": float},
              }, ...
          }  # only populated for fields where dither_original values are
             # actual booleans
        }
    """
    by_field: Dict[str, List[bool]] = {}
    by_field_direction: Dict[str, Dict[str, List[bool]]] = {}

    for r in condition_records:
        for field in r["dither_fields"]:
            by_field.setdefault(field, []).append(r["drifted"])

            original = r["dither_original"].get(field)
            current = r["dither_current_values"].get(field)

            if isinstance(original, bool) and isinstance(current, bool) and original != current:
                direction = f"{original}_to_{current}"
                by_field_direction.setdefault(field, {}).setdefault(
                    direction, []).append(r["drifted"])

    by_field_summary = {
        field: {"n": len(flags), "drift_rate": sum(flags) / len(flags)}
        for field, flags in by_field.items()
    }

    by_field_direction_summary = {}
    for field, directions in by_field_direction.items():
        by_field_direction_summary[field] = {
            direction: {"n": len(flags), "drift_rate": sum(flags) / len(flags)}
            for direction, flags in directions.items()
        }

    return {
        "by_field": by_field_summary,
        "by_field_direction": by_field_direction_summary,
    }


# ============================================================================
# 4. SEGMENT / PROFILE MISMATCH
# ============================================================================
#
# Computed EMPIRICALLY from canonical_customers.json (the clean ground
# truth population), never from the generator's internal literal bounds —
# this avoids a second source of truth that could drift out of sync with
# the generator. Scoped to ONLY the field(s) a given condition actually
# dithered — checking every field regardless of what was touched would
# mean most "mismatches" are customers whose UNTOUCHED fields happened to
# sit near a percentile edge for unrelated reasons, pure noise riding
# alongside the signal this lens exists to detect.

def compute_segment_field_ranges(
    canonical_customers: List[Dict[str, Any]],
    fields: List[str],
    low_pct: float = 5,
    high_pct: float = 95,
) -> Dict[str, Dict[str, Any]]:
    """
    For each field and each segment, compute the empirical reference
    range from the clean ground truth population.

    Numeric fields: the [low_pct, high_pct] percentile range.
    Boolean fields: the empirical P(True | segment) rate — used
    downstream to flag a dithered boolean value as a mismatch if it
    represents a state that rarely occurs naturally for that segment
    (e.g. is_vip=True for a non-high_value customer, since is_vip is
    gated to high_value in the generator and only ~25% of high_value
    customers get it besides).

    Returns: {field: {segment: {"type": "numeric", "low": x, "high": y}
                                 or {"type": "boolean", "p_true": z}}}
    """
    by_segment: Dict[str, List[Dict[str, Any]]] = {}
    for c in canonical_customers:
        by_segment.setdefault(c["customer_segment"], []).append(c)

    ranges: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        ranges[field] = {}
        for segment, customers in by_segment.items():
            values = [c[field] for c in customers if field in c]
            if not values:
                continue

            if isinstance(values[0], bool):
                p_true = sum(values) / len(values)
                ranges[field][segment] = {"type": "boolean", "p_true": p_true}
            else:
                arr = np.array(values, dtype=float)
                ranges[field][segment] = {
                    "type": "numeric",
                    "low":  float(np.percentile(arr, low_pct)),
                    "high": float(np.percentile(arr, high_pct)),
                }

    return ranges


def compute_segment_mismatch(
    condition_records: List[Dict[str, Any]],
    segment_ranges: Dict[str, Dict[str, Any]],
    rare_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    For each customer, check whether the DITHERED value of each field
    this condition touched still falls within their ORIGINAL segment's
    typical range (numeric) or typical rate (boolean).

    A customer whose dithered profile has "left" their assigned segment's
    normal territory is flagged. This is a free enrichment cross-cutting
    H1, H2, H4, H7 — anywhere a numeric or boolean field gets dithered —
    computed here once rather than re-derived per hypothesis file.

    rare_threshold: for boolean fields, a dithered value is flagged as a
    mismatch if its segment-conditional empirical rate is below this
    threshold (default 5%) — i.e. this segment essentially never shows
    this boolean state naturally.
    """
    mismatched_customers = []
    n_checked = 0

    for r in condition_records:
        segment = r["customer_segment"]
        for field in r["dither_fields"]:
            if field not in segment_ranges or segment not in segment_ranges[field]:
                continue
            n_checked += 1

            current_value = r["dither_current_values"].get(field)
            if current_value is None:
                continue

            field_range = segment_ranges[field][segment]
            is_mismatch = False

            if field_range["type"] == "numeric":
                if current_value < field_range["low"] or current_value > field_range["high"]:
                    is_mismatch = True
            elif field_range["type"] == "boolean":
                p_true = field_range["p_true"]
                if current_value is True and p_true < rare_threshold:
                    is_mismatch = True
                elif current_value is False and (1 - p_true) < rare_threshold:
                    is_mismatch = True

            if is_mismatch:
                mismatched_customers.append({
                    "customer_id": r["customer_id"],
                    "field": field,
                    "segment": segment,
                    "dithered_value": current_value,
                    "reference_range": field_range,
                })

    return {
        "n_field_checks":       n_checked,
        "n_mismatches":         len(mismatched_customers),
        "mismatch_rate":        len(mismatched_customers) / n_checked if n_checked else 0.0,
        "mismatched_customers": mismatched_customers,
    }


# ============================================================================
# 5. STATISTICAL PRIMITIVES
# ============================================================================

# Minimal stop-word list — deliberately small and auditable rather than
# pulling in a full NLP library's list, consistent with keeping this
# metric simple and deterministic (see 1a's original Jaccard rationale:
# "deterministic and reproducible; LLM-based similarity would introduce
# AI variability that contaminates the measurement").
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "from",
    "it", "its", "their", "they", "has", "have", "had", "will", "would",
}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Word-overlap Jaccard similarity, stop-word filtered, deterministic.
    Unweighted set overlap — deliberately, not a frequency-weighted
    variant. Weighting toward frequency would amplify boilerplate
    phrasing rather than substantive differences; stop-word removal
    already solves the "common words drowning out signal" problem more
    simply, by removing the noise rather than trying to mathematically
    down-weight it. Consistent with 1a's original Jaccard rationale and
    1b's H3/H5 design.
    """
    def tokenize(text: str) -> set:
        words = text.lower().replace(",", " ").replace(".", " ").split()
        return {w for w in words if w not in STOPWORDS and w.isalpha()}

    set_a, set_b = tokenize(text_a), tokenize(text_b)
    if not set_a and not set_b:
        return 1.0  # both empty — trivially identical
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def mean_pairwise_jaccard(target_text: str, reference_texts: List[str]) -> Optional[float]:
    """
    Mean pairwise Jaccard between a target text and a list of reference
    texts — NOT pooled (reference texts merged into one bag of words
    first). Pooling would let the reference vocabulary's SIZE grow with
    however many reference texts happen to be available, meaning the
    metric would partly measure "how many runs happened to agree" rather
    than "how similar is the reasoning" — a raw-count confound of exactly
    the kind hunted down elsewhere in this series (1a's volume inflation
    finding). Mean pairwise avoids it: each comparison is apples-to-apples
    regardless of how many reference points exist.

    Returns None if reference_texts is empty (e.g. a customer with zero
    baseline runs matching their final ground truth decision — should be
    rare but not impossible for tied_no_majority customers with unusual
    convergence patterns).
    """
    if not reference_texts:
        return None
    scores = [jaccard_similarity(target_text, ref) for ref in reference_texts]
    return sum(scores) / len(scores)


def prediction_interval_t_test(
    new_observation: float,
    baseline_values: List[float],
) -> Dict[str, Any]:
    """
    Is a SINGLE new observation unusual relative to a small reference
    sample? This is NOT a standard one-sample t-test (which tests
    whether a SAMPLE MEAN differs from a hypothesized value — the wrong
    question for our case, since we have one new point, not a competing
    sample). Uses the prediction-interval-style t-statistic instead:

        t = (new_obs - mean) / (s * sqrt(1 + 1/n))

    The extra "+1" inside the sqrt (vs. a standard t-test's s/sqrt(n))
    accounts for BOTH the uncertainty in estimating the true mean from a
    small sample AND the inherent variability of a new individual
    observation around that mean. Verified against a standard one-sample
    t-test formula on the same data 2026-08 — the standard formula
    produces t=15.9 (absurdly inflated, would flag nearly any deviation
    as significant) vs. this formula's properly-calibrated t=-6.5 on
    identical input, confirming the distinction matters in practice, not
    just in theory.

    Used for comparing a dithered condition's confidence against a
    customer's baseline confidence distribution (H3, H6).

    df = n - 1, where n is however many baseline runs this specific
    customer actually has (5 for most customers; more for anyone who
    went through boundary expansion — their expansion-run confidence
    values are real additional data, not something to discard).
    """
    n = len(baseline_values)
    if n < 2:
        return {"t_statistic": None, "p_value": None, "df": None,
                "note": "Need at least 2 baseline values to estimate variance"}

    arr = np.array(baseline_values)
    mean = arr.mean()
    s = arr.std(ddof=1)
    df = n - 1

    if s == 0:
        # Baseline had zero variance (e.g. identical confidence every run)
        # — any deviation at all is meaningful, but a t-statistic is
        # undefined with zero variance in the denominator.
        return {
            "t_statistic": None, "p_value": None, "df": df,
            "note": "Baseline confidence had zero variance — any deviation "
                    "in the new observation is notable but not expressible "
                    "as a t-statistic",
            "baseline_mean": float(mean),
            "new_observation": new_observation,
            "matches_baseline_exactly": bool(new_observation == mean),
        }

    t_stat = (new_observation - mean) / (s * math.sqrt(1 + 1/n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": df,
        "baseline_mean": float(mean),
        "baseline_std": float(s),
        "new_observation": new_observation,
    }


def mann_whitney_test(
    group_a: List[float],
    group_b: List[float],
    label_a: str = "group_a",
    label_b: str = "group_b",
) -> Dict[str, Any]:
    """
    Generic exact Mann-Whitney U test for two independent samples — NOT
    the normal approximation, which breaks down at small sample sizes
    (verified against both a null and a clearly-separated scenario,
    2026-08, originally in the context of Jaccard dispersion — see
    jaccard_dispersion_test() below, now a thin wrapper around this).

    Valid ONLY when the two groups are genuinely independent samples.
    Do NOT use this to compare data drawn from the SAME underlying
    population under two conditions (e.g. the same customers' scores
    under two different dithering conditions) — that's a paired
    comparison and needs wilcoxon_signed_rank_test() instead. See "A
    Note on Statistical Methodology" for the full reasoning on when
    each applies.

    Used for: H1's Question A group-level check (5 self-reported top-5
    field drift rates vs. 6 comparison field drift rates — genuinely
    independent since each is a summary statistic from a DIFFERENT
    field's condition, not the same customers measured twice), and
    jaccard_dispersion_test()'s per-customer diagnostic.
    """
    if len(group_a) < 1 or len(group_b) < 1:
        return {"u_statistic": None, "p_value": None,
                "note": "Insufficient data for Mann-Whitney test"}

    u_stat, p_value = stats.mannwhitneyu(
        group_a, group_b, method='exact', alternative='two-sided',
    )

    return {
        "u_statistic":            float(u_stat),
        "p_value":                float(p_value),
        f"n_{label_a}":            len(group_a),
        f"n_{label_b}":            len(group_b),
        f"{label_a}_mean":         float(np.mean(group_a)),
        f"{label_b}_mean":         float(np.mean(group_b)),
    }


def jaccard_dispersion_test(
    dithered_vs_baseline_scores: List[float],
    baseline_self_similarity_scores: List[float],
) -> Dict[str, Any]:
    """
    PER-CUSTOMER diagnostic only — see jaccard_condition_level_shift()
    for the correct CONDITION-level (whole-population) headline metric.

    Does a dithered condition's reasoning look like ordinary baseline
    wobble, or is it genuinely less coherent than the customer's own
    reasoning ever is with itself? Two-sample comparison via EXACT
    Mann-Whitney U — not the normal approximation, which breaks down at
    our sample sizes (as few as 5 dithered scores vs. 10 baseline-pair
    scores for a stable customer). Verified 2026-08: correctly reads a
    "looks like normal wobble" scenario as non-significant (p=0.44) and
    a genuinely degraded scenario as sharply significant (p=0.0007,
    U=0 — every dithered score below every baseline score).

    Thin wrapper around mann_whitney_test() — kept as its own named
    function (rather than calling mann_whitney_test() directly at every
    call site) specifically for its scope warning below, and to keep
    the historical verified-numbers reference attached to a stable name.

    SCOPE WARNING: valid only WITHIN a single customer's own two small
    score sets. Do NOT pool scores across multiple customers and feed
    them here — the same baseline texts feed both a customer's dithered-
    vs-baseline scores AND their own self-similarity scores, so pooling
    across the population would compare correlated data as if it were
    independent, exactly the mistake McNemar's test was built to avoid
    for drift rates. For a condition-level (whole-population) finding,
    use jaccard_condition_level_shift() instead, which correctly reduces
    each customer to one paired difference before testing.

    dithered_vs_baseline_scores: Jaccard(dithered_reasoning, each
        matching baseline reasoning text) — one score per baseline text,
        for ONE customer
    baseline_self_similarity_scores: Jaccard between every PAIR of THAT
        SAME customer's baseline reasoning texts (C(n,2) scores) — their
        own natural reasoning variability, with no dithering involved
    """
    result = mann_whitney_test(
        dithered_vs_baseline_scores, baseline_self_similarity_scores,
        label_a="dithered", label_b="baseline_self",
    )
    if result["u_statistic"] is None:
        return {"u_statistic": None, "p_value": None,
                "note": "Insufficient data for dispersion test"}
    return result


def fishers_exact_tier_check(
    drift_count_tier_a: int,
    total_tier_a: int,
    drift_count_tier_b: int,
    total_tier_b: int,
) -> Dict[str, Any]:
    """
    Fisher's exact test on a 2x2 contingency table comparing drift rate
    between two stability tiers within a single condition — better
    suited than chi-square for small cell counts (H6's boundary tiers
    are often thin, especially tied_no_majority). Per-condition check
    only — H6's actual headline claim rests on the MEDIAN ratio holding
    consistently across all 44+ conditions, not on any single condition's
    p-value in isolation (declaring one cell "significant" risks the
    exact multiple-comparisons inflation this test alone can't fix).
    """
    table = [
        [drift_count_tier_a, total_tier_a - drift_count_tier_a],
        [drift_count_tier_b, total_tier_b - drift_count_tier_b],
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "tier_a_rate": drift_count_tier_a / total_tier_a if total_tier_a else None,
        "tier_b_rate": drift_count_tier_b / total_tier_b if total_tier_b else None,
    }


def mcnemar_paired_test(
    drifted_under_x: List[bool],
    drifted_under_y: List[bool],
) -> Dict[str, Any]:
    """
    Are two fields' drift rates genuinely different, accounting for the
    fact that every condition in 1b dithers the SAME underlying 1,000-
    customer population? Comparing field X's drift rate to field Y's
    drift rate via a standard two-proportion test would treat them as
    independent samples — but they're not. Customer C's drift-under-X
    and drift-under-Y both depend on the same customer's baseline
    profile (a customer already near a decision boundary is more likely
    to drift under EITHER field's dithering than a rock-solid stable
    customer is). That shared dependency makes this a PAIRED comparison,
    the same structural category as a before/after medical trial — just
    with "before" and "after" replaced by "under field X" and "under
    field Y" for the same customer, with no risk of order effects since
    the agent has no memory between calls and both conditions are
    independently generated from the same clean baseline.

    McNemar's test is the correct tool for paired binary outcomes. It
    only draws information from DISCORDANT pairs — customers who
    drifted under exactly one of the two fields, not both or neither.
    This has a nice side effect: customers who are simply prone to
    drifting regardless of which field gets touched (general fragility)
    are naturally excluded from the field-specific comparison, echoing
    the same general-fragility-vs-field-specific-sensitivity distinction
    H6 was built to draw, just surfacing for free at the pairwise level.

    Uses the EXACT binomial formulation (not the chi-square
    approximation), consistent with using exact methods rather than
    normal approximations wherever sample sizes might be small — same
    principle as choosing Wilson over Wald and exact Mann-Whitney over
    its normal approximation elsewhere in this module.

    drifted_under_x, drifted_under_y: aligned lists (same customer, same
        order) of whether each customer drifted under condition X and
        condition Y respectively. Must be pre-filtered to customers
        present in BOTH conditions before calling this.
    """
    if len(drifted_under_x) != len(drifted_under_y):
        raise ValueError(
            f"Mismatched lengths ({len(drifted_under_x)} vs "
            f"{len(drifted_under_y)}) — inputs must be aligned per customer."
        )

    b = sum(1 for x, y in zip(drifted_under_x, drifted_under_y) if x and not y)
    c = sum(1 for x, y in zip(drifted_under_x, drifted_under_y) if not x and y)
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "b_x_only": b, "c_y_only": c, "n_discordant": 0,
            "p_value": 1.0,
            "note": "No discordant pairs — fields agree on every customer "
                    "in this sample, nothing to distinguish them on.",
        }

    result = stats.binomtest(b, n_discordant, p=0.5, alternative='two-sided')

    return {
        "b_x_only":     b,   # drifted under X but not Y
        "c_y_only":     c,   # drifted under Y but not X
        "n_discordant": n_discordant,
        "n_pairs":      len(drifted_under_x),
        "p_value":      float(result.pvalue),
    }


def wilcoxon_signed_rank_test(paired_differences: List[float]) -> Dict[str, Any]:
    """
    Does a condition-level continuous metric show a genuine shift across
    the SAME customer population, when raw values can't be pooled into
    two independent groups?

    This matters everywhere in 1b, not just one place: every condition
    dithers the same 1,000-customer baseline population. Comparing two
    conditions' raw per-customer scores as if they were independent
    samples (the naive Mann-Whitney approach) breaks down whenever the
    same customer contributes correlated information to both sides — the
    same underlying baseline texts feed BOTH a customer's dithered-vs-
    baseline Jaccard scores AND their own baseline-self-similarity
    scores, so a verbose customer's writing style shows up in both
    measurements, not as two independent observations.

    The fix: reduce each customer to ONE paired difference (their
    condition-A summary value minus their condition-B summary value, or
    dithered-coherence minus baseline-coherence for the Jaccard case),
    then test whether the MEDIAN of those paired differences departs
    from zero across the population. This is the direct continuous-
    variable analog to McNemar's test for paired binary outcomes —
    same underlying principle (respect the pairing, don't pretend
    independence), different data type.

    Used for: condition-level Jaccard coherence shift (H3, H5), and any
    future continuous paired comparison across magnitude levels or dither
    types for the same field (H2, H4) where the same customers are being
    compared under two treatments rather than two independent samples.

    Exact zero differences are dropped before ranking (scipy's default
    behavior, `zero_method='wilcox'`) — a customer whose scores were
    identical under both conditions contributes no directional
    information either way.
    """
    if len(paired_differences) < 1:
        return {"statistic": None, "p_value": None,
                "note": "No paired differences to test"}

    nonzero = [d for d in paired_differences if d != 0]
    if len(nonzero) < 1:
        return {"statistic": None, "p_value": 1.0, "n_pairs": len(paired_differences),
                "n_nonzero": 0,
                "note": "Every paired difference was exactly zero — no "
                        "directional signal either way"}

    result = stats.wilcoxon(nonzero, alternative='two-sided')

    return {
        "statistic":     float(result.statistic),
        "p_value":       float(result.pvalue),
        "n_pairs":       len(paired_differences),
        "n_nonzero":     len(nonzero),
        "median_diff":   float(np.median(paired_differences)),
    }


def jaccard_condition_level_shift(
    per_customer_dithered_coherence: List[float],
    per_customer_baseline_coherence: List[float],
) -> Dict[str, Any]:
    """
    Condition-level headline metric: does this dithering condition, in
    general, degrade reasoning coherence across the customer population?

    per_customer_dithered_coherence: one value per customer — their mean
        Jaccard(dithered_reasoning, matching baseline texts) — see
        mean_pairwise_jaccard().
    per_customer_baseline_coherence: one value per customer, SAME ORDER —
        their mean baseline self-similarity (mean pairwise Jaccard among
        their own matching baseline texts).

    Wraps wilcoxon_signed_rank_test() on the per-customer paired
    differences (dithered - baseline) rather than pooling raw scores —
    see that function's docstring for why pooling would violate
    independence. This REPLACES a naive condition-level Mann-Whitney,
    which would have incorrectly treated correlated per-customer scores
    as independent samples.
    """
    if len(per_customer_dithered_coherence) != len(per_customer_baseline_coherence):
        raise ValueError(
            f"Mismatched lengths ({len(per_customer_dithered_coherence)} vs "
            f"{len(per_customer_baseline_coherence)}) — inputs must be "
            f"aligned per customer, same order."
        )

    diffs = [
        d - b for d, b in zip(per_customer_dithered_coherence, per_customer_baseline_coherence)
    ]
    result = wilcoxon_signed_rank_test(diffs)
    result["interpretation"] = (
        "negative median_diff means dithered reasoning is LESS coherent "
        "than the customer's own baseline wobble, on average"
    )
    return result


def spearman_monotonicity_test(
    ordered_x: List[float],
    ordered_y: List[float],
) -> Dict[str, Any]:
    """
    Does y trend monotonically with x? Used for H2's magnitude-vs-drift-
    rate curves: does drift rate increase monotonically as dither
    magnitude increases, within a single field?

    This is a field-level test on already-aggregated summary statistics
    (one drift rate per magnitude level, e.g. 4 points for the 5/15/40/
    100% ladder), NOT a customer-level test — so it does not have the
    same-population pairing problem McNemar's and Wilcoxon exist to fix
    elsewhere in this module. Each point is one field's condition at one
    magnitude, not the same customers measured twice at two magnitudes.
    (Comparing two SPECIFIC magnitude levels' drift rates directly for
    the same field, e.g. "is 40% significantly higher than 15%?", DOES
    have that pairing problem and needs mcnemar_paired_test() instead —
    this function only tests the overall trend shape across all levels.)

    ordered_x: the ordinal/numeric sequence (e.g. magnitude levels in
        order: [0.05, 0.15, 0.40, 1.00])
    ordered_y: the corresponding values at each x (e.g. drift rates)

    Returns rho close to +1 for a clean monotonic increase, close to -1
    for a clean monotonic decrease, near 0 for no consistent trend.
    """
    if len(ordered_x) != len(ordered_y):
        raise ValueError(
            f"Mismatched lengths ({len(ordered_x)} vs {len(ordered_y)})"
        )
    if len(ordered_x) < 3:
        return {"rho": None, "p_value": None,
                "note": "Need at least 3 points to assess a trend"}

    rho, p_value = stats.spearmanr(ordered_x, ordered_y)

    return {
        "rho":     float(rho),
        "p_value": float(p_value),
        "n_points": len(ordered_x),
        "interpretation": (
            "monotonic increase" if rho > 0.5 else
            "monotonic decrease" if rho < -0.5 else
            "no clear monotonic trend"
        ),
    }


def cochrans_q_test(binary_matrix: List[List[bool]]) -> Dict[str, Any]:
    """
    Generalizes McNemar's test to MORE THAN TWO paired conditions applied
    to the same population — needed anywhere a ladder of k>2 conditions
    (not just two) is compared for the same customers. First identified
    as needed for H7's breadth ladder (1 field, 3, 6, all — 4 paired
    conditions); actually needed FIRST for H2's magnitude ladder (5%,
    15%, 40%, 100% — also 4 paired conditions per field), which is why
    it's built here rather than deferred. H7 reuses this same primitive
    when built.

    Uses the standard chi-square approximation, NOT an exact permutation
    test — deliberately, and for a different reason than every other
    "prefer exact" choice in this module. Wilson-over-Wald, exact
    Mann-Whitney, and the exact binomial inside McNemar's were all
    correcting for a SMALL number of RUNS OR OBSERVATIONS (5 baseline
    runs, 5-vs-10 Jaccard scores). Cochran's Q's asymptotic validity
    depends on the number of SUBJECTS (~1,000 customers here), which is
    large — the chi-square approximation is well-behaved at that scale,
    and no readily-available exact permutation version exists without
    adding a new dependency (statsmodels) for a test whose approximation
    is already sound for our actual sample size.

    Verified against a proven mathematical identity rather than a
    recalled reference number: when k=2, Cochran's Q must algebraically
    reduce to McNemar's uncorrected chi-square statistic, (b-c)^2/(b+c).
    Confirmed both by hand derivation and numerically, 2026-08.

    binary_matrix: one row per customer, one column per condition (k
    columns, all conditions applied to the SAME customers, same order).
    Each cell is True/False (drifted or not) under that condition.

    Usage pattern for a magnitude/breadth ladder: if this comes back
    significant, follow up with adjacent-step McNemar's tests to
    localize WHERE in the ladder the difference occurs. If not
    significant, report the null finding directly — do not go fishing
    in post-hoc adjacent tests the omnibus gate didn't earn (same
    discipline as H8b's conditional generation rule).
    """
    arr = np.array(binary_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("binary_matrix must be 2D: rows=customers, columns=conditions")
    n, k = arr.shape
    if k < 2:
        raise ValueError(f"Need at least 2 conditions, got {k}")

    T = arr.sum(axis=0)  # column totals — total drifted count per condition
    L = arr.sum(axis=1)  # row totals — total conditions each customer drifted under
    N = arr.sum()

    denominator = k * np.sum(L) - np.sum(L**2)
    if denominator == 0:
        return {
            "Q_statistic": None, "df": k - 1, "p_value": None,
            "note": "Degenerate case — every customer drifted under all "
                    "conditions or none, no variation to test.",
        }

    numerator = (k - 1) * (k * np.sum(T**2) - N**2)
    Q = numerator / denominator
    df = k - 1
    p_value = float(stats.chi2.sf(Q, df))

    return {
        "Q_statistic":    float(Q),
        "df":             df,
        "p_value":        p_value,
        "n_subjects":     n,
        "k_conditions":   k,
        "column_totals":  T.tolist(),
    }


def align_drift_by_customer(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    perturbed_only: bool = False,
) -> Tuple[List[bool], List[bool], int]:
    """
    Build aligned (same customer, same order) drift-boolean lists for
    two conditions, intersecting on customer_id — the standard prep step
    before any paired test (McNemar's, Wilcoxon, Cochran's Q) between
    two or more conditions applied to the same population.

    Originally built inside evaluate_h1.py for its 30 pairwise McNemar's
    comparisons; moved here once evaluate_h2.py needed the identical
    utility for its magnitude-ladder comparisons — same "one correct
    implementation, reused everywhere" principle already applied to
    mann_whitney_test().

    Conditions should share the same 1,000-customer population in
    practice (no H1/H2 condition uses segment_filter), but intersecting
    defensively rather than assuming identical customer sets protects
    against a silent misalignment if that ever changes.
    """
    # perturbed_only=True: restrict to customers actually perturbed in BOTH
    # conditions — the exposure-adjusted paired comparison. Valid for the
    # same reason compute_effective_drift_rate() is: perturbation is set by
    # each condition's own random draw, never by agent outcomes.
    if perturbed_only:
        records_a = [r for r in records_a if r["dither_fields"]]
        records_b = [r for r in records_b if r["dither_fields"]]
    by_customer_a = {r["customer_id"]: r["drifted"] for r in records_a}
    by_customer_b = {r["customer_id"]: r["drifted"] for r in records_b}
    shared_customers = sorted(set(by_customer_a) & set(by_customer_b))

    drift_a = [by_customer_a[c] for c in shared_customers]
    drift_b = [by_customer_b[c] for c in shared_customers]
    return drift_a, drift_b, len(shared_customers)


def align_drift_by_customer_multi(
    condition_records_list: List[List[Dict[str, Any]]],
) -> Tuple[List[List[bool]], List[str]]:
    """
    Generalizes align_drift_by_customer() to MORE THAN TWO conditions —
    needed for Cochran's Q, which requires one binary_matrix row per
    customer across all k conditions at once, not pairwise. Intersects
    on customer_id across ALL conditions in the list (defensive, same
    reasoning as the two-condition version).

    Returns (binary_matrix, shared_customer_ids) where binary_matrix has
    one row per shared customer, one column per condition IN THE SAME
    ORDER as condition_records_list — the caller is responsible for
    passing conditions in a meaningful order (e.g. ascending magnitude)
    since Cochran's Q itself is order-agnostic, but any downstream
    adjacent-step follow-up depends on the columns being correctly ordered.
    """
    per_condition_maps = [
        {r["customer_id"]: r["drifted"] for r in records}
        for records in condition_records_list
    ]
    shared_customers = sorted(set.intersection(*(set(m) for m in per_condition_maps)))

    binary_matrix = [
        [m[c] for m in per_condition_maps]
        for c in shared_customers
    ]
    return binary_matrix, shared_customers


def descriptive_trend_correlation(
    x_values: List[float],
    y_values: List[float],
) -> Dict[str, Any]:
    """
    Spearman's rank correlation for a small ordered ladder (e.g. drift
    rate across 4 magnitude levels) — DESCRIPTIVE, not confirmatory. At
    n=4 points, this is not a significance test we lean statistical
    weight against; it's a description of whether the observed curve
    trends consistently in one direction, worth reporting as exploratory
    signal ("this pattern is worth further investigation at larger
    sample size") rather than a claim of established monotonicity.

    The caveat is attached directly to the returned data, not left to
    documentation alone — same pattern as h1_distributed's scope_note —
    so nobody downstream can report the correlation without the caveat
    traveling with it.
    """
    if len(x_values) < 3:
        return {"rho": None, "p_value": None,
                "note": "Need at least 3 points for a meaningful trend correlation"}

    rho, p_value = stats.spearmanr(x_values, y_values)
    return {
        "rho":     float(rho),
        "p_value": float(p_value),
        "n_points": len(x_values),
        "interpretation": (
            "DESCRIPTIVE ONLY, not confirmatory. At this few points, rho "
            "describes whether the curve trends consistently in one "
            "direction in THIS sample — it does not establish "
            "monotonicity or carry statistical weight on its own. A "
            "rho close to +-1 is exploratory signal worth further "
            "investigation at larger sample size, not a validated finding."
        ),
    }


def binary_did_gee(
    rows: List[Dict[str, Any]],
    treatment_group: str,
    reference_group: str,
    group_col: str = "pair_type",
    arm_col: str = "is_uncorrelated",
    outcome_col: str = "drift",
    cluster_col: str = "customer_id",
    cov_struct: str = "independence",
) -> Dict[str, Any]:
    """
    Binary difference-in-differences via a Generalized Estimating
    Equation (logit link, clustered by customer). Tests whether the
    effect of an arm (e.g. uncorrelated vs. correlated dithering) is
    LARGER for a treatment group (e.g. a genuinely correlated field
    pair) than for a reference group (e.g. a pair with no real
    relationship to break).

    WHY THIS EXISTS INSTEAD OF A WILCOXON ON PER-CUSTOMER DIFFERENCES:
    the per-customer construction D_i - R_i (each a difference of two
    binary drift outcomes) produces values in {-2,-1,0,+1,+2}. Wilcoxon
    discards zero differences, and under realistic drift rates ~60% of
    customers land at exactly zero (verified by simulation, 2026-08) —
    gutting power — and the remaining mass sits on a handful of
    discrete values, far from the continuous distribution Wilcoxon is
    built around. GEE models the binary outcome directly, uses every
    observation, and handles the within-customer clustering (each
    customer contributes 4 correlated observations) via robust sandwich
    standard errors.

    WHY THIS IS DIFFERENT FROM THE REJECTED MIXED MODEL FOR CONFIDENCE:
    that model used Drift as a covariate to explain Confidence — Drift
    is itself an outcome caused by treatment (a mediator), so
    conditioning on it reintroduces post-treatment selection bias.
    Here, group and arm are both pure experimental design variables,
    and drift is modeled as the outcome. No post-treatment conditioning,
    so the interaction term cleanly estimates the DiD quantity.

    The key estimate is the INTERACTION coefficient (group x arm), on
    the log-odds scale: the extra change in log-odds of drift caused by
    the arm, specifically in the treatment group, beyond what the arm
    does to the reference group. Reported as an odds ratio (exp(beta)).
    OR > 1 means decorrelating the treatment group costs MORE than
    decorrelating the reference group.

    Working correlation defaults to Independence, not Exchangeable:
    GEE point estimates stay consistent and sandwich SEs stay valid even
    under a misspecified working correlation, and Exchangeable's "every
    pair of a customer's observations correlates equally" assumption
    isn't obviously more correct here than Independence. Either is
    selectable; Independence is the simpler, conservative default.

    New dependency: statsmodels (added to requirements.txt deliberately —
    a reversal of the position taken for Cochran's Q, where the simpler
    approximation was sufficient. Here there is no simpler tool that
    answers the question correctly).

    Verified 2026-08 against synthetic data: a planted null (arm has
    identical effect in both groups) stays non-significant; a planted
    DiD effect (arm raises drift only in the treatment group) is
    detected with the correct sign and an odds ratio near the planted
    value.

    rows: flat list, one dict per (customer, group, arm) observation.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    df = df[df[group_col].isin([treatment_group, reference_group])].copy()

    required = {group_col, arm_col, outcome_col, cluster_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[outcome_col] = df[outcome_col].astype(int)
    df[arm_col] = df[arm_col].astype(int)

    # Degenerate-case guards: GEE on a constant outcome is undefined
    if df[outcome_col].nunique() < 2:
        return {"odds_ratio": None, "p_value": None,
                "note": "Outcome has no variation (all drift or none) — "
                        "DiD not estimable."}

    cov = {
        "independence": sm.cov_struct.Independence(),
        "exchangeable": sm.cov_struct.Exchangeable(),
    }[cov_struct]

    # Explicit reference level, so the interaction sign always means
    # "treatment group's extra arm effect," never alphabetical accident.
    formula = (f"{outcome_col} ~ C({group_col}, Treatment(reference='{reference_group}'))"
               f" * {arm_col}")
    model = smf.gee(formula, groups=cluster_col, data=df,
                    family=sm.families.Binomial(), cov_struct=cov)
    res = model.fit()

    # Locate the interaction term structurally rather than hardcoding
    # statsmodels' formatted name (which embeds the Treatment() spec).
    interaction_names = [n for n in res.params.index if ":" in n and arm_col in n]
    if len(interaction_names) != 1:
        raise RuntimeError(f"Could not uniquely identify interaction term: {list(res.params.index)}")
    iname = interaction_names[0]

    beta = float(res.params[iname])
    ci_low, ci_high = res.conf_int().loc[iname]

    # Raw drift rates per cell, for a human-readable companion to the OR
    cell_rates = (df.groupby([group_col, arm_col])[outcome_col].mean().to_dict())
    delta_treatment = cell_rates.get((treatment_group, 1), 0) - cell_rates.get((treatment_group, 0), 0)
    delta_reference = cell_rates.get((reference_group, 1), 0) - cell_rates.get((reference_group, 0), 0)

    return {
        "treatment_group":   treatment_group,
        "reference_group":   reference_group,
        "interaction_logodds": beta,
        "odds_ratio":        float(np.exp(beta)),
        "odds_ratio_95ci":   [float(np.exp(ci_low)), float(np.exp(ci_high))],
        "p_value":           float(res.pvalues[iname]),
        "delta_treatment":   round(float(delta_treatment), 4),
        "delta_reference":   round(float(delta_reference), 4),
        "raw_did":           round(float(delta_treatment - delta_reference), 4),
        "n_customers":       int(df[cluster_col].nunique()),
        "n_observations":    int(len(df)),
        "cov_struct":        cov_struct,
    }


def align_values_by_customer(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    field: str,
    perturbed_only: bool = False,
) -> Tuple[List[Any], List[Any], List[str]]:
    """
    Generalizes align_drift_by_customer() to any field, not just the
    boolean 'drifted' outcome — e.g. pairing agent_confidence across two
    conditions for the same customer. Same customer-intersection logic,
    same defensive reasoning (conditions should share the same population
    in practice, but intersecting rather than assuming protects against
    silent misalignment).

    perturbed_only: restrict to customers whose data was actually
    perturbed in BOTH conditions (dither_fields non-empty) — the same
    exposure-adjustment principle as align_drift_by_customer's option,
    valid for the same reason: perturbation is set by each condition's
    own random draw, never by the agent's outcome.

    Returns (values_a, values_b, shared_customer_ids) — the customer id
    list is returned (unlike the drift-specific version) since callers
    of this generic version more often need it for further joins (e.g.
    intersecting with ground truth stability tier).
    """
    if perturbed_only:
        records_a = [r for r in records_a if r["dither_fields"]]
        records_b = [r for r in records_b if r["dither_fields"]]
    by_a = {r["customer_id"]: r[field] for r in records_a}
    by_b = {r["customer_id"]: r[field] for r in records_b}
    shared = sorted(set(by_a) & set(by_b))
    return [by_a[c] for c in shared], [by_b[c] for c in shared], shared


def gee_style_plausibility_test(
    rows: List[Dict[str, Any]],
    mechanism_col: str = "mechanism",
    outcome_col: str = "drift",
    cluster_col: str = "customer_id",
    field_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    H4's mechanism comparison: two orthogonal contrasts on a 3-level
    Mechanism factor (drift / plausible entry error / implausible entry
    error), via GEE (logit link, clustered by customer).

    Uses HAND-CONSTRUCTED numeric contrast columns, not a library's
    built-in Helmert contrast class -- deliberately. A built-in class's
    internal ordering/scaling convention is exactly the kind of thing
    that can silently flip a sign without erroring. Building the columns
    directly means what's in the data IS the convention: nothing to
    misinterpret, and the same numeric columns used here are exactly
    what the verification fixtures construct by hand.

    Style contrast: drift=-2, plausible=+1, implausible=+1 (drift vs.
        the average of both entry-error types).
    Plausibility contrast: drift=0, plausible=+1, implausible=-1.
        LOCKED SIGN CONVENTION:
          positive & significant -> Garbage Filter Effect (plausible
            errors slip through and cause MORE drift than implausible
            ones, which the agent apparently recognizes as unphysical).
          negative & significant -> Outlier Vulnerability (implausible,
            unphysical errors disrupt the agent MORE than plausible
            ones -- the opposite finding).

    Neither contrast is meaningful for "drift" rows on the plausibility
    axis (gradual drift has no plausibility dimension) -- drift=0 on
    that contrast correctly removes it from that comparison entirely,
    which is exactly why this needed hand-built columns rather than a
    factor's default treatment coding.

    Scope: tests the POOLED style/plausibility effect. Per the working
    document, this should only be trusted as the headline finding if a
    Field x Mechanism interaction check (built separately, on top of
    this primitive) is NOT significant -- each field's entry-error
    operator is structurally different, and a pooled effect could
    average away a real per-field difference.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    valid = {"drift", "plausible", "implausible"}
    if not set(df[mechanism_col].unique()) <= valid:
        raise ValueError(f"mechanism_col must only contain {valid}, "
                         f"got {set(df[mechanism_col].unique())}")

    contrast_map = {"drift": (-2, 0), "plausible": (1, 1), "implausible": (1, -1)}
    df["style_contrast"] = df[mechanism_col].map(lambda m: contrast_map[m][0])
    df["plausibility_contrast"] = df[mechanism_col].map(lambda m: contrast_map[m][1])

    if df[outcome_col].nunique() < 2:
        return {"style_coefficient": None, "plausibility_coefficient": None,
                "note": "Outcome has no variation -- not estimable."}

    formula = f"{outcome_col} ~ style_contrast + plausibility_contrast"
    if field_col:
        formula += f" + C({field_col})"

    model = smf.gee(formula, groups=cluster_col, data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    res = model.fit()

    plaus_beta = float(res.params["plausibility_contrast"])
    style_beta = float(res.params["style_contrast"])
    ci = res.conf_int()
    # SCALING (verified against a direct 2x2 odds calculation, 2026-08):
    # the plausibility contrast is coded +1/-1, so its coefficient is HALF
    # the log-odds gap between plausible and implausible; the odds ratio
    # for plausible-vs-implausible is exp(2*beta), NOT exp(beta) (an
    # earlier version of this function returned exp(beta), understating
    # the effect). The style contrast is coded -2/+1/+1, so its coefficient
    # is one THIRD of the gap between drift and the average entry-error
    # log-odds; exp(3*beta) is the entry-error-vs-drift odds ratio, where
    # "entry error" means the average on the LOG-ODDS scale.
    interpretation = (
        "Garbage Filter Effect: plausible errors cause more drift than implausible ones"
        if plaus_beta > 0 else
        "Outlier Vulnerability: implausible errors cause more drift than plausible ones"
    )

    return {
        "style_coefficient":        float(res.params["style_contrast"]),
        "style_p_value":            float(res.pvalues["style_contrast"]),
        "plausibility_coefficient": plaus_beta,
        "plausibility_p_value":     float(res.pvalues["plausibility_contrast"]),
        "plausibility_odds_ratio":  float(np.exp(2 * plaus_beta)),  # plausible vs implausible
        "plausibility_odds_ratio_95ci": [float(np.exp(2 * ci.loc["plausibility_contrast"][0])),
                                         float(np.exp(2 * ci.loc["plausibility_contrast"][1]))],
        "style_odds_ratio":         float(np.exp(3 * style_beta)),  # avg entry error (log-odds scale) vs drift
        "interpretation":           interpretation,
        "n_customers":              int(df[cluster_col].nunique()),
        "n_observations":           int(len(df)),
    }


def gee_field_mechanism_interaction_gate(
    rows: List[Dict[str, Any]],
    field_col: str = "field",
    mechanism_col: str = "mechanism",
    outcome_col: str = "drift",
    cluster_col: str = "customer_id",
) -> Dict[str, Any]:
    """
    Decides whether the pooled style/plausibility contrasts from
    gee_style_plausibility_test() are safe to report, or whether each
    field's entry-error operator is different enough that per-field
    contrasts must carry the conclusion instead.

    Fits ONE model with both contrasts AND their interactions with
    Field: drift ~ C(field) + style + plausibility + style:C(field) +
    plausibility:C(field). A joint Wald test on each contrast's
    interaction terms (2 coefficients per contrast, since Field has 3
    levels) answers "does this contrast's effect differ by field."

    Uses the exact same hand-built style_contrast/plausibility_contrast
    columns as gee_style_plausibility_test() -- same sign convention,
    same reasoning for hand-building rather than a library's Helmert
    class. Term names for the joint Wald test are matched EXACTLY
    ('style_contrast:C(field)', not a substring/":" search) -- verified
    necessary, since a formula with two separate interaction groups
    produces two distinctly-named terms, and a generic ":" match would
    silently grab whichever one statsmodels happened to list first.

    Verified by simulation (see h4_working_doc.md): on a genuinely
    opposite-sign two-field scenario, this gate fired in 20/20
    replicates; on MILD same-sign heterogeneity (per-field plausibility
    effects of ~1.3 vs ~0.9), it still fired in 100/100. At n=1000 this
    gate is powerful enough to catch nearly any real between-field
    difference -- the intended conservative behavior: prefer
    over-triggering per-field reporting over risking a misleading pooled
    number (a pooled effect near zero was found significant in 19/20
    replicates on the opposite-sign scenario despite describing neither
    field).

    Returns both interaction p-values and a boolean recommendation per
    contrast. When a contrast is not safe to pool, gee_style_plausibility_test()
    should be re-run separately for each field's own rows instead.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    contrast_map = {"drift": (-2, 0), "plausible": (1, 1), "implausible": (1, -1)}
    valid = set(contrast_map)
    if not set(df[mechanism_col].unique()) <= valid:
        raise ValueError(f"{mechanism_col} must only contain {valid}, "
                         f"got {set(df[mechanism_col].unique())}")

    df["style_contrast"] = df[mechanism_col].map(lambda m: contrast_map[m][0])
    df["plausibility_contrast"] = df[mechanism_col].map(lambda m: contrast_map[m][1])

    formula = (f"{outcome_col} ~ C({field_col}) + style_contrast + plausibility_contrast + "
               f"style_contrast:C({field_col}) + plausibility_contrast:C({field_col})")
    model = smf.gee(formula, groups=cluster_col, data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    res = model.fit()

    terms = res.wald_test_terms(scalar=True).table
    style_term = f"style_contrast:C({field_col})"
    plaus_term = f"plausibility_contrast:C({field_col})"
    if style_term not in terms.index or plaus_term not in terms.index:
        raise RuntimeError(
            f"Expected interaction terms not found in Wald test table. "
            f"Got: {terms.index.tolist()}")

    style_p = float(terms.loc[style_term].iloc[1])
    plaus_p = float(terms.loc[plaus_term].iloc[1])

    return {
        "style_interaction_p_value":        style_p,
        "plausibility_interaction_p_value": plaus_p,
        "safe_to_pool_style":               style_p >= 0.05,
        "safe_to_pool_plausibility":        plaus_p >= 0.05,
        "n_customers":                      int(df[cluster_col].nunique()),
        "n_fields":                         int(df[field_col].nunique()),
        "recommendation": (
            "Per-field contrasts required for at least one axis -- pooled "
            "result(s) would be descriptive only."
            if (style_p < 0.05 or plaus_p < 0.05) else
            "Pooled style and plausibility contrasts are safe to report "
            "as the headline finding."
        ),
    }


# ============================================================================
# H5 — FROZEN DETECTION KEYWORD LIST (25 patterns, adverb-form gap fixed)
# ============================================================================
# Patterns locked in 1b_DESIGN_AMENDMENT_1.md. The adverb-form gap (e.g.
# "unusually" not matching \bunusual\b) was found and fixed during the H5
# design review -- the original patterns promised to close inflectional
# gaps via regex but seven adjective patterns didn't actually include an
# adverb suffix. Verified against concrete sentences before being locked
# here, not assumed correct.

H5_KEYWORD_PATTERNS = [
    # Direct inconsistency language
    r"\binconsisten(?:t|cy|cies)\b",
    r"\bdoes(?:n't| not) match\b",
    r"\bcontradict(?:s|ion|ing|ed)?\b",
    r"\bconflict(?:s|ing)?\b",
    # Plausibility/surprise language (adverb forms added)
    r"\bunusual(?:ly)?\b",
    r"\batypical(?:ly)?\b",
    r"\bimplausib(?:le|ly)\b",
    r"\bseems? off\b",
    r"\bdoes(?:n't| not) add up\b",
    r"\bodd(?:ly)?\b",
    r"\bstrange(?:ly)?\b",
    r"\bsurpris(?:ing|e|ed|ingly)\b",
    r"\banomal(?:y|ies|ous|ously)\b",
    # Doubt/verification language (adverb forms added)
    r"\b(?:hard|difficult) to reconcile\b",
    r"\bquestionable|questionably\b",
    r"\bsuspicious(?:ly)?\b",
    r"\bseems? wrong\b",
    r"\bappears? incorrect\b",
    r"\bmay be (?:an )?error\b",
    r"\b(?:possible|likely|apparent) error\b",
    r"\bdata error\b",
    r"\bmistake in (?:the )?data\b",
    # Explicit data-quality language
    r"\bdata quality\b",
    r"\bdata issue\b",
    r"\bdata problem\b",
]
_H5_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in H5_KEYWORD_PATTERNS]


def detect_h5_keywords(text: str) -> Dict[str, Any]:
    """
    Scans decision_reasoning for the frozen 25-pattern detection keyword
    list. Deterministic, auditable, zero marginal cost -- the primary H5
    metric, per 1b_DESIGN_AMENDMENT_1.md. "Uncertain about" deliberately
    excluded (object-ambiguous; decision-level uncertainty is already
    measured directly via agent_confidence).

    Returns detected (bool) and matched_patterns (which specific patterns
    fired, for transparency/debugging -- not meant to be over-interpreted
    per-pattern, since the list is reported as one pooled metric).
    """
    if not text:
        return {"detected": False, "matched_patterns": []}
    matched = [p for p, compiled in zip(H5_KEYWORD_PATTERNS, _H5_COMPILED_PATTERNS)
               if compiled.search(text)]
    return {"detected": len(matched) > 0, "matched_patterns": matched}

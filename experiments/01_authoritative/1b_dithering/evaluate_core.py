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

    SCOPE WARNING: valid only WITHIN a single customer's own two small
    score sets. Do NOT pool scores across multiple customers and feed
    them here — the same baseline texts feed both a customer's dithered-
    vs-baseline scores AND their own self-similarity scores, so pooling
    across the population would compare correlated data as if it were
    independent, exactly the mistake McNemar's test was built to avoid
    for drift rates. For a condition-level (whole-population) finding,
    use jaccard_condition_level_shift() instead, which correctly reduces
    each customer to one paired difference before testing.

    Chosen over a t-test because Jaccard scores are bounded [0,1], often
    skewed, and we're comparing two whole SETS of scores rather than one
    new point against a reference distribution — genuinely different
    data shape than the confidence comparison above.

    dithered_vs_baseline_scores: Jaccard(dithered_reasoning, each
        matching baseline reasoning text) — one score per baseline text,
        for ONE customer
    baseline_self_similarity_scores: Jaccard between every PAIR of THAT
        SAME customer's baseline reasoning texts (C(n,2) scores) — their
        own natural reasoning variability, with no dithering involved
    """
    if len(dithered_vs_baseline_scores) < 1 or len(baseline_self_similarity_scores) < 1:
        return {"u_statistic": None, "p_value": None,
                "note": "Insufficient data for dispersion test"}

    u_stat, p_value = stats.mannwhitneyu(
        dithered_vs_baseline_scores, baseline_self_similarity_scores,
        method='exact', alternative='two-sided',
    )

    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "n_dithered": len(dithered_vs_baseline_scores),
        "n_baseline_self": len(baseline_self_similarity_scores),
        "dithered_mean": float(np.mean(dithered_vs_baseline_scores)),
        "baseline_self_mean": float(np.mean(baseline_self_similarity_scores)),
    }


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

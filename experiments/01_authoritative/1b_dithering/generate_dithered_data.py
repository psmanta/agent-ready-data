#!/usr/bin/env python3
"""
Generate Dithered Data — Experiment 1b: Dithering
====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Generates all UNCONDITIONAL data artifacts needed for Experiment 1b:
  1. Ground truth dataset (DAMA-validated canonical customers)
  2. Baseline agent input (clean data, stripped for agent consumption)
  3. All 44 unconditional dither condition files (agent input + reference)
     covering H1 (12), H2 (12), H3 (11), H4 (3), H7 (4), H8a (2)

H8b (0-1 conditions) is NOT generated here — it is conditional on agent
decisions from h8a_pair2_purchase_risk, h1_category_purchase_behavior,
and h1_category_risk_factors, none of which exist until the agent has
actually run against this script's output. See check_and_generate_h8b.py,
which runs AFTER those three conditions have decisions.

Does NOT run the agent — this script only produces the JSONL/JSON
files. Running the agent against these files is a separate step
(business_decision_agent.py, run_baseline.sh, run_dither_conditions.sh).

Usage:
    # Generate everything: ground truth, baseline, and all 44 conditions
    python generate_dithered_data.py --n 1000 --seed 42

    # Generate only ground truth + baseline (skip dither conditions)
    python generate_dithered_data.py --n 1000 --seed 42 --baseline-only

    # Regenerate a single condition (for debugging)
    python generate_dithered_data.py --n 1000 --seed 42 --condition h2_churn_risk_score_mag15pct
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Resolve shared directory — 1b_dithering -> 01_authoritative -> experiments -> project_root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
shared_dir = project_root / "shared"
sys.path.insert(0, str(shared_dir / "data_generation"))

from base_customer_generator import (
    generate_base_customers,
    save_canonical_customers,
    CONSISTENCY_RULES,
)
from validate_dama_dimensions import run_audit
from dither_engine import (
    DitherEngine,
    DitherConfig,
    build_all_conditions,
    validate_condition_ids_unique,
    BOOLEAN_FIELDS,
    save_dithered_condition,
    save_dither_reference,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_N = 1000
DEFAULT_SEED = 42

# Fields that must never be shown to the agent — internal bookkeeping only
INTERNAL_ONLY_FIELDS = {"customer_id"}


def strip_internal_fields(customer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove fields the agent must never see: customer_id (identity linkage)
    and any _dither_* metadata. Add a record_id for tracing instead.
    """
    record = {k: v for k, v in customer.items()
              if k not in INTERNAL_ONLY_FIELDS and not k.startswith("_dither")}
    return record


def assign_record_ids(customers: List[Dict[str, Any]], prefix: str = "REC") -> List[Dict[str, Any]]:
    """
    Assign a record_id to each customer for agent-facing tracing.
    Uses a hash of customer_id so it's deterministic but doesn't leak identity.
    """
    import hashlib
    result = []
    for c in customers:
        c = dict(c)  # shallow copy
        h = hashlib.md5(c["customer_id"].encode()).hexdigest()[:12].upper()
        c["record_id"] = f"{prefix}_{h}"
        result.append(c)
    return result


def verify_uniqueness(
    records: List[Dict[str, Any]],
    context: str,
    id_field: str = "record_id",
) -> None:
    """
    Verify that a list of records has no duplicate IDs before writing to disk.

    This experiment intentionally excludes duplication as a variable — 1a
    already tested duplication effects, and 1b's dither conditions must not
    accidentally reintroduce duplicate records as a confound. Every file
    written by this script passes through this check first.

    Raises ValueError immediately if any duplicate is found — fails loudly
    rather than silently writing a corrupted dataset.

    Args:
        records:  List of record dicts to check
        context:  Human-readable description of what's being checked,
                  used in the error message (e.g. "baseline agent input",
                  "h2_churn_risk_score_mag15pct agent input")
        id_field: Field name to check for uniqueness (default: record_id)
    """
    ids = [r.get(id_field) for r in records]
    seen = set()
    duplicates = set()

    for id_val in ids:
        if id_val in seen:
            duplicates.add(id_val)
        seen.add(id_val)

    if duplicates:
        raise ValueError(
            f"Duplicate {id_field} detected in {context}: "
            f"{len(duplicates)} duplicate value(s) found "
            f"(e.g. {list(duplicates)[:3]}). "
            f"1b explicitly excludes duplication as a variable — "
            f"this must be investigated before proceeding."
        )

    if len(ids) != len(set(ids)):
        # Should be unreachable given the check above, but belt-and-suspenders
        raise ValueError(f"Record count mismatch in {context} — possible data corruption")


def verify_customer_uniqueness(
    records: List[Dict[str, Any]],
    context: str,
) -> None:
    """
    Verify customer_id uniqueness specifically. Used for files that retain
    customer_id (ground truth, dither reference files) rather than agent
    input files (which use record_id only, with customer_id stripped).
    """
    verify_uniqueness(records, context, id_field="customer_id")


# ============================================================================
# STEP 1: GROUND TRUTH
# ============================================================================

def generate_ground_truth(n: int, seed: int, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Generate the canonical ground truth dataset and validate it against
    all six DAMA dimensions. Raises if validation fails — ground truth
    must be clean before anything else proceeds.
    """
    print(f"\n{'='*60}")
    print("STEP 1: Ground Truth Generation")
    print(f"{'='*60}")
    print(f"Generating {n} customers (seed={seed})...")

    customers = generate_base_customers(n=n, seed=seed)
    print(f"  Generated {len(customers)} customers")

    print("\nVerifying customer_id uniqueness...")
    verify_customer_uniqueness(customers, context="ground truth generation")
    print(f"  PASSED — {len(customers)} unique customer_ids, no duplicates")

    print("\nValidating against DAMA dimensions...")
    report = run_audit(customers, verbose=False)
    if report["passed"]:
        print("  PASSED — all six DAMA dimensions")
    else:
        raise ValueError("Ground truth failed DAMA validation — see audit report")

    # Save the audit report as a verifiable artifact alongside the data.
    # Anyone skeptical of the DAMA-compliance claim can inspect this file
    # directly, or independently re-run validate_dama_dimensions.py against
    # canonical_customers.json to verify the claim themselves.
    audit_path = output_dir / "ground_truth" / "dama_audit_report.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved audit report: {audit_path}")

    gt_path = output_dir / "ground_truth" / "canonical_customers.json"
    save_canonical_customers(customers, gt_path)

    return customers


# ============================================================================
# STEP 2: BASELINE AGENT INPUT
# ============================================================================

def generate_baseline_input(
    customers: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Produce the baseline agent input file — clean data, no customer_id,
    record_id assigned for tracing. This is the file run 5 times through
    the agent at temperature 0.0 to establish the ground truth decision
    baseline (majority vote + stability classification).
    """
    print(f"\n{'='*60}")
    print("STEP 2: Baseline Agent Input")
    print(f"{'='*60}")

    with_record_ids = assign_record_ids(customers, prefix="BASE")

    print("Verifying record_id uniqueness...")
    verify_uniqueness(with_record_ids, context="baseline agent input")
    print(f"  PASSED — {len(with_record_ids)} unique record_ids, no duplicates")

    agent_facing = [strip_internal_fields(c) for c in with_record_ids]

    # Also save a customer_id <-> record_id map for the evaluator
    id_map = {c["record_id"]: orig["customer_id"]
              for c, orig in zip(with_record_ids, with_record_ids)}

    out_path = output_dir / "baseline" / "agent_input" / "baseline_customers.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in agent_facing:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  Saved: {out_path} ({len(agent_facing)} records)")

    map_path = output_dir / "baseline" / "record_id_map.json"
    with open(map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"  Saved: {map_path}")


# ============================================================================
# STEP 3: DITHER CONDITIONS
# ============================================================================

# ============================================================================
# PERTURBATION VALIDITY CHECK — runs before any condition is written
# ============================================================================
#
# Added 2026-08 after an audit found the engine silently failing to
# perturb data the way the hypotheses assumed: acquisition_channel never
# changed at any magnitude (missing code branch); small-integer counts
# barely changed (support_tickets_open 7.9%, payment_failures 2.8% at 15%,
# since multiplicative drift rounds 1 or 2 back to itself and 0 stays 0);
# and H3's "correlated" arm was no more coherent than its uncorrelated arm.
# None of these raised an error. This check asks, for every condition,
# "did the data actually get perturbed the way the hypothesis claims?" —
# for free, before any API spend.

def check_condition_validity(config, dithered, min_exposure: float = 0.80) -> Dict[str, Any]:
    """
    Per-field EXPOSURE: share of perturbed customers whose field actually
    changed (from _dither_fields, which lists only fields that changed).
      - numeric / categorical fields: must meet min_exposure (default 80%).
        Drift rates across fields are only comparable when exposure is
        comparable.
      - boolean fields: exposure is BY DESIGN equal to magnitude (flip
        probability), so it's checked against the magnitude within a 3-SE
        binomial tolerance instead of the 80% bar. Cross-field comparisons
        involving booleans must use the evaluator's exposure-adjusted
        drift rate.

    COHERENCE (coupled H3 conditions only): among customers where every
    field moved, the share whose joint movement matches the coupling signs.
      - correlated arm: must be >= 95%
      - uncorrelated arm: must sit near chance, 2 / 2^k (both signs of the
        shared pattern), within 0.08
    """
    perturbed = [c for c in dithered if c.get("_dither_applied")]
    failures, exposure = [], {}
    n = len(perturbed)
    if n == 0:
        return {"condition_id": config.condition_id, "passed": False,
                "failures": ["no customers were perturbed"], "field_exposure": {}}

    for f in config.fields:
        n_changed = sum(1 for c in perturbed if f in c["_dither_fields"])
        n_blocked = sum(1 for c in perturbed if f in c.get("_dither_blocked", []))
        n_possible = n - n_blocked
        raw_rate = n_changed / n
        # Denominator excludes customers for whom a move was structurally
        # IMPOSSIBLE (e.g. payment_failures=0, drawn direction=down) — not
        # customers the mechanism simply failed to move, which is exactly
        # what this check exists to catch. n_blocked=0 for every field
        # type except numeric drift, where the concept applies.
        possible_rate = (n_changed / n_possible) if n_possible else None
        exposure[f] = {
            "raw_rate":      round(raw_rate, 4),
            "n_blocked":     n_blocked,
            "possible_rate": round(possible_rate, 4) if possible_rate is not None else None,
        }
        if f in BOOLEAN_FIELDS:
            mag = config.get_magnitude(f)
            tol = 3 * (mag * (1 - mag) / n) ** 0.5 + 0.01
            if abs(raw_rate - mag) > tol:
                failures.append(f"boolean {f}: exposure {raw_rate:.1%} outside "
                                f"expected {mag:.1%} +/- {tol:.1%}")
        elif possible_rate is not None and possible_rate < min_exposure:
            failures.append(f"{f}: possible-move exposure {possible_rate:.1%} "
                            f"below {min_exposure:.0%} ({n_blocked} of {n} "
                            f"customers structurally blocked, correctly excluded)")

    coherence = None
    if config.coupling_signs is not None:
        signs = config.coupling_signs
        fields = list(signs)
        judged = coherent = 0
        for c in perturbed:
            if not all(f in c["_dither_fields"] for f in fields):
                continue
            d = {f: (1 if c[f] > c["_dither_original"][f] else -1) for f in fields}
            judged += 1
            if all(d[a] * d[b] == signs[a] * signs[b]
                   for i, a in enumerate(fields) for b in fields[i + 1:]):
                coherent += 1
        rate = coherent / judged if judged else 0.0
        chance = 2 / (2 ** len(fields))
        # Sample-size-adjusted tolerance, not a flat constant -- same
        # principle as the boolean exposure check just above. A flat 0.08
        # band is ~5 SE at n~1000 (safe) but only ~1.5 SE at a smoke
        # test's n~100 (false-alarm-prone: an 8pp deviation from 50%
        # chance is unremarkable sampling noise at that scale). Caught by
        # running the free validity check at a smaller n than the real
        # experiment specifically to stress-test cases n=1000 would never
        # surface -- confirmed by reproducing the exact failure and
        # computing its z-score (2.18, p=0.029) against a null of no bug.
        coherence_tol = 3 * (chance * (1 - chance) / judged) ** 0.5 + 0.01 if judged else 0.08
        coherence = {"rate": round(rate, 4), "n_judged": judged, "chance": chance,
                    "tolerance": round(coherence_tol, 4)}
        if config.correlated and rate < 0.95:
            failures.append(f"correlated arm coherence {rate:.1%} below 95%")
        if not config.correlated and abs(rate - chance) > coherence_tol:
            failures.append(f"uncorrelated arm coherence {rate:.1%} not near chance "
                            f"{chance:.0%} (+/-{coherence_tol:.1%} at n={judged})")

    return {"condition_id": config.condition_id, "passed": not failures,
            "failures": failures, "field_exposure": exposure,
            "coherence": coherence, "n_perturbed": n}


def generate_all_conditions(
    customers: List[Dict[str, Any]],
    output_dir: Path,
    only_condition: str = None,
    min_exposure: float = 0.80,
    allow_invalid: bool = False,
) -> None:
    """
    Generate agent input + reference files for all 55 unconditional H1,
    H2, H3, H4, H7, H8a dither conditions. If only_condition is specified,
    regenerate just that one condition (useful for debugging without
    regenerating all 50).

    Every condition passes check_condition_validity() BEFORE its files are
    written. A failing condition is not written at all (so the agent runner
    can never pick up data that doesn't perturb what its hypothesis claims),
    the full report is saved to validity_report.json, and the script exits
    with an error. allow_invalid=True writes everything anyway, for
    deliberate debugging only.

    H8b is NOT included here — it is conditional on agent decisions that
    do not exist yet at this point in the pipeline (H8a's
    h1_category_purchase_behavior, h1_category_risk_factors, and
    h8a_pair2_purchase_risk must all have been run through the agent
    first). See check_and_generate_h8b.py, run after the agent has
    processed all 55 conditions here.
    """
    print(f"\n{'='*60}")
    print("STEP 3: Dither Conditions")
    print(f"{'='*60}")

    all_configs = build_all_conditions()

    print("Verifying condition_id uniqueness across all conditions...")
    validate_condition_ids_unique(all_configs)
    print(f"  PASSED — {len(all_configs)} unique condition_ids, no collisions")

    if only_condition:
        all_configs = [c for c in all_configs if c.condition_id == only_condition]
        if not all_configs:
            raise ValueError(f"Unknown condition_id: {only_condition}")

    print(f"\nGenerating {len(all_configs)} condition(s)...\n")

    validity_results = []
    for config in all_configs:
        print(f"  [{config.condition_id}]")
        engine = DitherEngine(config)
        dithered = engine.apply(customers)

        validity = check_condition_validity(config, dithered, min_exposure)
        validity_results.append(validity)
        def _fmt(e):
            s = f"{e['raw_rate']:.0%}"
            if e["possible_rate"] is not None and e["n_blocked"] > 0:
                s += f" ({e['possible_rate']:.0%} of possible, {e['n_blocked']} blocked)"
            return s
        exp_str = ", ".join(f"{f}={_fmt(r)}" for f, r in validity["field_exposure"].items())
        coh = validity.get("coherence")
        coh_str = f" | coherence {coh['rate']:.0%} (chance {coh['chance']:.0%})" if coh else ""
        print(f"    exposure: {exp_str}{coh_str}")
        if not validity["passed"]:
            for msg in validity["failures"]:
                print(f"    ❌ INVALID: {msg}")
            if not allow_invalid:
                print(f"    -> NOT written")
                continue

        with_record_ids = assign_record_ids(dithered, prefix="COND")

        verify_uniqueness(with_record_ids,
            context=f"{config.condition_id} agent input")
        verify_customer_uniqueness(with_record_ids,
            context=f"{config.condition_id} dither reference")

        cond_dir = output_dir / "conditions" / config.condition_id
        cond_dir.mkdir(parents=True, exist_ok=True)

        # Agent input — stripped of customer_id and _dither metadata
        agent_facing = [strip_internal_fields(c) for c in with_record_ids]
        agent_input_path = cond_dir / "agent_input.jsonl"
        with open(agent_input_path, "w") as f:
            for record in agent_facing:
                f.write(json.dumps(record, default=str) + "\n")

        # Reference — full record with customer_id and _dither metadata,
        # for evaluator use only. Never shown to the agent.
        reference_path = cond_dir / "dither_reference.json"
        with open(reference_path, "w") as f:
            json.dump(with_record_ids, f, indent=2, default=str)

        n_dithered = sum(1 for c in dithered if c.get("_dither_applied"))
        print(f"    agent_input.jsonl      ({len(agent_facing)} records)")
        print(f"    dither_reference.json  ({n_dithered} dithered)")

    report_path = output_dir / "validity_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = [v for v in validity_results if not v["passed"]]
    with open(report_path, "w") as f:
        json.dump({"min_exposure": min_exposure, "n_conditions": len(validity_results),
                   "n_failed": len(failed), "conditions": validity_results}, f, indent=2)
    print(f"\n  Validity report: {report_path}")

    if failed and not allow_invalid:
        ids = ", ".join(v["condition_id"] for v in failed)
        raise SystemExit(
            f"\n❌ {len(failed)} condition(s) failed the perturbation validity "
            f"check and were NOT written: {ids}\nSee {report_path}. Do not run "
            f"the agent until these are fixed.")
    print(f"  ✅ Done — {len(validity_results) - len(failed)} condition(s) "
          f"generated, all passing the validity check"
          + (f" ({len(failed)} INVALID written anyway via --allow-invalid)" if failed else ""))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth, baseline, and dither condition data for 1b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything
  python generate_dithered_data.py --n 1000 --seed 42

  # Ground truth + baseline only, skip dither conditions
  python generate_dithered_data.py --n 1000 --seed 42 --baseline-only

  # Regenerate a single condition
  python generate_dithered_data.py --n 1000 --seed 42 --condition h2_churn_risk_score_mag15pct
        """
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N,
        help=f"Number of base customers (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--out", type=str, default="experiments_output",
        help="Output directory (default: experiments_output)")
    parser.add_argument("--baseline-only", action="store_true",
        help="Generate ground truth and baseline input only, skip dither conditions")
    parser.add_argument("--min-exposure", type=float, default=0.80,
        help="Minimum share of perturbed customers whose non-boolean field "
             "must actually change (default: 0.80)")
    parser.add_argument("--allow-invalid", action="store_true",
        help="Write conditions that fail the validity check anyway "
             "(debugging only — never run the agent on these)")
    parser.add_argument("--condition", type=str, default=None,
        help="Regenerate only this specific condition_id")

    args = parser.parse_args()
    output_dir = Path(args.out)

    print(f"\n{'#'*60}")
    print(f"# Experiment 1b: Dithering — Data Generation")
    print(f"# n={args.n}  seed={args.seed}  out={output_dir}")
    print(f"{'#'*60}")

    # Step 1: Ground truth
    customers = generate_ground_truth(args.n, args.seed, output_dir)

    # Step 2: Baseline
    generate_baseline_input(customers, output_dir)

    # Step 3: Dither conditions (unless baseline-only)
    if not args.baseline_only:
        generate_all_conditions(customers, output_dir, only_condition=args.condition,
                                min_exposure=args.min_exposure,
                                allow_invalid=args.allow_invalid)

    print(f"\n{'#'*60}")
    print("# DONE")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

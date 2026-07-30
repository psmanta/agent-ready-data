#!/usr/bin/env python3
"""
Generate Dithered Data — Experiment 1b: Dithering
====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Generates all data artifacts needed for Experiment 1b:
  1. Ground truth dataset (DAMA-validated canonical customers)
  2. Baseline agent input (clean data, stripped for agent consumption)
  3. All H1-H4 dither condition files (agent input + reference)

Does NOT run the agent — this script only produces the JSONL/JSON
files. Running the agent against these files is a separate step
(business_decision_agent.py, run_baseline.sh, run_dither_conditions.sh).

Usage:
    # Generate everything: ground truth, baseline, and all 21 conditions
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
    build_h1_conditions,
    build_h2_conditions,
    build_h3_conditions,
    build_h4_conditions,
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

def generate_all_conditions(
    customers: List[Dict[str, Any]],
    output_dir: Path,
    only_condition: str = None,
) -> None:
    """
    Generate agent input + reference files for all H1-H4 dither conditions.
    If only_condition is specified, regenerate just that one condition
    (useful for debugging without regenerating all 21).
    """
    print(f"\n{'='*60}")
    print("STEP 3: Dither Conditions")
    print(f"{'='*60}")

    all_configs = (
        build_h1_conditions()
        + build_h2_conditions()
        + build_h3_conditions()
        + build_h4_conditions()
    )

    if only_condition:
        all_configs = [c for c in all_configs if c.condition_id == only_condition]
        if not all_configs:
            raise ValueError(f"Unknown condition_id: {only_condition}")

    print(f"Generating {len(all_configs)} condition(s)...\n")

    for config in all_configs:
        print(f"  [{config.condition_id}]")
        engine = DitherEngine(config)
        dithered = engine.apply(customers)

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

    print(f"\n  Done — {len(all_configs)} condition(s) generated")


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
        generate_all_conditions(customers, output_dir, only_condition=args.condition)

    print(f"\n{'#'*60}")
    print("# DONE")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Refine Boundary Convergence — Experiment 1b: Dithering
====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Implements the adaptive convergence rule for boundary customers (deeply_boundary + lightly_boundary),
locked before any real data existed:

  - Wilson score interval on each customer's plurality-decision proportion,
    recomputed fresh from ALL accumulated runs after every new batch
    (never locked onto whichever decision led first)
  - Batches of 10 additional runs at a time
  - Stop when the interval width <= 30 percentage points (+-15)
  - Hard cap at 60 total runs regardless of convergence — customers who
    hit the cap without converging are flagged explicitly as
    "did not converge," not silently treated as resolved

This is a diagnostic enrichment. It NEVER changes the primary
majority_decision used everywhere else in the experiment (H1-H8b
comparisons all continue to use the original 5-run baseline vote).
This module only produces a refined, better-resolved read for the
specific customers whose primary 5-run vote was too close to trust.

Usage (called after each batch of additional runs has been executed):
    python refine_boundary_convergence.py \
        --baseline_reference experiments_output/baseline/baseline_reference.json \
        --boundary_dir experiments_output/baseline/boundary_expansion \
        --batch_decisions_dir experiments_output/baseline/boundary_expansion/decisions \
        --record_id_map experiments_output/baseline/boundary_expansion/boundary_record_id_map.json

This script is idempotent and re-runnable: call it again after each new
batch of 10 runs completes. It tracks convergence state on disk and
reports which customers still need another batch.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PRECISION_THRESHOLD_PP = 30.0  # +-15pp, i.e. 30pp total interval width
MAX_TOTAL_RUNS = 60
BATCH_SIZE = 10
Z_SCORE_95 = 1.96  # for the Wilson score interval at 95% confidence


def wilson_interval(successes: int, n: int, z: float = Z_SCORE_95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.
    Used instead of the naive normal approximation because it behaves
    much better at small sample sizes (we start as low as n=5-15),
    where the normal approximation can produce nonsensical intervals
    (e.g. extending below 0% or above 100%).

    Returns (lower_bound, upper_bound) as proportions in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    denominator = 1 + (z**2 / n)
    center = (p_hat + (z**2) / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


def interval_width_pp(lower: float, upper: float) -> float:
    """Interval width in percentage points."""
    return (upper - lower) * 100


def compute_convergence(all_decisions: List[str]) -> Dict[str, Any]:
    """
    Given all accumulated decisions for one customer (across the primary
    5 runs plus any additional batches), recompute the plurality decision
    fresh and check convergence against the precision threshold.
    """
    n = len(all_decisions)
    counts = Counter(all_decisions)
    plurality_decision, plurality_count = counts.most_common(1)[0]

    lower, upper = wilson_interval(plurality_count, n)
    width = interval_width_pp(lower, upper)

    converged = width <= PRECISION_THRESHOLD_PP
    hit_cap = n >= MAX_TOTAL_RUNS

    return {
        "n_runs":               n,
        "decision_distribution": dict(counts),
        "plurality_decision":    plurality_decision,
        "plurality_rate":        round(plurality_count / n, 4),
        "wilson_interval":       [round(lower, 4), round(upper, 4)],
        "interval_width_pp":     round(width, 2),
        "converged":             converged,
        "hit_run_cap":           hit_cap,
        "needs_another_batch":   (not converged) and (not hit_cap),
        "status": (
            "converged" if converged
            else "did_not_converge_at_cap" if hit_cap
            else "needs_more_runs"
        ),
    }


def load_batch_decisions(decisions_dir: Path, batch_pattern: str = "batch*.decisions.jsonl") -> Dict[str, List[str]]:
    """
    Load all batch decision files, grouped by record_id.
    Expects files named like batch1.decisions.jsonl, batch2.decisions.jsonl, etc.
    Each batch file contains 10 runs' worth of decisions per customer,
    i.e. each record_id should appear 10 times per batch file (once per
    run within that batch) OR the batch file itself may already be
    organized as one-file-per-run — this function handles both by simply
    collecting every decision seen, in order, per record_id.
    """
    from collections import defaultdict
    by_record: Dict[str, List[str]] = defaultdict(list)

    batch_files = sorted(decisions_dir.glob(batch_pattern))
    for batch_file in batch_files:
        with open(batch_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                by_record[d["record_id"]].append(d["business_decision"])

    return dict(by_record)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_reference", type=str, required=True,
        help="Path to the PRIMARY baseline_reference.json (5-run vote)")
    parser.add_argument("--boundary_dir", type=str, required=True,
        help="Directory containing boundary_record_id_map.json")
    parser.add_argument("--batch_decisions_dir", type=str, required=True,
        help="Directory containing batch*.decisions.jsonl files from "
             "additional boundary expansion runs")
    parser.add_argument("--out", type=str, default=None,
        help="Output path for refined results (default: "
             "<boundary_dir>/refined_classification.json)")

    args = parser.parse_args()
    boundary_dir = Path(args.boundary_dir)
    batch_decisions_dir = Path(args.batch_decisions_dir)
    out_path = Path(args.out) if args.out else boundary_dir / "refined_classification.json"

    print(f"\n{'='*60}")
    print("Refine Boundary Convergence — Convergence Check")
    print(f"{'='*60}\n")

    with open(args.baseline_reference) as f:
        primary_baseline = json.load(f)

    with open(boundary_dir / "boundary_record_id_map.json") as f:
        record_to_customer = json.load(f)

    print(f"Loading batch decisions from {batch_decisions_dir}...")
    batch_decisions_by_record = load_batch_decisions(batch_decisions_dir)
    print(f"  Found decisions for {len(batch_decisions_by_record)} record_ids")

    results = {}
    needs_more = []

    for record_id, customer_id in record_to_customer.items():
        primary_entry = primary_baseline["customers"].get(customer_id)
        if primary_entry is None:
            print(f"  WARNING: {customer_id} not found in primary baseline — skipping")
            continue

        # Combine primary 5 runs with any additional batch runs so far
        primary_decisions = [r["business_decision"] for r in primary_entry["run_details"]]
        additional_decisions = batch_decisions_by_record.get(record_id, [])
        all_decisions = primary_decisions + additional_decisions

        convergence = compute_convergence(all_decisions)
        convergence["customer_id"] = customer_id
        convergence["record_id"] = record_id
        convergence["primary_majority_decision"] = primary_entry["majority_decision"]

        results[customer_id] = convergence

        if convergence["needs_another_batch"]:
            needs_more.append(customer_id)

    n_converged = sum(1 for r in results.values() if r["status"] == "converged")
    n_capped = sum(1 for r in results.values() if r["status"] == "did_not_converge_at_cap")
    n_needs_more = len(needs_more)

    print(f"\n{'='*60}")
    print("STATUS")
    print(f"{'='*60}")
    print(f"  Converged:              {n_converged}")
    print(f"  Needs another batch:    {n_needs_more}")
    print(f"  Hit cap, did NOT converge: {n_capped}")

    if needs_more:
        print(f"\n  Run another batch of {BATCH_SIZE} for these {n_needs_more} customer(s),")
        print(f"  then re-run this script. Customer IDs needing more runs saved to:")
        needs_more_path = boundary_dir / "needs_another_batch.json"
        with open(needs_more_path, "w") as f:
            json.dump(needs_more, f, indent=2)
        print(f"  {needs_more_path}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full refined classification saved: {out_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())

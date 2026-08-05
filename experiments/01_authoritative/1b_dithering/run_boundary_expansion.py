#!/usr/bin/env python3
"""
Run Boundary Expansion — Experiment 1b: Dithering
=====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Orchestrates the full adaptive convergence loop for boundary customers
(deeply_boundary + lightly_boundary + tied_no_majority — see
extract_boundary_subset.py for
why both tiers, not just deeply_boundary). Built as a Python driver
rather than a shell script, unlike run_baseline.sh: that script's job is
a fixed, unconditional 5-run loop with no branching. This orchestration
has real conditional control flow — dynamic batch sizing, checking JSON
state to decide whether to continue, per-round convergence checks — which
is a much better fit for Python than bash's JSON handling.

The loop:
  1. Extract the boundary subset once (deeply_boundary + lightly_boundary
     + tied_no_majority
     customers, per the primary 5-run baseline_reference.json)
  2. For each round:
     a. Compute this round's batch size: min(BATCH_SIZE, MAX_TOTAL_RUNS
        minus the current run count for still-active customers) — shrinks
        on the final round so nobody exceeds the 60-run cap (a naive
        fixed batch-of-10 schedule would overshoot: 5+10*6=65 > 60)
     b. Build the batch's agent input (everyone, on round 1; only
        still-unconverged customers, on every round after)
     c. If nobody needs this round, stop — everyone has resolved
     d. Run the agent against the batch
     e. Recompute convergence for every boundary customer using ALL
        accumulated decisions so far (primary 5 + every batch run to
        date) — the plurality decision is recomputed fresh each round,
        never locked onto whichever decision led first
  3. Stop when everyone has converged, or the safety-capped round count
     is reached (a backstop against an unforeseen bug causing an infinite
     loop — should never actually trigger given the batch-size scheduling
     above, but costs nothing to have as a second line of defense)

Usage:
    python run_boundary_expansion.py \
        --baseline_reference experiments_output/baseline/baseline_reference.json \
        --baseline_input experiments_output/baseline/agent_input/baseline_customers.jsonl \
        --record_id_map experiments_output/baseline/record_id_map.json \
        --out experiments_output/baseline/boundary_expansion \
        --model claude-haiku-4-5-20251001
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_boundary_subset import load_boundary_customer_ids, extract_subset
from prepare_next_batch import prepare_batch_input
from refine_boundary_convergence import (
    load_batch_decisions, compute_convergence, PRECISION_THRESHOLD_PP, MAX_TOTAL_RUNS
)

BATCH_SIZE = 10
PRIMARY_RUNS = 5
SAFETY_MAX_ROUNDS = 10  # backstop only — should never be reached given the
                         # batch-size schedule below correctly lands at the
                         # cap by round 6


def run_agent(input_path: Path, output_path: Path, model: str, agent_script: Path) -> None:
    """
    Run business_decision_agent.py as a subprocess against a batch input
    file. Subprocess (not a direct import) because this is the one step
    that actually calls the Anthropic API and may have its own process
    lifecycle, logging, and error handling worth keeping isolated.
    """
    result = subprocess.run(
        [sys.executable, str(agent_script),
         "--input", str(input_path),
         "--output", str(output_path),
         "--model", model,
         "--temperature", "0.0"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Agent run failed (exit {result.returncode}):\n{result.stderr}"
        )
    print(result.stdout)


def run_convergence_check(
    baseline_reference_path: Path,
    boundary_dir: Path,
    decisions_dir: Path,
) -> dict:
    """
    Recompute convergence for every boundary customer using all
    accumulated batch decisions to date. Returns the results dict and
    also writes refined_classification.json + needs_another_batch.json
    to disk (same artifacts refine_boundary_convergence.py's CLI would
    produce — this calls the same underlying functions directly).
    """
    with open(baseline_reference_path) as f:
        primary_baseline = json.load(f)
    with open(boundary_dir / "boundary_record_id_map.json") as f:
        record_to_customer = json.load(f)

    batch_decisions_by_record = load_batch_decisions(decisions_dir)

    results = {}
    needs_more = []

    for record_id, customer_id in record_to_customer.items():
        primary_entry = primary_baseline["customers"].get(customer_id)
        if primary_entry is None:
            continue

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

    with open(boundary_dir / "refined_classification.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(boundary_dir / "needs_another_batch.json", "w") as f:
        json.dump(needs_more, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_reference", type=str, required=True)
    parser.add_argument("--baseline_input", type=str, required=True)
    parser.add_argument("--record_id_map", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--agent_script", type=str,
        default=str(Path(__file__).resolve().parent / "business_decision_agent.py"),
        help="Path to business_decision_agent.py (override for testing "
             "with a mock agent)")

    args = parser.parse_args()
    boundary_dir = Path(args.out)
    decisions_dir = boundary_dir / "decisions"
    agent_script = Path(args.agent_script)

    print(f"\n{'#'*60}")
    print("# Boundary Expansion — Adaptive Convergence Loop")
    print(f"# precision threshold: +-{PRECISION_THRESHOLD_PP/2}pp   "
          f"cap: {MAX_TOTAL_RUNS} runs   batch size: {BATCH_SIZE}")
    print(f"{'#'*60}\n")

    # --- Step 1: Extract the boundary subset (deeply + lightly) ---
    print("Step 1: Extracting boundary subset...")
    by_tier = load_boundary_customer_ids(Path(args.baseline_reference))
    all_boundary_ids = (
        by_tier["deeply_boundary"] + by_tier["lightly_boundary"] + by_tier["tied_no_majority"]
    )
    print(f"  deeply_boundary:  {len(by_tier['deeply_boundary'])}")
    print(f"  lightly_boundary: {len(by_tier['lightly_boundary'])}")
    print(f"  tied_no_majority: {len(by_tier['tied_no_majority'])}")
    print(f"  Combined:         {len(all_boundary_ids)}\n")

    if not all_boundary_ids:
        print("No boundary customers in either tier — nothing to expand. Done.")
        return 0

    extract_subset(
        all_boundary_ids, Path(args.baseline_input),
        Path(args.record_id_map), boundary_dir,
    )

    # --- Step 2: Adaptive convergence loop ---
    current_n = PRIMARY_RUNS
    round_num = 1

    while round_num <= SAFETY_MAX_ROUNDS:
        print(f"\n{'='*60}")
        print(f"Round {round_num} — current n = {current_n}")
        print(f"{'='*60}")

        # Dynamic batch size — shrinks on the final round so nobody
        # exceeds MAX_TOTAL_RUNS
        remaining_capacity = MAX_TOTAL_RUNS - current_n
        if remaining_capacity <= 0:
            print(f"  Already at or past the {MAX_TOTAL_RUNS}-run cap. Stopping.")
            break
        batch_size_this_round = min(BATCH_SIZE, remaining_capacity)
        print(f"  This round's batch size: {batch_size_this_round}")

        batch_input_path = decisions_dir / f"batch{round_num}_input.jsonl"
        n_records = prepare_batch_input(boundary_dir, round_num, batch_input_path)

        if n_records == 0:
            print("  No customers need another round — everyone has resolved. Stopping.")
            break

        # Launch batch_size_this_round SEPARATE agent subprocess calls,
        # each independently processing the same (unduplicated) batch
        # input file — mirroring run_baseline.sh's established pattern
        # for the primary 5-run baseline exactly. This matters because
        # even at temperature=0.0, the real Anthropic API is not
        # perfectly deterministic across separate calls (see the ground
        # truth methodology note in 1b_DESIGN.md) — repeated independent
        # subprocess launches are what actually samples that real-world
        # stochasticity. A single call against a file with duplicated
        # rows would NOT reproduce this; it would just be duplicate
        # requests within the same batch submission.
        print(f"  Running {batch_size_this_round} independent agent call(s) "
              f"against {n_records} record(s)...")
        for run_k in range(1, batch_size_this_round + 1):
            batch_output_path = decisions_dir / f"batch{round_num}_run{run_k}.decisions.jsonl"
            run_agent(batch_input_path, batch_output_path, args.model, agent_script)

        current_n += batch_size_this_round

        print(f"  Checking convergence at n={current_n}...")
        results = run_convergence_check(
            Path(args.baseline_reference), boundary_dir, decisions_dir)

        n_converged = sum(1 for r in results.values() if r["status"] == "converged")
        n_capped = sum(1 for r in results.values() if r["status"] == "did_not_converge_at_cap")
        n_needs_more = sum(1 for r in results.values() if r["status"] == "needs_more_runs")
        print(f"    Converged: {n_converged}   Needs more: {n_needs_more}   "
              f"Hit cap (unresolved): {n_capped}")

        if n_needs_more == 0:
            print("\n  All customers resolved (converged or hit cap). Stopping.")
            break

        round_num += 1
    else:
        print(f"\n  WARNING: reached SAFETY_MAX_ROUNDS ({SAFETY_MAX_ROUNDS}) without "
              f"the loop naturally terminating. This should not happen given the "
              f"batch-size schedule — investigate before trusting these results.")

    print(f"\n{'#'*60}")
    print("DONE")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Prepare Next Batch — Experiment 1b: Dithering
=================================================
The Agentic Data Contract · Pillar 1: Authoritative

Filters the full boundary_subset.jsonl down to just the customers who
still need another round of runs, per refine_boundary_convergence.py's
needs_another_batch.json. This is what actually gets passed to the agent
for each round after the first — without this filter, every batch would
re-run already-converged customers, wasting real API spend on estimates
that are already precise enough.

For batch 1 specifically, no needs_another_batch.json exists yet (nobody
could have converged with zero additional runs beyond the primary 5 —
every possible n=5 outcome has a Wilson interval far wider than the
convergence threshold, see refine_boundary_convergence.py). So batch 1
always uses the full subset, unfiltered.

Both the CLI entry point and prepare_batch_input() (the importable
function) are provided — run_boundary_expansion.py calls the function
directly rather than shelling out, since both scripts are Python.

Usage:
    python prepare_next_batch.py \
        --boundary_dir experiments_output/baseline/boundary_expansion \
        --batch_number 2 \
        --out experiments_output/baseline/boundary_expansion/decisions/batch2_input.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional


def prepare_batch_input(
    boundary_dir: Path,
    batch_number: int,
    output_path: Path,
) -> int:
    """
    Build the agent input file for the given batch number.

    batch_number == 1: use the full boundary_subset.jsonl, unfiltered.
    batch_number > 1:  filter down to only customer_ids listed in
                        needs_another_batch.json from the previous round.

    Returns the number of records written. A return value of 0 means
    everyone has converged — the caller should stop the loop rather
    than run an empty batch.
    """
    subset_path = boundary_dir / "boundary_subset.jsonl"
    record_map_path = boundary_dir / "boundary_record_id_map.json"
    needs_batch_path = boundary_dir / "needs_another_batch.json"

    with open(record_map_path) as f:
        record_to_customer = json.load(f)

    if batch_number == 1:
        target_customer_ids = None  # signals "everyone" — no filtering
    else:
        if not needs_batch_path.exists():
            # No file means the previous round found nobody left needing
            # runs — everyone converged or hit the cap. Nothing to do.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("")
            return 0
        with open(needs_batch_path) as f:
            target_customer_ids = set(json.load(f))
        if not target_customer_ids:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("")
            return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(subset_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            rid = record.get("record_id")
            cid = record_to_customer.get(rid)

            if target_customer_ids is None or cid in target_customer_ids:
                outfile.write(json.dumps(record, default=str) + "\n")
                n_written += 1

    return n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boundary_dir", type=str, required=True)
    parser.add_argument("--batch_number", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Preparing Batch {args.batch_number} Input")
    print(f"{'='*60}\n")

    n = prepare_batch_input(
        Path(args.boundary_dir), args.batch_number, Path(args.out))

    if n == 0:
        print(f"  No customers need another batch — nothing to run.")
    else:
        print(f"  Wrote {n} record(s) to {args.out}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

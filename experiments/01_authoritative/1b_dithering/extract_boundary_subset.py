#!/usr/bin/env python3
"""
Extract Boundary Subset — Experiment 1b: Dithering
======================================================
The Agentic Data Contract · Pillar 1: Authoritative

Reads baseline_reference.json, identifies customers classified as EITHER
deeply_boundary OR lightly_boundary, and extracts their records from the
baseline agent input into a smaller subset file for expanded runs.

Why both tiers, not just deeply_boundary: at n=5, the Wilson interval
width for every possible outcome is far above our +-15pp convergence
threshold -- including 4-1 splits (58.8pp) nearly as wide as 3-2 splits
(65.2pp). The discrete count-based tier labels are a cheap triage
heuristic, not a statistically precise partition -- a customer landing
in lightly_boundary by chance may be nearly as genuinely uncertain as
one landing in deeply_boundary. stable (5-0) customers are deliberately
EXCLUDED even though their interval is also technically wide (43.4pp):
for a stable customer every observed run already agrees, so the
practical reference decision is unambiguous regardless of the abstract
statistical uncertainty in the "true" underlying rate. This mechanism
exists to resolve customers with a CONTESTED reference decision, not to
pin down everyone's exact true probability.

This subset then gets run through additional baseline-style agent calls
(see run_boundary_expansion.py) to narrow the uncertainty on these
specific customers' true decision tendency -- a diagnostic enrichment,
not a replacement for the primary 5-run majority vote used everywhere
else in the experiment.

Usage:
    python extract_boundary_subset.py \
        --baseline_reference experiments_output/baseline/baseline_reference.json \
        --baseline_input experiments_output/baseline/agent_input/baseline_customers.jsonl \
        --record_id_map experiments_output/baseline/record_id_map.json \
        --out experiments_output/baseline/boundary_expansion
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_boundary_customer_ids(baseline_reference_path: Path) -> Dict[str, List[str]]:
    """
    Identify customer_ids classified as deeply_boundary, lightly_boundary,
    or tied_no_majority.

    tied_no_majority customers are included for a different reason than
    deeply_boundary/lightly_boundary: those two get expanded because a
    real majority exists but is statistically thin at n=5. tied_no_majority
    customers get expanded because no majority exists at all at the
    primary level — expansion isn't refining an existing answer, it's
    establishing the only answer these customers will ever have. See
    aggregate_baseline.py's module docstring for the full reasoning on
    why this is a documented exception to "the primary vote never
    changes."
    """
    with open(baseline_reference_path) as f:
        baseline_ref = json.load(f)

    by_tier = {"deeply_boundary": [], "lightly_boundary": [], "tied_no_majority": []}
    for cid, entry in baseline_ref["customers"].items():
        stability = entry["stability"]
        if stability in by_tier:
            by_tier[stability].append(cid)

    return by_tier


def extract_subset(
    boundary_ids: List[str],
    baseline_input_path: Path,
    record_id_map_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Extract just the boundary customers' (deeply + lightly) records from
    the baseline agent input file. Returns the record_id -> customer_id
    map restricted to this subset (needed downstream for recombining
    results).
    """
    with open(record_id_map_path) as f:
        full_map = json.load(f)

    customer_to_record = {v: k for k, v in full_map.items()}
    boundary_record_ids = {
        customer_to_record[cid] for cid in boundary_ids
        if cid in customer_to_record
    }

    missing = set(boundary_ids) - set(customer_to_record.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} boundary customer_id(s) not found in "
            f"record_id_map.json: {list(missing)[:5]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_records = []

    with open(baseline_input_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_id") in boundary_record_ids:
                subset_records.append(record)

    if len(subset_records) != len(boundary_record_ids):
        raise ValueError(
            f"Expected {len(boundary_record_ids)} boundary records but "
            f"found {len(subset_records)} in baseline input. Check that "
            f"baseline_input_path matches the same run that produced "
            f"record_id_map.json."
        )

    subset_path = output_dir / "boundary_subset.jsonl"
    with open(subset_path, "w") as f:
        for record in subset_records:
            f.write(json.dumps(record, default=str) + "\n")

    subset_map = {
        rid: cid for rid, cid in full_map.items()
        if rid in boundary_record_ids
    }
    map_path = output_dir / "boundary_record_id_map.json"
    with open(map_path, "w") as f:
        json.dump(subset_map, f, indent=2)

    print(f"  Extracted {len(subset_records)} matching records")
    print(f"  Saved: {subset_path}")
    print(f"  Saved: {map_path}")

    return subset_map


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_reference", type=str, required=True)
    parser.add_argument("--baseline_input", type=str, required=True)
    parser.add_argument("--record_id_map", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Extracting Boundary Subset (deeply + lightly + tied_no_majority)")
    print(f"{'='*60}\n")

    by_tier = load_boundary_customer_ids(Path(args.baseline_reference))
    all_boundary_ids = (
        by_tier["deeply_boundary"] + by_tier["lightly_boundary"] + by_tier["tied_no_majority"]
    )

    print(f"  deeply_boundary:  {len(by_tier['deeply_boundary'])} customers")
    print(f"  lightly_boundary: {len(by_tier['lightly_boundary'])} customers")
    print(f"  tied_no_majority: {len(by_tier['tied_no_majority'])} customers")
    print(f"  Combined total:   {len(all_boundary_ids)} customers\n")
    print(f"  Combined total:   {len(all_boundary_ids)} customers\n")

    if not all_boundary_ids:
        print("  No boundary customers found in either tier. Nothing to extract.")
        print("  (This is a valid outcome — it means the primary 5-run")
        print("  baseline produced only 5-0 splits, i.e. every customer")
        print("  was fully stable.)")
        return 0

    extract_subset(
        all_boundary_ids,
        Path(args.baseline_input),
        Path(args.record_id_map),
        Path(args.out),
    )

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

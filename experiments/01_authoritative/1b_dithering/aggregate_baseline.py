#!/usr/bin/env python3
"""
Aggregate Baseline — Experiment 1b: Dithering
================================================
The Agentic Data Contract · Pillar 1: Authoritative

Combines 5 independent baseline runs (same clean data, temperature 0.0)
into a single baseline reference file. For each customer:

  - majority_decision:     the most common decision across 5 runs
  - decision_distribution: exact counts (e.g. {"HIGH_PRIORITY": 5})
  - stability:             "stable" (5/5), "lightly_boundary" (4/1),
                            or "deeply_boundary" (3/2)
  - avg_confidence:        mean agent_confidence across the 5 runs
  - confidence_range:      (min, max) confidence across the 5 runs

This baseline reference is the ground truth decision every dither
condition is compared against. It never reaches the agent — it exists
exclusively for the evaluation layer (H6: boundary customer vulnerability).

Why 5 runs at temperature 0.0 rather than one run:
Even at temperature 0.0, the LLM can produce non-deterministic outputs
for effectively identical inputs — this was observed directly in 1a
(the Mario Cohen example, CUST_000520). A single baseline run would
conflate genuine model stochasticity with the effects we're trying to
measure in 1b. Five runs and a majority vote separates "this customer's
decision is inherently unstable" from "this customer's decision changed
because of dithered data."

Usage:
    python aggregate_baseline.py \
        --decisions_dir experiments_output/baseline/decisions \
        --record_id_map experiments_output/baseline/record_id_map.json \
        --n_runs 5 \
        --output experiments_output/baseline/baseline_reference.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_decisions(decisions_dir: Path, n_runs: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all N run files and group decisions by record_id.

    Returns: {record_id: [decision_run1, decision_run2, ..., decision_run5]}
    """
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for run_num in range(1, n_runs + 1):
        run_path = decisions_dir / f"run{run_num}.decisions.jsonl"
        if not run_path.exists():
            raise FileNotFoundError(
                f"Missing baseline run file: {run_path}. "
                f"Expected {n_runs} run files (run1 through run{n_runs})."
            )

        with open(run_path) as f:
            for line in f:
                if not line.strip():
                    continue
                decision = json.loads(line)
                by_record[decision["record_id"]].append(decision)

    return by_record


def classify_stability(decisions: List[str]) -> str:
    """
    Classify baseline stability based on the distribution of decisions
    across 5 runs.

    stable:           5/5 — all runs agree
    lightly_boundary: 4/1 — one dissenting run
    deeply_boundary:  3/2 — near-even split (most ambiguous)

    A 5-run distribution can only ever be 5/0, 4/1, or 3/2 (or a 3-way
    split like 3/1/1, 2/2/1) given 3 possible decision categories.
    We fold any split more fragmented than 4/1 into deeply_boundary
    since the interpretation — "no clear majority" — is the same.
    """
    counts = Counter(decisions)
    top_count = counts.most_common(1)[0][1]

    if top_count == 5:
        return "stable"
    elif top_count == 4:
        return "lightly_boundary"
    else:
        # Covers 3/2, 3/1/1, 2/2/1 — all represent meaningful disagreement
        return "deeply_boundary"


def aggregate_customer(
    record_id: str,
    customer_id: str,
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate 5 runs for a single customer into a baseline reference entry.
    """
    decisions = [r["business_decision"] for r in runs]
    confidences = [r["agent_confidence"] for r in runs
                   if r.get("agent_confidence") is not None]

    decision_counts = dict(Counter(decisions))
    majority_decision = Counter(decisions).most_common(1)[0][0]
    stability = classify_stability(decisions)

    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    confidence_range = (
        [round(min(confidences), 4), round(max(confidences), 4)]
        if confidences else None
    )

    # Preserve all 5 raw decisions for full transparency / re-analysis
    run_details = [
        {
            "business_decision": r["business_decision"],
            "agent_confidence":  r.get("agent_confidence"),
            "decision_reasoning": r.get("decision_reasoning"),
            "key_factors":       r.get("key_factors", []),
        }
        for r in runs
    ]

    return {
        "record_id":            record_id,
        "customer_id":          customer_id,
        "n_runs":               len(runs),
        "majority_decision":    majority_decision,
        "decision_distribution": decision_counts,
        "stability":            stability,
        "avg_confidence":       avg_confidence,
        "confidence_range":     confidence_range,
        "run_details":          run_details,
    }


def run_aggregation(
    decisions_dir: Path,
    record_id_map_path: Path,
    n_runs: int,
) -> Dict[str, Any]:
    """
    Full aggregation pipeline. Returns the complete baseline reference dict
    keyed by customer_id, plus a summary of stability distribution.
    """
    print(f"Loading {n_runs} baseline run files from {decisions_dir}...")
    by_record = load_decisions(decisions_dir, n_runs)
    print(f"  Loaded decisions for {len(by_record)} unique record_ids")

    print(f"\nLoading record_id -> customer_id map...")
    with open(record_id_map_path) as f:
        record_to_customer = json.load(f)
    print(f"  Loaded {len(record_to_customer)} mappings")

    # Sanity check: every record_id in decisions should have exactly n_runs entries
    incomplete = [rid for rid, runs in by_record.items() if len(runs) != n_runs]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} record_id(s) do not have exactly {n_runs} runs. "
            f"First few: {incomplete[:5]}. "
            f"This likely means one or more baseline runs failed partway through."
        )

    # Sanity check: every record_id in decisions should map to a known customer
    unmapped = [rid for rid in by_record if rid not in record_to_customer]
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} record_id(s) in decisions have no entry in "
            f"record_id_map.json. First few: {unmapped[:5]}"
        )

    print(f"\nAggregating {len(by_record)} customers across {n_runs} runs each...")

    baseline_reference = {}
    for record_id, runs in by_record.items():
        customer_id = record_to_customer[record_id]
        entry = aggregate_customer(record_id, customer_id, runs)
        baseline_reference[customer_id] = entry

    # Summary stats
    stability_counts = Counter(e["stability"] for e in baseline_reference.values())
    total = len(baseline_reference)

    summary = {
        "total_customers":   total,
        "n_runs_per_customer": n_runs,
        "stability_counts":  dict(stability_counts),
        "stability_pct": {
            k: round(v / total * 100, 1) for k, v in stability_counts.items()
        },
    }

    print(f"\nStability distribution:")
    for k, v in stability_counts.items():
        pct = summary["stability_pct"][k]
        print(f"  {k:<20} {v:>5} ({pct}%)")

    return {
        "summary": summary,
        "customers": baseline_reference,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate 5 baseline runs into a single reference file"
    )
    parser.add_argument("--decisions_dir", type=str, required=True,
        help="Directory containing run1.decisions.jsonl ... runN.decisions.jsonl")
    parser.add_argument("--record_id_map", type=str, required=True,
        help="Path to record_id_map.json")
    parser.add_argument("--n_runs", type=int, default=5,
        help="Number of baseline runs (default: 5)")
    parser.add_argument("--output", type=str, required=True,
        help="Path to save baseline_reference.json")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Baseline Aggregation — Experiment 1b")
    print(f"{'='*60}\n")

    result = run_aggregation(
        decisions_dir=Path(args.decisions_dir),
        record_id_map_path=Path(args.record_id_map),
        n_runs=args.n_runs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Saved: {output_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())

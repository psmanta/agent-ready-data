#!/usr/bin/env python3
"""
Check and Generate H8b — Experiment 1b: Dithering
====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Implements H8b's locked decision rule from 1b_DESIGN_AMENDMENT_1.md.
Run this AFTER the agent has processed h8a_pair2_purchase_risk,
h1_category_purchase_behavior, and h1_category_risk_factors — it reads
their DECISIONS (not just dithered data), computes whether H8a's Pair 2
shows a super-additive interaction effect, and only if that threshold is
met does it empirically select the top-drifting fields and generate
H8b's single condition.

This script does NOT run the agent. If it generates H8b's data, that
data must then be run through business_decision_agent.py separately,
same as any other condition — see the printed instructions at the end.

The decision rule (from the amendment, locked before any data existed):
  1. Compute drift rate for h1_category_purchase_behavior (rate_A)
  2. Compute drift rate for h1_category_risk_factors (rate_B)
  3. Compute drift rate for h8a_pair2_purchase_risk (rate_combined)
  4. additive_prediction = rate_A + rate_B - (rate_A * rate_B)
  5. relative_excess = (rate_combined - additive_prediction) / additive_prediction
  6. If relative_excess > 0.20 (20% relative excess): generate H8b,
     selecting the two fields with the highest individual drift rate
     from Purchase Behavior and Risk Factors respectively (using H1's
     individual-field conditions and H2's field data where available).
  7. If not: report the null finding directly. No H8b condition needed.

Ground truth customers are loaded directly from the already-saved
canonical_customers.json (produced by generate_dithered_data.py's Step 1)
rather than regenerated from n/seed — this guarantees H8b, if generated,
dithers the exact same 1,000 customers as every other condition, with
zero risk of an n/seed mismatch from re-running the generator separately.

Usage:
    python check_and_generate_h8b.py \
        --decisions_dir experiments_output/agent_results/decisions \
        --baseline_reference experiments_output/baseline/baseline_reference.json \
        --out experiments_output \
        --threshold 0.20
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Resolve shared directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
shared_dir = project_root / "shared"
sys.path.insert(0, str(shared_dir / "data_generation"))

from dither_engine import DitherEngine, build_h8b_condition

# Fields whose individual drift rate is checked when selecting H8b's
# empirical field pair, per category. These are the fields with existing
# individual-condition data (from H1's individual conditions and H2's
# magnitude ladder) — not every field in the category has individual
# data, only these do.
PURCHASE_BEHAVIOR_INDIVIDUAL_FIELDS = {
    "last_purchase_days_ago":  "h1_individual_last_purchase_days_ago",
    "lifetime_value_estimate": "h1_individual_lifetime_value_estimate",
    "total_spend":             "h2_total_spend_mag15pct",
}
RISK_FACTORS_INDIVIDUAL_FIELDS = {
    "churn_risk_score": "h2_churn_risk_score_mag15pct",
}

ADDITIVE_EXCESS_THRESHOLD = 0.20  # 20% relative excess, per the amendment


# ============================================================================
# HELPERS — small, duplicated deliberately rather than imported from
# generate_dithered_data.py, which is written to run as __main__ and not
# designed as an importable module. Kept minimal and self-contained.
# ============================================================================

INTERNAL_ONLY_FIELDS = {"customer_id"}


def strip_internal_fields(customer: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in customer.items()
            if k not in INTERNAL_ONLY_FIELDS and not k.startswith("_dither")}


def assign_record_ids(customers: List[Dict[str, Any]], prefix: str = "H8B") -> List[Dict[str, Any]]:
    result = []
    for c in customers:
        c = dict(c)
        h = hashlib.md5(c["customer_id"].encode()).hexdigest()[:12].upper()
        c["record_id"] = f"{prefix}_{h}"
        result.append(c)
    return result


def verify_uniqueness(records: List[Dict[str, Any]], context: str, id_field: str = "record_id") -> None:
    ids = [r.get(id_field) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {id_field} detected in {context} — investigate before proceeding.")


def load_decisions(decisions_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a condition's decisions, keyed by record_id."""
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Decisions file not found: {decisions_path}. "
            f"H8a's conditions must be run through the agent before this "
            f"script can check the decision rule."
        )
    decisions = {}
    with open(decisions_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            decisions[d["record_id"]] = d
    return decisions


def compute_drift_rate(
    condition_decisions: Dict[str, Dict[str, Any]],
    baseline_reference: Dict[str, Any],
    dither_reference_path: Path,
) -> float:
    """
    Compute binary drift rate for a condition: the fraction of customers
    whose decision differs from their baseline majority decision.
    """
    with open(dither_reference_path) as f:
        dither_ref = json.load(f)

    record_to_customer = {r["record_id"]: r["customer_id"] for r in dither_ref}
    baseline_customers = baseline_reference["customers"]

    total = 0
    drifted = 0
    for record_id, decision in condition_decisions.items():
        customer_id = record_to_customer.get(record_id)
        if customer_id is None or customer_id not in baseline_customers:
            continue
        total += 1
        baseline_decision = baseline_customers[customer_id]["majority_decision"]
        if decision["business_decision"] != baseline_decision:
            drifted += 1

    if total == 0:
        raise ValueError("No matching customers found — check dither_reference alignment")

    return drifted / total


def select_top_field(
    candidate_fields: Dict[str, str],
    decisions_dir: Path,
    baseline_reference: Dict[str, Any],
    output_dir: Path,
) -> Tuple[str, float]:
    """
    Given {field_name: condition_id_with_individual_data}, compute each
    field's individual drift rate and return the highest-drifting field.
    """
    rates = {}
    for field_name, condition_id in candidate_fields.items():
        decisions_path = decisions_dir / f"{condition_id}.decisions.jsonl"
        dither_ref_path = output_dir / "conditions" / condition_id / "dither_reference.json"
        if not decisions_path.exists():
            print(f"    WARNING: no decisions found for {condition_id} — skipping {field_name}")
            continue
        rate = compute_drift_rate(
            load_decisions(decisions_path), baseline_reference, dither_ref_path)
        rates[field_name] = rate
        print(f"    {field_name:<30} (via {condition_id}): drift rate = {rate:.4f}")

    if not rates:
        raise ValueError("No individual field drift rates could be computed")

    top_field = max(rates, key=rates.get)
    return top_field, rates[top_field]


def generate_h8b_data(
    field_a: str,
    field_b: str,
    output_dir: Path,
    seed: int,
) -> None:
    """
    Generate H8b's agent input and dither reference files using the
    empirically-selected field pair. Ground truth customers are loaded
    directly from the already-saved canonical_customers.json.
    """
    gt_path = output_dir / "ground_truth" / "canonical_customers.json"
    print(f"\nLoading ground truth from {gt_path}...")
    with open(gt_path) as f:
        customers = json.load(f)
    print(f"  Loaded {len(customers)} customers")

    config = build_h8b_condition(field_a=field_a, field_b=field_b, seed=seed)
    print(f"\nGenerating condition: {config.condition_id}")

    engine = DitherEngine(config)
    dithered = engine.apply(customers)

    with_record_ids = assign_record_ids(dithered, prefix="H8B")
    verify_uniqueness(with_record_ids, context=f"{config.condition_id} agent input")

    cond_dir = output_dir / "conditions" / config.condition_id
    cond_dir.mkdir(parents=True, exist_ok=True)

    agent_facing = [strip_internal_fields(c) for c in with_record_ids]
    agent_input_path = cond_dir / "agent_input.jsonl"
    with open(agent_input_path, "w") as f:
        for record in agent_facing:
            f.write(json.dumps(record, default=str) + "\n")

    reference_path = cond_dir / "dither_reference.json"
    with open(reference_path, "w") as f:
        json.dump(with_record_ids, f, indent=2, default=str)

    n_dithered = sum(1 for c in dithered if c.get("_dither_applied"))
    print(f"  Saved: {agent_input_path} ({len(agent_facing)} records)")
    print(f"  Saved: {reference_path} ({n_dithered} dithered)")
    print(f"\n  NEXT STEP: run business_decision_agent.py against")
    print(f"  {agent_input_path}")
    print(f"  to produce decisions for {config.condition_id} before it")
    print(f"  can be included in the final evaluation.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions_dir", type=str, required=True,
        help="Directory containing all conditions' *.decisions.jsonl files")
    parser.add_argument("--baseline_reference", type=str, required=True,
        help="Path to baseline_reference.json")
    parser.add_argument("--out", type=str, default="experiments_output",
        help="Output directory (same as used by generate_dithered_data.py)")
    parser.add_argument("--threshold", type=float, default=ADDITIVE_EXCESS_THRESHOLD,
        help=f"Relative excess threshold for triggering H8b (default: {ADDITIVE_EXCESS_THRESHOLD})")
    parser.add_argument("--seed", type=int, default=999,
        help="Seed for H8b's dither generation, if triggered")

    args = parser.parse_args()
    decisions_dir = Path(args.decisions_dir)
    output_dir = Path(args.out)

    print(f"\n{'='*60}")
    print("H8b Decision Rule — Additive Baseline Check")
    print(f"{'='*60}\n")

    with open(args.baseline_reference) as f:
        baseline_reference = json.load(f)

    print("Computing drift rates for H8a's Pair 2 components...")

    rate_a = compute_drift_rate(
        load_decisions(decisions_dir / "h1_category_purchase_behavior.decisions.jsonl"),
        baseline_reference,
        output_dir / "conditions" / "h1_category_purchase_behavior" / "dither_reference.json")
    print(f"  h1_category_purchase_behavior (rate_A):  {rate_a:.4f}")

    rate_b = compute_drift_rate(
        load_decisions(decisions_dir / "h1_category_risk_factors.decisions.jsonl"),
        baseline_reference,
        output_dir / "conditions" / "h1_category_risk_factors" / "dither_reference.json")
    print(f"  h1_category_risk_factors (rate_B):       {rate_b:.4f}")

    rate_combined = compute_drift_rate(
        load_decisions(decisions_dir / "h8a_pair2_purchase_risk.decisions.jsonl"),
        baseline_reference,
        output_dir / "conditions" / "h8a_pair2_purchase_risk" / "dither_reference.json")
    print(f"  h8a_pair2_purchase_risk (rate_combined): {rate_combined:.4f}")

    additive_prediction = rate_a + rate_b - (rate_a * rate_b)
    relative_excess = (rate_combined - additive_prediction) / additive_prediction if additive_prediction > 0 else 0.0

    print(f"\n  Additive prediction (rate_A + rate_B - rate_A*rate_B): {additive_prediction:.4f}")
    print(f"  Observed combined rate:                                 {rate_combined:.4f}")
    print(f"  Relative excess:                                        {relative_excess:+.1%}")
    print(f"  Threshold:                                               {args.threshold:.1%}")

    result = {
        "rate_purchase_behavior": rate_a,
        "rate_risk_factors":      rate_b,
        "rate_combined":          rate_combined,
        "additive_prediction":    additive_prediction,
        "relative_excess":        relative_excess,
        "threshold":              args.threshold,
        "h8b_triggered":          relative_excess > args.threshold,
    }

    if relative_excess <= args.threshold:
        print(f"\n{'='*60}")
        print("RESULT: No meaningful excess detected.")
        print("H8b is NOT generated. This finding is complete and reportable")
        print("on its own: no category-level interaction detected between")
        print("Purchase Behavior and Risk Factors, therefore no field-level")
        print("tracing was necessary.")
        print(f"{'='*60}\n")

        result_path = output_dir / "h8b_decision_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Decision result saved: {result_path}")
        return 0

    print(f"\n{'='*60}")
    print("RESULT: Meaningful excess detected — generating H8b.")
    print(f"{'='*60}\n")

    print("Selecting empirically top-drifting field from Purchase Behavior...")
    field_a, rate_field_a = select_top_field(
        PURCHASE_BEHAVIOR_INDIVIDUAL_FIELDS, decisions_dir, baseline_reference, output_dir)
    print(f"  -> Selected: {field_a} (drift rate {rate_field_a:.4f})\n")

    print("Selecting empirically top-drifting field from Risk Factors...")
    field_b, rate_field_b = select_top_field(
        RISK_FACTORS_INDIVIDUAL_FIELDS, decisions_dir, baseline_reference, output_dir)
    print(f"  -> Selected: {field_b} (drift rate {rate_field_b:.4f})\n")

    result["h8b_field_a"] = field_a
    result["h8b_field_b"] = field_b

    generate_h8b_data(field_a, field_b, output_dir, seed=args.seed)

    result_path = output_dir / "h8b_decision_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDecision result saved: {result_path}")

    return 0


if __name__ == "__main__":
    exit(main())

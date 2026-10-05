#!/usr/bin/env python3
"""
Generic extraction step for the blind reasoning-pattern classifier.
Decoupled from any specific hypothesis's file layout on purpose -- the
classifier itself should never know it's "for H4" or "for H5"; this is
the one place that translates a hypothesis's real condition data into
the classifier's generic {id, field_label, reasoning_text} input
contract. Reusable for H1-H4, H7, H8 alike, not H4-specific.

If a record had more than one field dithered simultaneously, emits one
row per dithered field -- each row is independently classified.

Usage:
  python3 extract_for_classifier.py \
      --decisions experiments_output/conditions/h4_churn_risk_score_implausible/decisions.jsonl \
      --dither_ref experiments_output/conditions/h4_churn_risk_score_implausible/dither_reference.json \
      --output classifier_input/h4_churn_risk_score_implausible.jsonl
"""
import argparse, json
from pathlib import Path

# Natural-language labels for the classifier prompt's field_label
# parameter -- known, confirmed fields from this project's work so far.
# Extend this dict as other hypotheses' fields get run through the
# classifier; a field missing here raises loudly rather than silently
# guessing a label.
FIELD_LABELS = {
    "churn_risk_score": "churn risk score",
    "total_spend": "total spend",
    "tenure_months": "tenure in months",
    "nps_score": "NPS score",
    "lifetime_value_estimate": "lifetime value estimate",
    "avg_resolution_time_hours": "average support resolution time",
    "refund_rate": "refund rate",
    "support_tickets_open": "number of open support tickets",
    "payment_failures": "number of payment failures",
    "last_purchase_days_ago": "days since last purchase",
    "email_open_rate": "email open rate",
    "is_vip": "VIP status",
    "has_active_subscription": "active subscription status",
    "has_pending_order": "pending order status",
    "acquisition_channel": "acquisition channel",
}


def extract(decisions_path: Path, dither_ref_path: Path, output_path: Path) -> int:
    with open(dither_ref_path) as f:
        dither_ref = {r["customer_id"]: r for r in json.load(f)}
    with open(decisions_path) as f:
        decisions = {json.loads(l)["record_id"]: json.loads(l) for l in f}

    rows = []
    for customer_id, ref in dither_ref.items():
        if not ref.get("_dither_applied"):
            continue
        decision = decisions.get(ref["record_id"])
        if decision is None or not decision.get("decision_reasoning"):
            continue
        for field in ref.get("_dither_fields", []):
            if field not in FIELD_LABELS:
                raise ValueError(
                    f"Field '{field}' has no entry in FIELD_LABELS -- add one "
                    f"rather than guessing a natural-language label silently.")
            rows.append({
                "id": f"{customer_id}:{field}",
                "field_label": FIELD_LABELS[field],
                "reasoning_text": decision["decision_reasoning"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", type=str, required=True)
    parser.add_argument("--dither_ref", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    n = extract(Path(args.decisions), Path(args.dither_ref), Path(args.output))
    print(f"Wrote {n} classifier input rows to {args.output}")


if __name__ == "__main__":
    main()

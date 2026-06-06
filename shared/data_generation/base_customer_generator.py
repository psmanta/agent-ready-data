"""
Base Customer Generator — Shared Infrastructure
================================================
The Agentic Data Contract · Phase 1 · Shared Data Generation Layer

Generates a canonical, ground-truth-quality synthetic customer dataset
for use across all but 1 Phase 1 experiments. Experiments extend this
base via its own transformation layer (ditherer, etc.).

NOTE: This architecture was refactored after the 1a experiment was completed.
There is a specific data generator for that experiment only. The output of
the 1a data generator has been validated against the output of this new
reefactored data generator. This is fully documented in RESEARCH_NOTES.md

Data Quality Dimensions Enforced
---------------------------------
- Completeness:   No null values in any agent-visible field
- Consistency:    Internal correlations hold as hard rules
                  (e.g. is_at_risk derived from churn_risk_score,
                   recently_contacted_support derived from open tickets,
                   last_login bounded by last_purchase)
- Validity:       All values within defined plausible ranges;
                  formats valid (email, phone, date)
- Accuracy:       Reduces to internal consistency + distributional
                  realism for synthetic data (no external oracle)
- Uniqueness:     No accidental duplicate customer_ids
- Timeliness:     All dates relative to a single coherent snapshot date;
                  no future last_purchase_dates, no created_date after
                  first_purchase

Architecture
------------
This module is the shared base. Experiment-specific transformation
layers (1a duplicator, 1b ditherer, etc.) import and call
generate_base_customers() then apply their own logic.

Usage (direct):
    from base_customer_generator import generate_base_customers, FIELD_GROUPS
    customers = generate_base_customers(n=1000, seed=42)

Usage (CLI — for validation and distribution comparison):
    python base_customer_generator.py --n 1000 --out /tmp/base_output --seed 42

Methodology Notes
-----------------
- Fixed seed produces deterministic output for reproducibility.
- Strict consistency enforcement creates a cleaner signal-to-noise
  baseline for downstream dither experiments. Real enterprise data
  is noisier — this is acknowledged as a known limitation.
- Customer segments drive correlated field distributions, not
  independent sampling. This produces realistic co-variation
  (high-value customers have lower churn, higher NPS, etc.)
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
from faker import Faker

fake = Faker()

# ============================================================================
# CONFIGURATION — shared constants, importable by experiment extensions
# ============================================================================

# Snapshot date: the "as of" moment for all generated records.
# All dates are computed relative to this anchor so the dataset
# is internally coherent (no future purchases, no login before account).
SNAPSHOT_DATE = datetime(2025, 1, 1)

# Customer segment distribution
CUSTOMER_SEGMENTS: Dict[str, float] = {
    "high_value":   0.15,
    "medium_value": 0.50,
    "low_value":    0.25,
    "at_risk":      0.10,
}

# Acquisition channels
ACQUISITION_CHANNELS: List[str] = [
    "organic", "referral", "paid_search", "social", "email", "partner"
]

# Product categories
PRODUCT_CATEGORIES: List[str] = [
    "electronics", "home", "kitchen", "apparel", "outdoors",
    "beauty", "office", "food", "sports", "toys"
]

# Field groups — used by experiment extensions to target specific field types.
# These are properties of the data schema, not of any specific experiment.
FIELD_GROUPS: Dict[str, List[str]] = {
    "identity":      ["name", "email", "phone", "address", "dob"],
    "tier_critical": ["total_spend", "lifetime_value_estimate",
                      "avg_order_value", "total_purchases"],
    "engagement":    ["nps_score", "email_open_rate",
                      "last_login_days_ago", "last_purchase_days_ago"],
    "risk":          ["churn_risk_score", "payment_failures", "fraud_risk_score"],
    "flags":         ["is_vip", "is_at_risk", "has_active_subscription"],
    "support":       ["support_tickets_open", "support_tickets_closed",
                      "avg_resolution_time_hours", "recently_contacted_support"],
}

# Consistency thresholds — hard rules enforced during generation.
# These are documented here rather than buried in generation logic
# so they're visible to anyone reading the codebase.
CONSISTENCY_RULES = {
    # is_at_risk flag is derived from churn_risk_score, not set independently
    "at_risk_churn_threshold":      0.60,
    # recently_contacted_support is derived from support_tickets_open
    "support_contact_ticket_threshold": 1,
    # last_login cannot predate last_purchase by more than this many days
    # (a customer who bought recently almost certainly logged in)
    "max_login_purchase_gap_days":  30,
    # LTV multiplier range per segment — segments with better retention
    # have higher LTV multiples relative to current spend
    "ltv_multiplier": {
        "high_value":   (2.5, 4.0),
        "medium_value": (1.5, 2.5),
        "low_value":    (1.0, 1.5),
        "at_risk":      (0.8, 1.2),
    },
}


# ============================================================================
# SEGMENT-DRIVEN FIELD GENERATION
# ============================================================================

def _generate_segment_fields(segment: str, rng: random.Random) -> Dict[str, Any]:
    """
    Generate segment-specific fields with correlated distributions.

    All numeric ranges are driven by segment to produce realistic
    co-variation. High-value customers have low churn, high NPS,
    high spend. At-risk customers have high churn, low NPS, more
    support issues. Ranges do not overlap between segments except
    at their edges — this is intentional for boundary customer testing.

    Returns a dict of raw field values (before derived fields are computed).
    """
    if segment == "high_value":
        total_purchases   = rng.randint(30, 100)
        avg_order_value   = rng.uniform(200, 800)
        nps_score         = rng.randint(7, 10)
        churn_risk_score  = rng.uniform(0.0, 0.25)
        email_open_rate   = rng.uniform(0.60, 0.95)
        support_open      = rng.randint(0, 1)
        payment_failures  = 0
        fraud_risk_score  = rng.uniform(0.0, 0.08)

    elif segment == "medium_value":
        total_purchases   = rng.randint(10, 35)
        avg_order_value   = rng.uniform(50, 250)
        nps_score         = rng.randint(5, 8)
        churn_risk_score  = rng.uniform(0.20, 0.55)
        email_open_rate   = rng.uniform(0.35, 0.70)
        support_open      = rng.randint(0, 2)
        payment_failures  = rng.randint(0, 1)
        fraud_risk_score  = rng.uniform(0.0, 0.15)

    elif segment == "low_value":
        total_purchases   = rng.randint(1, 12)
        avg_order_value   = rng.uniform(10, 80)
        nps_score         = rng.randint(3, 6)
        churn_risk_score  = rng.uniform(0.35, 0.62)
        email_open_rate   = rng.uniform(0.15, 0.45)
        support_open      = rng.randint(0, 3)
        payment_failures  = rng.randint(0, 2)
        fraud_risk_score  = rng.uniform(0.0, 0.25)

    else:  # at_risk — strong, unambiguous signal by design
        total_purchases   = rng.randint(5, 25)
        avg_order_value   = rng.uniform(30, 150)
        nps_score         = rng.randint(1, 4)
        churn_risk_score  = rng.uniform(0.65, 0.95)
        email_open_rate   = rng.uniform(0.03, 0.25)
        support_open      = rng.randint(2, 6)
        payment_failures  = rng.randint(1, 4)
        fraud_risk_score  = rng.uniform(0.10, 0.40)

    return {
        "total_purchases":  total_purchases,
        "avg_order_value":  avg_order_value,
        "nps_score":        nps_score,
        "churn_risk_score": churn_risk_score,
        "email_open_rate":  email_open_rate,
        "support_open":     support_open,
        "payment_failures": payment_failures,
        "fraud_risk_score": fraud_risk_score,
    }


# ============================================================================
# CORE CUSTOMER GENERATION
# ============================================================================

def generate_base_customer(
    customer_id: str,
    seed: int,
    faker_instance: Optional[Faker] = None,
) -> Dict[str, Any]:
    """
    Generate a single ground-truth-quality customer record.

    Consistency guarantees enforced here:
    1. is_at_risk is DERIVED from churn_risk_score — never set independently
    2. recently_contacted_support is DERIVED from support_tickets_open
    3. last_login_days_ago is BOUNDED so it cannot exceed last_purchase_days_ago
       by more than CONSISTENCY_RULES['max_login_purchase_gap_days']
    4. lifetime_value_estimate uses segment-specific multiplier ranges
       (high-value customers have higher LTV multiples than at-risk)
    5. account_created_date is always before last_purchase_date
    6. support_tickets_closed >= support_tickets_open (can't have more open
       than total tickets)
    7. All derived boolean flags are computed, not randomly assigned
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    fk = faker_instance or fake

    # Seed faker for reproducibility
    Faker.seed(seed)

    # --- Segment ---
    rand = rng.random()
    cumulative = 0.0
    segment = "medium_value"
    for seg, prob in CUSTOMER_SEGMENTS.items():
        cumulative += prob
        if rand <= cumulative:
            segment = seg
            break

    # --- Identity fields ---
    name  = fk.name()
    email = name.lower().replace(" ", ".").replace("'", "") + "@example.com"
    phone = fk.phone_number()
    dob   = fk.date_of_birth(minimum_age=18, maximum_age=75).isoformat()

    street = f"{rng.randint(1, 9999)} {fk.street_name()}"
    city   = fk.city()
    state  = fk.state_abbr()
    zipcode = fk.zipcode()
    address = f"{street}, {city}, {state} {zipcode}"

    # --- Segment-driven behavioral fields ---
    sf = _generate_segment_fields(segment, rng)

    total_purchases  = sf["total_purchases"]
    avg_order_value  = sf["avg_order_value"]
    nps_score        = sf["nps_score"]
    churn_risk_score = sf["churn_risk_score"]
    email_open_rate  = sf["email_open_rate"]
    support_open     = sf["support_open"]
    payment_failures = sf["payment_failures"]
    fraud_risk_score = sf["fraud_risk_score"]

    # --- Derived spend fields ---
    total_spend = round(
        total_purchases * avg_order_value * rng.uniform(0.90, 1.10), 2
    )

    # Consistency rule 4: LTV multiplier is segment-specific
    ltv_lo, ltv_hi = CONSISTENCY_RULES["ltv_multiplier"][segment]
    lifetime_value_estimate = round(total_spend * rng.uniform(ltv_lo, ltv_hi), 2)

    # Purchase frequency: average days between orders
    purchase_frequency_days = max(
        1, int(365 / max(1, total_purchases / rng.uniform(1.0, 3.0)))
    )

    # --- Temporal fields ---
    tenure_months = rng.randint(3, 72)  # minimum 3 months for realistic history

    # last_purchase cannot exceed tenure
    max_purchase_days = min(365, tenure_months * 30)
    last_purchase_days_ago = rng.randint(1, max_purchase_days)

    # Consistency rule 3: last_login bounded relative to last_purchase
    # A customer who purchased recently almost certainly logged in.
    # login cannot be MORE than max_login_purchase_gap_days after purchase.
    max_gap = CONSISTENCY_RULES["max_login_purchase_gap_days"]
    login_upper = min(last_purchase_days_ago + max_gap, tenure_months * 30)
    last_login_days_ago = rng.randint(
        max(0, last_purchase_days_ago - 7),  # can log in after purchase
        login_upper
    )

    # --- Support fields ---
    support_tickets_open   = support_open
    # Consistency rule 6: closed >= open
    support_tickets_closed = rng.randint(support_open, support_open + 8)
    avg_resolution_time_hours = rng.randint(2, 72)

    # --- Date fields (relative to SNAPSHOT_DATE) ---
    account_created_date = (
        SNAPSHOT_DATE - timedelta(days=tenure_months * 30)
    ).date().isoformat()

    last_purchase_date = (
        SNAPSHOT_DATE - timedelta(days=last_purchase_days_ago)
    ).date().isoformat()

    # next_renewal is always in the future relative to snapshot
    next_renewal_date = (
        SNAPSHOT_DATE + timedelta(days=rng.randint(30, 365))
    ).date().isoformat()

    # --- Other fields ---
    refund_rate         = round(rng.uniform(0.0, 0.12), 3)
    acquisition_channel = rng.choice(ACQUISITION_CHANNELS)
    preferred_categories = rng.sample(
        PRODUCT_CATEGORIES, k=rng.randint(1, 4)
    )

    # --- Derived boolean flags (Consistency rules 1, 2, 7) ---
    # Round churn_risk_score first so is_at_risk is derived from
    # the same value that gets stored — avoids float precision edge cases
    churn_risk_score = round(churn_risk_score, 3)

    # Rule 1: is_at_risk derived from stored churn_risk_score value
    is_at_risk = churn_risk_score >= CONSISTENCY_RULES["at_risk_churn_threshold"]

    # Rule 2: recently_contacted_support derived from open tickets
    recently_contacted_support = (
        support_tickets_open >= CONSISTENCY_RULES["support_contact_ticket_threshold"]
    )

    # is_vip: only high_value segment, and only a subset of them
    is_vip = (segment == "high_value") and (rng.random() < 0.25)

    # has_active_subscription: segment-weighted probability
    sub_prob = {"high_value": 0.60, "medium_value": 0.35,
                "low_value": 0.15, "at_risk": 0.20}
    has_active_subscription = rng.random() < sub_prob[segment]

    has_pending_order = rng.random() < 0.18

    return {
        "customer_id":               customer_id,
        "name":                      name,
        "email":                     email,
        "phone":                     phone,
        "dob":                       dob,
        "address":                   address,
        "customer_segment":          segment,
        "acquisition_channel":       acquisition_channel,
        "tenure_months":             tenure_months,
        "preferred_categories":      preferred_categories,
        # spend
        "total_purchases":           total_purchases,
        "avg_order_value":           round(avg_order_value, 2),
        "total_spend":               total_spend,
        "lifetime_value_estimate":   lifetime_value_estimate,
        "purchase_frequency_days":   purchase_frequency_days,
        # temporal
        "last_purchase_days_ago":    last_purchase_days_ago,
        "last_login_days_ago":       last_login_days_ago,
        "account_created_date":      account_created_date,
        "last_purchase_date":        last_purchase_date,
        "next_renewal_date":         next_renewal_date,
        # engagement
        "nps_score":                 nps_score,
        "email_open_rate":           round(email_open_rate, 3),
        # risk
        "churn_risk_score":          churn_risk_score,  # already rounded above
        "payment_failures":          payment_failures,
        "fraud_risk_score":          round(fraud_risk_score, 3),
        "refund_rate":               refund_rate,
        # support
        "support_tickets_open":      support_tickets_open,
        "support_tickets_closed":    support_tickets_closed,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        # derived booleans — do NOT set these independently
        "is_at_risk":                is_at_risk,
        "recently_contacted_support": recently_contacted_support,
        "is_vip":                    is_vip,
        "has_active_subscription":   has_active_subscription,
        "has_pending_order":         has_pending_order,
    }


def generate_base_customers(
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate n ground-truth-quality canonical customer records.

    Fixed seed produces identical output on every run — required
    for experiment reproducibility. Different seeds produce different
    but equally valid customer populations (used for robustness checks).

    Args:
        n:    Number of customers to generate
        seed: Random seed. Document in experiment metadata.

    Returns:
        List of customer dicts. Each has a unique customer_id.
        No record_id at this stage — that's added by the experiment
        extension (duplicator adds multiple record_ids per customer;
        ditherer adds one record_id per customer).
    """
    fk = Faker()
    customers = []
    for i in range(n):
        customer_id = f"CUST_{i+1:06d}"
        customer = generate_base_customer(
            customer_id,
            seed=seed + i,
            faker_instance=fk,
        )
        customers.append(customer)
    return customers


# ============================================================================
# VALIDATION — run after generation to verify consistency rules
# ============================================================================

def validate_dataset(customers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a generated dataset against all consistency rules.
    Returns a report dict. Raises ValueError if any hard rule is violated.

    This is the code equivalent of the data quality dimension definitions
    in the module docstring. Run this after generate_base_customers() to
    confirm the dataset meets ground truth standards before use.
    """
    violations = []
    warnings = []

    seen_ids = set()
    for i, c in enumerate(customers):
        cid = c["customer_id"]

        # Uniqueness
        if cid in seen_ids:
            violations.append(f"[{cid}] Duplicate customer_id")
        seen_ids.add(cid)

        # Completeness — no None values in agent-visible fields
        for field, value in c.items():
            if value is None:
                violations.append(f"[{cid}] Null value in field '{field}'")

        # Consistency rule 1: is_at_risk must match churn_risk_score
        expected_at_risk = round(c["churn_risk_score"], 10) >= CONSISTENCY_RULES["at_risk_churn_threshold"]
        if c["is_at_risk"] != expected_at_risk:
            violations.append(
                f"[{cid}] is_at_risk={c['is_at_risk']} but "
                f"churn_risk_score={c['churn_risk_score']:.3f} "
                f"(threshold={CONSISTENCY_RULES['at_risk_churn_threshold']})"
            )

        # Consistency rule 2: recently_contacted_support must match tickets
        expected_contact = (
            c["support_tickets_open"] >=
            CONSISTENCY_RULES["support_contact_ticket_threshold"]
        )
        if c["recently_contacted_support"] != expected_contact:
            violations.append(
                f"[{cid}] recently_contacted_support={c['recently_contacted_support']} "
                f"but support_tickets_open={c['support_tickets_open']}"
            )

        # Consistency rule 3: login gap
        gap = c["last_login_days_ago"] - c["last_purchase_days_ago"]
        max_gap = CONSISTENCY_RULES["max_login_purchase_gap_days"]
        if gap > max_gap:
            violations.append(
                f"[{cid}] last_login_days_ago={c['last_login_days_ago']} is "
                f"{gap} days after last_purchase_days_ago={c['last_purchase_days_ago']} "
                f"(max allowed gap: {max_gap})"
            )

        # Consistency rule 6: closed >= open
        if c["support_tickets_closed"] < c["support_tickets_open"]:
            violations.append(
                f"[{cid}] support_tickets_closed={c['support_tickets_closed']} "
                f"< support_tickets_open={c['support_tickets_open']}"
            )

        # Validity: range checks
        if not (0 <= c["nps_score"] <= 10):
            violations.append(f"[{cid}] nps_score={c['nps_score']} out of range [0,10]")
        if not (0.0 <= c["churn_risk_score"] <= 1.0):
            violations.append(f"[{cid}] churn_risk_score out of range [0,1]")
        if not (0.0 <= c["email_open_rate"] <= 1.0):
            violations.append(f"[{cid}] email_open_rate out of range [0,1]")
        if not (0.0 <= c["fraud_risk_score"] <= 1.0):
            violations.append(f"[{cid}] fraud_risk_score out of range [0,1]")
        if c["total_spend"] < 0:
            violations.append(f"[{cid}] total_spend is negative")
        if c["lifetime_value_estimate"] < 0:
            violations.append(f"[{cid}] lifetime_value_estimate is negative")
        if c["tenure_months"] < 1:
            violations.append(f"[{cid}] tenure_months < 1")

        # Validity: LTV should be >= total_spend for non-at-risk customers
        if c["customer_segment"] != "at_risk":
            if c["lifetime_value_estimate"] < c["total_spend"]:
                warnings.append(
                    f"[{cid}] lifetime_value_estimate ({c['lifetime_value_estimate']}) "
                    f"< total_spend ({c['total_spend']}) for {c['customer_segment']} customer"
                )

    report = {
        "total_customers": len(customers),
        "violations": len(violations),
        "warnings": len(warnings),
        "violation_details": violations[:20],  # cap at 20 for readability
        "warning_details": warnings[:10],
        "passed": len(violations) == 0,
    }

    if violations:
        raise ValueError(
            f"Dataset failed consistency validation: "
            f"{len(violations)} violation(s). First: {violations[0]}"
        )

    return report


# ============================================================================
# FILE OUTPUT — shared save utilities
# ============================================================================

def save_canonical_customers(
    customers: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save canonical customers to JSON.
    The agent never sees this file — it is the ground truth reference
    used exclusively by the evaluation layer.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(customers, f, indent=2, default=str)
    print(f"  Saved canonical customers: {output_path} ({len(customers)} records)")


def compute_distribution_summary(
    customers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute summary statistics for distribution comparison.
    Used to validate that the new base generator produces
    distributions comparable to the 1a generator.
    """
    from collections import Counter

    segments = Counter(c["customer_segment"] for c in customers)
    total = len(customers)

    spend_values    = [c["total_spend"] for c in customers]
    churn_values    = [c["churn_risk_score"] for c in customers]
    nps_values      = [c["nps_score"] for c in customers]
    purchase_values = [c["last_purchase_days_ago"] for c in customers]
    ltv_values      = [c["lifetime_value_estimate"] for c in customers]

    def stats(values):
        arr = np.array(values)
        return {
            "mean":   round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "std":    round(float(arr.std()), 2),
            "min":    round(float(arr.min()), 2),
            "max":    round(float(arr.max()), 2),
            "p25":    round(float(np.percentile(arr, 25)), 2),
            "p75":    round(float(np.percentile(arr, 75)), 2),
        }

    at_risk_count = sum(1 for c in customers if c["is_at_risk"])
    vip_count     = sum(1 for c in customers if c["is_vip"])
    sub_count     = sum(1 for c in customers if c["has_active_subscription"])

    # Login/purchase gap consistency check
    gaps = [c["last_login_days_ago"] - c["last_purchase_days_ago"]
            for c in customers]
    max_gap = CONSISTENCY_RULES["max_login_purchase_gap_days"]
    gap_violations = sum(1 for g in gaps if g > max_gap)

    return {
        "total_customers":   total,
        "segment_counts":    dict(segments),
        "segment_pct":       {k: round(v/total*100, 1) for k, v in segments.items()},
        "total_spend":       stats(spend_values),
        "churn_risk_score":  stats(churn_values),
        "nps_score":         stats(nps_values),
        "last_purchase_days_ago": stats(purchase_values),
        "lifetime_value_estimate": stats(ltv_values),
        "derived_flags": {
            "is_at_risk_count":           at_risk_count,
            "is_at_risk_pct":             round(at_risk_count/total*100, 1),
            "is_vip_count":               vip_count,
            "is_vip_pct":                 round(vip_count/total*100, 1),
            "has_subscription_count":     sub_count,
            "has_subscription_pct":       round(sub_count/total*100, 1),
        },
        "consistency_checks": {
            "login_purchase_gap_violations": gap_violations,
            "login_purchase_gap_max_allowed": max_gap,
        },
    }


# ============================================================================
# CLI — for validation and standalone distribution analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth-quality base customer dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 customers with default seed
  python base_customer_generator.py --n 1000 --out /tmp/base_output

  # Generate with alternate seed (robustness check)
  python base_customer_generator.py --n 1000 --out /tmp/base_alt --seed 99

  # Validate only — generate, validate, print summary, no file output
  python base_customer_generator.py --n 1000 --validate-only
        """
    )
    parser.add_argument("--n",    type=int, default=1000,
        help="Number of customers (default: 1000)")
    parser.add_argument("--out",  type=str, default=None,
        help="Output directory (omit to skip file output)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument("--validate-only", action="store_true",
        help="Generate and validate without writing files")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Base Customer Generator")
    print(f"{'='*60}")
    print(f"Customers:  {args.n}")
    print(f"Seed:       {args.seed}")
    print(f"Snapshot:   {SNAPSHOT_DATE.date()}")
    print()

    print("Generating customers...")
    customers = generate_base_customers(n=args.n, seed=args.seed)
    print(f"  Generated {len(customers)} customers")

    print("\nValidating consistency rules...")
    try:
        report = validate_dataset(customers)
        print(f"  PASSED — {report['total_customers']} customers, "
              f"{report['violations']} violations, "
              f"{report['warnings']} warnings")
        if report["warning_details"]:
            print("  Warnings:")
            for w in report["warning_details"]:
                print(f"    {w}")
    except ValueError as e:
        print(f"  FAILED: {e}")
        return 1

    print("\nDistribution summary:")
    summary = compute_distribution_summary(customers)
    print(f"  Segments: {summary['segment_pct']}")
    print(f"  Total spend:   mean={summary['total_spend']['mean']:>10,.2f}  "
          f"median={summary['total_spend']['median']:>10,.2f}  "
          f"std={summary['total_spend']['std']:>10,.2f}")
    print(f"  Churn risk:    mean={summary['churn_risk_score']['mean']:>10.3f}  "
          f"median={summary['churn_risk_score']['median']:>10.3f}  "
          f"std={summary['churn_risk_score']['std']:>10.3f}")
    print(f"  NPS:           mean={summary['nps_score']['mean']:>10.2f}  "
          f"median={summary['nps_score']['median']:>10.2f}  "
          f"std={summary['nps_score']['std']:>10.2f}")
    print(f"  Last purchase: mean={summary['last_purchase_days_ago']['mean']:>10.1f}d  "
          f"median={summary['last_purchase_days_ago']['median']:>10.1f}d")
    print(f"  LTV:           mean={summary['lifetime_value_estimate']['mean']:>10,.2f}  "
          f"median={summary['lifetime_value_estimate']['median']:>10,.2f}")
    print(f"  Derived flags: {summary['derived_flags']}")
    print(f"  Consistency:   {summary['consistency_checks']}")

    if args.out and not args.validate_only:
        out_dir = Path(args.out)
        save_canonical_customers(customers, out_dir / "canonical_customers.json")
        summary_path = out_dir / "generation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved distribution summary: {summary_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

"""
DAMA Data Quality Dimensions Validator
=======================================
The Agentic Data Contract · Phase 1 · Shared Data Generation Layer

Validates that a generated customer dataset meets all six DAMA-canonical
data quality dimensions as cited in RESEARCH_NOTES.md:

    Accuracy, Completeness, Consistency, Timeliness, Validity, Uniqueness

Reference: DAMA UK Working Group on Data Quality Dimensions,
"The Six Primary Dimensions for Data Quality Assessment" (October 2013).

Run this before any experiment that relies on the base generator output
to confirm ground truth standards are met. Raises on any hard violation.

Usage:
    # Validate a freshly generated dataset
    python validate_dama_dimensions.py --input /path/to/canonical_customers.json

    # Generate and validate in one step
    python validate_dama_dimensions.py --generate --n 1000 --seed 42

    # Generate, validate, and save audit report
    python validate_dama_dimensions.py --generate --n 1000 --seed 42 --report audit_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# ============================================================================
# DIMENSION VALIDATORS
# ============================================================================

def check_accuracy(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Accuracy: data is internally plausible and segment-coherent.

    For synthetic data there is no external oracle to validate against,
    so accuracy reduces to two checks:
    1. No implausible individual values (negative spend, impossible scores)
    2. Segment-level plausibility: high_value customers have higher spend
       and lower churn than at_risk customers — the segments are meaningful
    """
    violations = []
    notes = []

    for c in customers:
        cid = c["customer_id"]
        if c["total_spend"] < 0:
            violations.append(f"[{cid}] Negative total_spend: {c['total_spend']}")
        if c["lifetime_value_estimate"] < 0:
            violations.append(f"[{cid}] Negative lifetime_value_estimate")
        if c["last_purchase_days_ago"] < 0:
            violations.append(f"[{cid}] Negative last_purchase_days_ago")
        if c["total_purchases"] < 1:
            violations.append(f"[{cid}] total_purchases < 1: {c['total_purchases']}")
        if c["avg_order_value"] <= 0:
            violations.append(f"[{cid}] avg_order_value <= 0: {c['avg_order_value']}")
        # LTV should be >= 80% of spend for non-at-risk customers
        if (c["customer_segment"] != "at_risk" and
                c["lifetime_value_estimate"] < c["total_spend"] * 0.8):
            violations.append(
                f"[{cid}] LTV ({c['lifetime_value_estimate']:.2f}) well below "
                f"spend ({c['total_spend']:.2f}) for {c['customer_segment']}"
            )

    # Segment plausibility check
    hv = [c for c in customers if c["customer_segment"] == "high_value"]
    ar = [c for c in customers if c["customer_segment"] == "at_risk"]
    if hv and ar:
        hv_spend = sum(c["total_spend"] for c in hv) / len(hv)
        ar_spend = sum(c["total_spend"] for c in ar) / len(ar)
        hv_churn = sum(c["churn_risk_score"] for c in hv) / len(hv)
        ar_churn = sum(c["churn_risk_score"] for c in ar) / len(ar)
        notes.append(f"high_value avg spend ${hv_spend:,.2f} vs at_risk ${ar_spend:,.2f}")
        notes.append(f"high_value avg churn {hv_churn:.3f} vs at_risk {ar_churn:.3f}")
        if hv_spend <= ar_spend:
            violations.append("Segment plausibility: high_value avg spend <= at_risk avg spend")
        if hv_churn >= ar_churn:
            violations.append("Segment plausibility: high_value avg churn >= at_risk avg churn")

    return len(violations) == 0, violations, notes


def check_completeness(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Completeness: all required fields present, no null or empty values.
    """
    violations = []
    notes = []

    for c in customers:
        cid = c["customer_id"]
        for field, value in c.items():
            if value is None:
                violations.append(f"[{cid}] Null value in field '{field}'")
            if isinstance(value, str) and value.strip() == "":
                violations.append(f"[{cid}] Empty string in field '{field}'")
        if isinstance(c.get("preferred_categories"), list) and len(c["preferred_categories"]) == 0:
            violations.append(f"[{cid}] Empty preferred_categories list")

    notes.append(f"Total fields per record: {len(customers[0]) if customers else 0}")
    notes.append(f"Total records checked: {len(customers)}")
    return len(violations) == 0, violations, notes


def check_consistency(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Consistency: internal correlations hold as hard rules.

    Cross-system consistency (same value in two systems) is not applicable
    to a single synthetic dataset. We instead enforce intra-record
    consistency — derived fields match their source fields.
    """
    violations = []
    notes = []

    AT_RISK_THRESHOLD = 0.60
    SUPPORT_CONTACT_THRESHOLD = 1
    MAX_LOGIN_PURCHASE_GAP = 30

    for c in customers:
        cid = c["customer_id"]

        # is_at_risk must match churn_risk_score
        expected_at_risk = c["churn_risk_score"] >= AT_RISK_THRESHOLD
        if c["is_at_risk"] != expected_at_risk:
            violations.append(
                f"[{cid}] is_at_risk={c['is_at_risk']} but "
                f"churn_risk_score={c['churn_risk_score']:.3f} "
                f"(threshold={AT_RISK_THRESHOLD})"
            )

        # recently_contacted_support must match open tickets
        expected_contact = c["support_tickets_open"] >= SUPPORT_CONTACT_THRESHOLD
        if c["recently_contacted_support"] != expected_contact:
            violations.append(
                f"[{cid}] recently_contacted_support={c['recently_contacted_support']} "
                f"but support_tickets_open={c['support_tickets_open']}"
            )

        # last_login cannot lag last_purchase by more than MAX_LOGIN_PURCHASE_GAP
        gap = c["last_login_days_ago"] - c["last_purchase_days_ago"]
        if gap > MAX_LOGIN_PURCHASE_GAP:
            violations.append(
                f"[{cid}] last_login is {gap}d after last_purchase "
                f"(max allowed: {MAX_LOGIN_PURCHASE_GAP}d)"
            )

        # closed tickets >= open tickets
        if c["support_tickets_closed"] < c["support_tickets_open"]:
            violations.append(
                f"[{cid}] support_tickets_closed ({c['support_tickets_closed']}) "
                f"< support_tickets_open ({c['support_tickets_open']})"
            )

        # is_vip only valid for high_value segment
        if c["is_vip"] and c["customer_segment"] != "high_value":
            violations.append(
                f"[{cid}] is_vip=True but segment={c['customer_segment']}"
            )

        # last_purchase cannot predate account creation
        if c["last_purchase_days_ago"] > c["tenure_months"] * 30:
            violations.append(
                f"[{cid}] last_purchase_days_ago ({c['last_purchase_days_ago']}) "
                f"exceeds tenure ({c['tenure_months']} months = "
                f"{c['tenure_months']*30} days)"
            )

    notes.append("Cross-system consistency N/A — single synthetic dataset")
    notes.append(f"Intra-record consistency rules checked: 6")
    return len(violations) == 0, violations, notes


def check_timeliness(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Timeliness: all dates reflect a coherent snapshot moment (2025-01-01).
    No future purchases, no account created after last purchase.
    """
    violations = []
    notes = []
    SNAPSHOT = date(2025, 1, 1)

    for c in customers:
        cid = c["customer_id"]
        try:
            last_purchase = date.fromisoformat(c["last_purchase_date"])
            created = date.fromisoformat(c["account_created_date"])
            renewal = date.fromisoformat(c["next_renewal_date"])

            if last_purchase > SNAPSHOT:
                violations.append(f"[{cid}] last_purchase_date {last_purchase} is after snapshot {SNAPSHOT}")
            if created > SNAPSHOT:
                violations.append(f"[{cid}] account_created_date {created} is after snapshot {SNAPSHOT}")
            if created > last_purchase:
                violations.append(f"[{cid}] account_created_date {created} is after last_purchase_date {last_purchase}")
            if renewal <= SNAPSHOT:
                violations.append(f"[{cid}] next_renewal_date {renewal} is not in the future")
        except (ValueError, KeyError) as e:
            violations.append(f"[{cid}] Date parse error: {e}")

    notes.append(f"Snapshot date: {SNAPSHOT}")
    notes.append("next_renewal_date must be after snapshot (all renewals are future-dated)")
    return len(violations) == 0, violations, notes


def check_validity(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Validity: values conform to defined ranges and format rules.
    """
    violations = []
    notes = []
    email_re = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

    for c in customers:
        cid = c["customer_id"]

        if not email_re.match(str(c.get("email", ""))):
            violations.append(f"[{cid}] Invalid email format: {c.get('email')}")
        if not (0 <= c["nps_score"] <= 10):
            violations.append(f"[{cid}] nps_score {c['nps_score']} out of range [0, 10]")
        if not (0.0 <= c["churn_risk_score"] <= 1.0):
            violations.append(f"[{cid}] churn_risk_score {c['churn_risk_score']} out of range [0, 1]")
        if not (0.0 <= c["email_open_rate"] <= 1.0):
            violations.append(f"[{cid}] email_open_rate {c['email_open_rate']} out of range [0, 1]")
        if not (0.0 <= c["fraud_risk_score"] <= 1.0):
            violations.append(f"[{cid}] fraud_risk_score {c['fraud_risk_score']} out of range [0, 1]")
        if not (0.0 <= c["refund_rate"] <= 1.0):
            violations.append(f"[{cid}] refund_rate {c['refund_rate']} out of range [0, 1]")
        if c["tenure_months"] < 1:
            violations.append(f"[{cid}] tenure_months {c['tenure_months']} < 1")
        if c["total_spend"] <= 0:
            violations.append(f"[{cid}] total_spend {c['total_spend']} <= 0")
        if c["total_purchases"] < 1:
            violations.append(f"[{cid}] total_purchases {c['total_purchases']} < 1")
        if c["avg_order_value"] <= 0:
            violations.append(f"[{cid}] avg_order_value {c['avg_order_value']} <= 0")
        if c["payment_failures"] < 0:
            violations.append(f"[{cid}] payment_failures {c['payment_failures']} < 0")
        if c["support_tickets_open"] < 0:
            violations.append(f"[{cid}] support_tickets_open < 0")
        if c["support_tickets_closed"] < 0:
            violations.append(f"[{cid}] support_tickets_closed < 0")
        if c["avg_resolution_time_hours"] < 0:
            violations.append(f"[{cid}] avg_resolution_time_hours < 0")

    notes.append("Email format: must contain @ with valid local and domain parts")
    notes.append("Score ranges: NPS [0,10], churn/open_rate/fraud/refund [0,1]")
    return len(violations) == 0, violations, notes


def check_uniqueness(customers: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Uniqueness: no duplicate customer_ids.

    Email addresses may collide for large datasets due to Faker generating
    names that produce the same email format (e.g. two 'John Smith' records).
    customer_id is the authoritative unique key — email collisions are noted
    as warnings, not violations.
    """
    violations = []
    notes = []

    ids = [c["customer_id"] for c in customers]
    emails = [c["email"] for c in customers]

    dup_ids = len(ids) - len(set(ids))
    dup_emails = len(emails) - len(set(emails))

    if dup_ids > 0:
        id_counts = Counter(ids)
        dups = [id_ for id_, count in id_counts.items() if count > 1]
        violations.append(f"Duplicate customer_ids found: {dups[:5]}")

    if dup_emails > 0:
        notes.append(
            f"⚠️  {dup_emails} duplicate email address(es) detected — Faker name collision. "
            f"customer_id is the authoritative unique key. "
            f"Email uniqueness is not enforced by design."
        )

    notes.append(f"Unique customer_ids: {len(set(ids))} of {len(ids)}")
    return len(violations) == 0, violations, notes


# ============================================================================
# MAIN AUDIT RUNNER
# ============================================================================

def run_audit(
    customers: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all six DAMA dimension checks and return a full audit report.
    Raises ValueError if any hard violation is found.
    """
    checks = {
        "Accuracy":     check_accuracy,
        "Completeness": check_completeness,
        "Consistency":  check_consistency,
        "Timeliness":   check_timeliness,
        "Validity":     check_validity,
        "Uniqueness":   check_uniqueness,
    }

    results = {}
    all_passed = True

    if verbose:
        print(f"\n{'='*65}")
        print("DAMA DATA QUALITY DIMENSIONS AUDIT")
        print(f"{'='*65}")
        print(f"Records audited: {len(customers)}\n")

    for dim, check_fn in checks.items():
        passed, violations, notes = check_fn(customers)
        results[dim] = {
            "passed":     passed,
            "violations": violations,
            "notes":      notes,
        }
        if not passed:
            all_passed = False

        if verbose:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"[{dim}]  {status}")
            for note in notes:
                print(f"    {note}")
            if violations:
                print(f"    Violations ({len(violations)}):")
                for v in violations[:5]:
                    print(f"      ❌ {v}")
                if len(violations) > 5:
                    print(f"      ... and {len(violations)-5} more")
            print()

    if verbose:
        print(f"{'='*65}")
        overall = "✅ ALL DIMENSIONS PASS" if all_passed else "❌ VIOLATIONS FOUND"
        print(f"RESULT: {overall}")
        print(f"{'='*65}\n")

    if not all_passed:
        failed = [d for d, r in results.items() if not r["passed"]]
        raise ValueError(
            f"Dataset failed DAMA audit on dimension(s): {failed}. "
            f"See report for details."
        )

    return {
        "passed":   all_passed,
        "n":        len(customers),
        "dimensions": results,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate customer dataset against DAMA data quality dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate existing canonical_customers.json
  python validate_dama_dimensions.py --input experiments_output/eval/canonical_customers.json

  # Generate fresh dataset and validate
  python validate_dama_dimensions.py --generate --n 1000 --seed 42

  # Generate, validate, and save audit report
  python validate_dama_dimensions.py --generate --n 1000 --seed 42 --report audit.json
        """
    )
    parser.add_argument("--input", type=str, default=None,
        help="Path to canonical_customers.json to validate")
    parser.add_argument("--generate", action="store_true",
        help="Generate a fresh dataset before validating")
    parser.add_argument("--n", type=int, default=1000,
        help="Number of customers to generate (requires --generate)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed (requires --generate)")
    parser.add_argument("--report", type=str, default=None,
        help="Path to save JSON audit report")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress detailed output, only print PASS/FAIL")

    args = parser.parse_args()

    if not args.generate and not args.input:
        print("Error: provide --input or --generate")
        return 1

    # Load or generate customers
    if args.generate:
        # Import here to avoid circular dependency if used as a module
        sys.path.insert(0, str(Path(__file__).parent))
        from base_customer_generator import generate_base_customers
        print(f"Generating {args.n} customers (seed={args.seed})...")
        customers = generate_base_customers(n=args.n, seed=args.seed)
        print(f"Generated {len(customers)} customers\n")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}")
            return 1
        print(f"Loading {input_path}...")
        with open(input_path) as f:
            customers = json.load(f)
        print(f"Loaded {len(customers)} customers\n")

    # Run audit
    try:
        report = run_audit(customers, verbose=not args.quiet)
        if args.quiet:
            print("✅ ALL DAMA DIMENSIONS PASS")
    except ValueError as e:
        if args.quiet:
            print(f"❌ AUDIT FAILED: {e}")
        return 1

    # Save report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Audit report saved: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())

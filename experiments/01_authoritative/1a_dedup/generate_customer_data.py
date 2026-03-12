"""
Enhanced Data Generator for Deduplication Experiment

Generates synthetic customer data with:
- Rich, correlated customer profiles (20+ fields)
- Variable duplication (1-5+ duplicates per customer)
- Controlled variation levels (minimal/moderate/significant)
- Field-specific variation mode for H4 (feature importance testing)
- Segment-weighted duplication for H6 (segment distortion testing)
- Skeleton key for cluster-aware evaluation
- Shuffled agent input (no clustering bias)

Usage:
    # Standard run
    python generate_customer_data.py --n 500 --out experiments_output --levels 0,10,20,50,100

    # H4: Field-specific variation (test which fields affect decisions)
    python generate_customer_data.py --n 500 --out experiments_output --field-variation-mode --vary-fields email,phone,name

    # H5: Fine-grained duplication levels
    python generate_customer_data.py --n 500 --out experiments_output --levels 0,5,10,15,20,25,30,40,50,75,100

    # H6: Segment-biased duplication
    python generate_customer_data.py --n 500 --out experiments_output --segment-bias high_value

Outputs:
    agent/customers_dup_{level}pct.jsonl       -> Agent input (shuffled, no customer_id)
    eval/cluster_map_{level}pct.json           -> Skeleton key (record_id -> customer_id)
    eval/canonical_customers.json              -> Ground truth customer data
    metadata/generation_stats_{level}pct.json  -> Generation statistics
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
from faker import Faker

fake = Faker()
Faker.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Duplication distribution (how many duplicates per customer)
DUPLICATION_DISTRIBUTION = {
    1: 0.50,  # 50% of duplicated customers have 1 duplicate (2 total records)
    2: 0.25,  # 25% have 2 duplicates (3 total records)
    3: 0.15,  # 15% have 3 duplicates (4 total records)
    4: 0.07,  # 7% have 4 duplicates (5 total records)
    5: 0.03,  # 3% have 5+ duplicates (6+ total records)
}

# Variation levels for duplicates
VARIATION_LEVELS = {
    "minimal": 0.60,     # 60% of duplicates have minimal variation
    "moderate": 0.30,    # 30% have moderate variation
    "significant": 0.10  # 10% have significant variation
}

# Customer segments (affects field correlations)
CUSTOMER_SEGMENTS = {
    "high_value": 0.15,
    "medium_value": 0.50,
    "low_value": 0.25,
    "at_risk": 0.10
}

# H4: Fields grouped by decision relevance for tier assignment
# This lets experiments isolate whether field importance affects agent decisions
FIELD_GROUPS = {
    "identity_only":        ["name", "email", "phone", "address"],
    "tier_critical":        ["total_spend", "lifetime_value_estimate", "avg_order_value", "total_purchases", "customer_segment"],
    "engagement":           ["nps_score", "email_open_rate", "last_login_days_ago", "last_purchase_days_ago"],
    "risk":                 ["churn_risk_score", "payment_failures", "fraud_risk_score"],
    "flags":                ["is_vip", "is_at_risk", "has_active_subscription"],
}

# H6: Segment bias options — controls which segments are preferentially duplicated
SEGMENT_BIAS_OPTIONS = ["high_value", "medium_value", "low_value", "at_risk", "uniform"]

ACQUISITION_CHANNELS = ["organic", "referral", "paid_search", "social", "email", "partner"]
PRODUCT_CATEGORIES = ["electronics", "home", "kitchen", "apparel", "outdoors", "beauty", "office", "food", "sports", "toys"]


# ============================================================================
# CUSTOMER GENERATION
# ============================================================================

def generate_customer_segment() -> str:
    """Select customer segment based on distribution"""
    rand = random.random()
    cumulative = 0
    for segment, prob in CUSTOMER_SEGMENTS.items():
        cumulative += prob
        if rand <= cumulative:
            return segment
    return "medium_value"


def generate_base_customer(customer_id: str, seed: int) -> Dict[str, Any]:
    """
    Generate a single customer with rich, correlated attributes
    """
    random.seed(seed)
    np.random.seed(seed)

    segment = generate_customer_segment()

    name = fake.name()
    email_local = name.lower().replace(' ', '.').replace("'", '')
    email = f"{email_local}@example.com"
    phone = fake.phone_number()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=75).isoformat()

    if segment == "high_value":
        total_purchases = random.randint(30, 100)
        avg_order_value = random.uniform(200, 800)
        nps_score = random.randint(7, 10)
        churn_risk_score = random.uniform(0.0, 0.3)
        email_open_rate = random.uniform(0.6, 0.95)
        support_tickets_open = random.randint(0, 1)
        payment_failures = 0
        fraud_risk_score = random.uniform(0.0, 0.1)

    elif segment == "medium_value":
        total_purchases = random.randint(10, 35)
        avg_order_value = random.uniform(50, 250)
        nps_score = random.randint(5, 8)
        churn_risk_score = random.uniform(0.2, 0.5)
        email_open_rate = random.uniform(0.4, 0.7)
        support_tickets_open = random.randint(0, 2)
        payment_failures = random.randint(0, 1)
        fraud_risk_score = random.uniform(0.0, 0.2)

    elif segment == "low_value":
        total_purchases = random.randint(1, 12)
        avg_order_value = random.uniform(10, 80)
        nps_score = random.randint(4, 7)
        churn_risk_score = random.uniform(0.3, 0.6)
        email_open_rate = random.uniform(0.2, 0.5)
        support_tickets_open = random.randint(0, 3)
        payment_failures = random.randint(0, 2)
        fraud_risk_score = random.uniform(0.0, 0.3)

    else:  # at_risk
        total_purchases = random.randint(5, 25)
        avg_order_value = random.uniform(30, 150)
        nps_score = random.randint(2, 5)
        churn_risk_score = random.uniform(0.65, 0.95)
        email_open_rate = random.uniform(0.05, 0.3)
        support_tickets_open = random.randint(1, 5)
        payment_failures = random.randint(1, 3)
        fraud_risk_score = random.uniform(0.1, 0.4)

    total_spend = round(total_purchases * avg_order_value * random.uniform(0.9, 1.1), 2)
    lifetime_value_estimate = round(total_spend * random.uniform(1.2, 2.0), 2)
    purchase_frequency_days = max(1, int(365 / (total_purchases / random.uniform(1, 3))))

    tenure_months = random.randint(1, 60)
    last_purchase_days_ago = random.randint(1, min(365, tenure_months * 30))
    last_login_days_ago = random.randint(0, last_purchase_days_ago)
    support_tickets_closed = random.randint(support_tickets_open, support_tickets_open + 5)
    avg_resolution_time_hours = random.randint(2, 72)
    refund_rate = round(random.uniform(0.0, 0.15), 3)

    account_created_date = (datetime.now() - timedelta(days=tenure_months * 30)).date().isoformat()
    last_purchase_date = (datetime.now() - timedelta(days=last_purchase_days_ago)).date().isoformat()
    next_renewal_date = (datetime.now() + timedelta(days=random.randint(30, 365))).date().isoformat()

    acquisition_channel = random.choice(ACQUISITION_CHANNELS)
    preferred_categories = random.sample(PRODUCT_CATEGORIES, k=random.randint(1, 4))

    has_active_subscription = random.random() < 0.4
    is_vip = segment == "high_value" and random.random() < 0.3
    is_at_risk = churn_risk_score > 0.6
    has_pending_order = random.random() < 0.2
    recently_contacted_support = support_tickets_open > 0

    street_number = random.randint(1, 9999)
    street_name = fake.street_name()
    city = fake.city()
    state = fake.state_abbr()
    zip_code = fake.zipcode()
    address = f"{street_number} {street_name}, {city}, {state} {zip_code}"

    return {
        "customer_id": customer_id,
        "name": name,
        "email": email,
        "phone": phone,
        "dob": dob,
        "address": address,
        "total_purchases": total_purchases,
        "total_spend": total_spend,
        "avg_order_value": round(avg_order_value, 2),
        "purchase_frequency_days": purchase_frequency_days,
        "last_purchase_days_ago": last_purchase_days_ago,
        "lifetime_value_estimate": lifetime_value_estimate,
        "support_tickets_open": support_tickets_open,
        "support_tickets_closed": support_tickets_closed,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        "nps_score": nps_score,
        "email_open_rate": round(email_open_rate, 3),
        "last_login_days_ago": last_login_days_ago,
        "churn_risk_score": round(churn_risk_score, 3),
        "payment_failures": payment_failures,
        "refund_rate": refund_rate,
        "fraud_risk_score": round(fraud_risk_score, 3),
        "customer_segment": segment,
        "acquisition_channel": acquisition_channel,
        "tenure_months": tenure_months,
        "preferred_categories": preferred_categories,
        "account_created_date": account_created_date,
        "last_purchase_date": last_purchase_date,
        "next_renewal_date": next_renewal_date,
        "has_active_subscription": has_active_subscription,
        "is_vip": is_vip,
        "is_at_risk": is_at_risk,
        "has_pending_order": has_pending_order,
        "recently_contacted_support": recently_contacted_support,
    }


def generate_base_customers(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate n base customers"""
    random.seed(seed)
    customers = []
    for i in range(n):
        customer_id = f"CUST_{i+1:06d}"
        customer = generate_base_customer(customer_id, seed=seed + i)
        customers.append(customer)
    return customers


# ============================================================================
# FIELD-LEVEL VARIATION HELPERS
# ============================================================================

def vary_name(name: str) -> str:
    variations = [
        name,
        name.replace("a", "e"),
        name.replace("i", "y"),
        name.split()[0] + " " + name.split()[-1][0] + ".",
        name.upper(),
        name.lower(),
    ]
    return random.choice(variations)


def vary_email(email: str) -> str:
    local, domain = email.split("@")
    variations = [
        email,
        local.replace(".", "") + "@" + domain,
        local + str(random.randint(1, 99)) + "@" + domain,
        local.replace(".", "_") + "@" + domain,
    ]
    return random.choice(variations)


def vary_phone(phone: str) -> str:
    digits = ''.join(c for c in phone if c.isdigit())
    if len(digits) >= 10:
        formats = [
            f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}",
            f"{digits[:3]}-{digits[3:6]}-{digits[6:10]}",
            f"{digits[:10]}",
            f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:10]}",
        ]
        return random.choice(formats)
    return phone


def vary_address(address: str) -> str:
    variations = [
        address,
        address.replace("Street", "St").replace("Avenue", "Ave").replace("Road", "Rd"),
        address.upper(),
        address.replace(",", ""),
    ]
    return random.choice(variations)


# ============================================================================
# DUPLICATE GENERATION
# ============================================================================

def create_duplicate(
    base_customer: Dict[str, Any],
    variation_level: str,
    duplicate_num: int,
    field_variation_mode: bool = False,
    vary_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a duplicate with controlled variation.

    Two modes:
    - Standard mode (original): variation_level controls which fields change
    - Field variation mode (H4): vary_fields explicitly lists which fields to perturb,
      everything else stays identical to the canonical record
    """
    duplicate = deepcopy(base_customer)
    duplicate["record_id"] = f"REC_{uuid.uuid4().hex[:12].upper()}"

    # ----------------------------------------------------------------
    # H4 MODE: field-specific variation
    # Only the fields listed in vary_fields are perturbed.
    # This isolates the effect of individual field types on agent decisions.
    # ----------------------------------------------------------------
    if field_variation_mode and vary_fields:
        for field in vary_fields:
            if field == "name":
                duplicate["name"] = vary_name(base_customer["name"])
            elif field == "email":
                duplicate["email"] = vary_email(base_customer["email"])
            elif field == "phone":
                duplicate["phone"] = vary_phone(base_customer["phone"])
            elif field == "address":
                duplicate["address"] = vary_address(base_customer["address"])
            elif field == "total_spend":
                duplicate["total_spend"] = round(base_customer["total_spend"] * random.uniform(0.85, 1.15), 2)
            elif field == "lifetime_value_estimate":
                duplicate["lifetime_value_estimate"] = round(base_customer["lifetime_value_estimate"] * random.uniform(0.85, 1.15), 2)
            elif field == "avg_order_value":
                duplicate["avg_order_value"] = round(base_customer["avg_order_value"] * random.uniform(0.85, 1.15), 2)
            elif field == "total_purchases":
                duplicate["total_purchases"] = max(1, base_customer["total_purchases"] + random.randint(-5, 5))
            elif field == "nps_score":
                duplicate["nps_score"] = max(0, min(10, base_customer["nps_score"] + random.randint(-2, 2)))
            elif field == "churn_risk_score":
                duplicate["churn_risk_score"] = round(max(0.0, min(1.0, base_customer["churn_risk_score"] + random.uniform(-0.2, 0.2))), 3)
            elif field == "email_open_rate":
                duplicate["email_open_rate"] = round(max(0.0, min(1.0, base_customer["email_open_rate"] + random.uniform(-0.15, 0.15))), 3)
            elif field == "payment_failures":
                duplicate["payment_failures"] = max(0, base_customer["payment_failures"] + random.randint(-1, 2))
            elif field == "fraud_risk_score":
                duplicate["fraud_risk_score"] = round(max(0.0, min(1.0, base_customer["fraud_risk_score"] + random.uniform(-0.1, 0.1))), 3)
            elif field == "customer_segment":
                other_segments = [s for s in CUSTOMER_SEGMENTS if s != base_customer["customer_segment"]]
                duplicate["customer_segment"] = random.choice(other_segments)
            elif field == "is_vip":
                duplicate["is_vip"] = not base_customer["is_vip"]
            elif field == "is_at_risk":
                duplicate["is_at_risk"] = not base_customer["is_at_risk"]
            elif field == "has_active_subscription":
                duplicate["has_active_subscription"] = not base_customer["has_active_subscription"]

        # Record which fields were varied (for eval tracking)
        duplicate["_varied_fields"] = vary_fields
        duplicate["_variation_mode"] = "field_specific"

    # ----------------------------------------------------------------
    # STANDARD MODE: coarse variation levels (original behavior)
    # ----------------------------------------------------------------
    else:
        if variation_level == "minimal":
            duplicate["name"] = vary_name(base_customer["name"])
            duplicate["email"] = vary_email(base_customer["email"])
            duplicate["phone"] = vary_phone(base_customer["phone"])
            duplicate["address"] = vary_address(base_customer["address"])

        elif variation_level == "moderate":
            duplicate["name"] = vary_name(base_customer["name"])
            duplicate["email"] = vary_email(base_customer["email"])
            duplicate["phone"] = vary_phone(base_customer["phone"])
            duplicate["last_purchase_days_ago"] = max(0, base_customer["last_purchase_days_ago"] + random.randint(-7, 7))
            duplicate["last_login_days_ago"] = max(0, base_customer["last_login_days_ago"] + random.randint(-3, 3))
            duplicate["nps_score"] = max(0, min(10, base_customer["nps_score"] + random.randint(-1, 1)))
            duplicate["email_open_rate"] = max(0.0, min(1.0, base_customer["email_open_rate"] + random.uniform(-0.05, 0.05)))
            duplicate["support_tickets_open"] = max(0, base_customer["support_tickets_open"] + random.randint(-1, 1))

        elif variation_level == "significant":
            duplicate["name"] = vary_name(base_customer["name"])
            duplicate["email"] = vary_email(base_customer["email"])
            duplicate["phone"] = vary_phone(base_customer["phone"])
            duplicate["address"] = vary_address(base_customer["address"])
            duplicate["total_spend"] = round(base_customer["total_spend"] * random.uniform(0.9, 1.1), 2)
            duplicate["last_purchase_days_ago"] = max(0, base_customer["last_purchase_days_ago"] + random.randint(-30, 30))
            duplicate["churn_risk_score"] = max(0.0, min(1.0, base_customer["churn_risk_score"] + random.uniform(-0.15, 0.15)))
            duplicate["nps_score"] = max(0, min(10, base_customer["nps_score"] + random.randint(-2, 2)))
            duplicate["support_tickets_open"] = max(0, base_customer["support_tickets_open"] + random.randint(-2, 2))
            duplicate["payment_failures"] = max(0, base_customer["payment_failures"] + random.randint(-1, 1))
            duplicate["email_open_rate"] = max(0.0, min(1.0, base_customer["email_open_rate"] + random.uniform(-0.15, 0.15)))
            duplicate["is_at_risk"] = duplicate["churn_risk_score"] > 0.6
            duplicate["recently_contacted_support"] = duplicate["support_tickets_open"] > 0

        duplicate["_variation_mode"] = "standard"

    # Metadata for evaluation (not visible to agent)
    duplicate["_variation_level"] = variation_level
    duplicate["_duplicate_number"] = duplicate_num
    duplicate["_canonical_customer_id"] = base_customer["customer_id"]

    return duplicate


def select_variation_level() -> str:
    rand = random.random()
    cumulative = 0
    for level, prob in VARIATION_LEVELS.items():
        cumulative += prob
        if rand <= cumulative:
            return level
    return "minimal"


def select_num_duplicates() -> int:
    rand = random.random()
    cumulative = 0
    for num_dups, prob in DUPLICATION_DISTRIBUTION.items():
        cumulative += prob
        if rand <= cumulative:
            return num_dups
    return 1


# ============================================================================
# H6: SEGMENT-BIASED CUSTOMER SELECTION
# ============================================================================

def select_customers_to_duplicate(
    base_customers: List[Dict[str, Any]],
    n_to_duplicate: int,
    segment_bias: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Select which customers to duplicate, optionally biasing toward a segment.

    segment_bias options:
    - None / "uniform": random selection (original behavior)
    - "high_value", "medium_value", "low_value", "at_risk":
        customers in that segment are 4x more likely to be selected

    This lets H6 test whether segment-specific duplication distorts
    aggregate decisions differently than uniform duplication.
    """
    if not segment_bias or segment_bias == "uniform":
        return random.sample(base_customers, n_to_duplicate) if n_to_duplicate > 0 else []

    # Assign weights: biased segment gets 4x weight
    weights = [
        4.0 if c["customer_segment"] == segment_bias else 1.0
        for c in base_customers
    ]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    indices = np.random.choice(
        len(base_customers),
        size=min(n_to_duplicate, len(base_customers)),
        replace=False,
        p=probabilities
    )
    return [base_customers[i] for i in indices]


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_duplicates_for_level(
    base_customers: List[Dict[str, Any]],
    duplication_pct: int,
    seed: int,
    segment_bias: Optional[str] = None,
    field_variation_mode: bool = False,
    vary_fields: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Generate duplicates for a given duplication percentage.

    New parameters vs original:
    - segment_bias: H6 - bias duplication toward a specific customer segment
    - field_variation_mode: H4 - use explicit field-level variation instead of coarse levels
    - vary_fields: H4 - which fields to perturb when field_variation_mode=True
    """
    random.seed(seed)
    np.random.seed(seed)

    n_customers = len(base_customers)
    n_to_duplicate = int(n_customers * (duplication_pct / 100.0))

    # H6: segment-biased selection
    customers_to_duplicate = select_customers_to_duplicate(
        base_customers, n_to_duplicate, segment_bias
    )

    all_records = []
    cluster_map = {}
    duplication_stats = {
        "total_customers": n_customers,
        "customers_duplicated": n_to_duplicate,
        "segment_bias": segment_bias or "uniform",
        "field_variation_mode": field_variation_mode,
        "varied_fields": vary_fields or [],
        "cluster_size_distribution": Counter(),
        "variation_level_distribution": Counter(),
        # H6: track duplication rate per segment for analysis
        "duplicated_by_segment": Counter(),
    }

    # Add all base customers with record_ids
    for customer in base_customers:
        base_record = deepcopy(customer)
        base_record["record_id"] = f"REC_{uuid.uuid4().hex[:12].upper()}"
        all_records.append(base_record)

        cluster_map[customer["customer_id"]] = {
            "customer_id": customer["customer_id"],
            "record_ids": [base_record["record_id"]],
            "cluster_size": 1,
            "canonical_record": base_record,
            "has_duplicates": False,
            "variation_levels": [],
            "customer_segment": customer["customer_segment"],  # H6: segment tracking
        }

    # Generate duplicates
    for customer in customers_to_duplicate:
        customer_id = customer["customer_id"]
        num_duplicates = select_num_duplicates()

        duplication_stats["duplicated_by_segment"][customer["customer_segment"]] += 1

        for dup_num in range(1, num_duplicates + 1):
            variation_level = select_variation_level()
            duplicate = create_duplicate(
                customer,
                variation_level,
                dup_num,
                field_variation_mode=field_variation_mode,
                vary_fields=vary_fields,
            )
            all_records.append(duplicate)

            cluster_map[customer_id]["record_ids"].append(duplicate["record_id"])
            cluster_map[customer_id]["cluster_size"] += 1
            cluster_map[customer_id]["has_duplicates"] = True
            cluster_map[customer_id]["variation_levels"].append(variation_level)

            duplication_stats["variation_level_distribution"][variation_level] += 1

        duplication_stats["cluster_size_distribution"][cluster_map[customer_id]["cluster_size"]] += 1

    random.shuffle(all_records)

    duplication_stats["total_records"] = len(all_records)
    duplication_stats["total_duplicates"] = len(all_records) - n_customers
    duplication_stats["avg_cluster_size"] = len(all_records) / n_customers

    # H6: compute per-segment duplication rates for easy analysis
    segment_counts = Counter(c["customer_segment"] for c in base_customers)
    duplication_stats["segment_duplication_rates"] = {
        seg: round(duplication_stats["duplicated_by_segment"].get(seg, 0) / count, 3)
        for seg, count in segment_counts.items()
    }

    return all_records, cluster_map, duplication_stats


def prepare_agent_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove customer_id and all metadata fields before handing to agent"""
    agent_records = []
    for record in all_records:
        agent_record = {
            key: value for key, value in record.items()
            if key != "customer_id" and not key.startswith("_")
        }
        agent_records.append(agent_record)
    return agent_records


def prepare_eval_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return deepcopy(all_records)


# ============================================================================
# FILE OUTPUT
# ============================================================================

def save_agent_data(agent_records: List[Dict[str, Any]], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for record in agent_records:
            f.write(json.dumps(record, default=str) + '\n')
    print(f"  ✅ Agent data: {output_file} ({len(agent_records)} records)")


def save_cluster_map(cluster_map: Dict[str, Any], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serializable_map = {}
    for customer_id, cluster_info in cluster_map.items():
        serializable_map[customer_id] = {
            "customer_id": cluster_info["customer_id"],
            "record_ids": cluster_info["record_ids"],
            "cluster_size": cluster_info["cluster_size"],
            "has_duplicates": cluster_info["has_duplicates"],
            "variation_levels": cluster_info["variation_levels"],
            "customer_segment": cluster_info["customer_segment"],  # H6
            "canonical_customer_segment": cluster_info["canonical_record"]["customer_segment"],
            "canonical_total_spend": cluster_info["canonical_record"]["total_spend"],
        }
    with open(output_file, 'w') as f:
        json.dump(serializable_map, f, indent=2, default=str)
    print(f"  ✅ Cluster map: {output_file} ({len(cluster_map)} clusters)")


def save_canonical_customers(base_customers: List[Dict[str, Any]], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(base_customers, f, indent=2, default=str)
    print(f"  ✅ Canonical customers: {output_file} ({len(base_customers)} customers)")


def save_generation_stats(stats: Dict[str, Any], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serializable_stats = deepcopy(stats)
    for key in ["cluster_size_distribution", "variation_level_distribution",
                "duplicated_by_segment"]:
        if key in serializable_stats:
            serializable_stats[key] = dict(serializable_stats[key])
    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2, default=str)
    print(f"  ✅ Generation stats: {output_file}")


# ============================================================================
# MAIN GENERATION WORKFLOW
# ============================================================================

def generate_all_datasets(
    n_customers: int,
    duplication_levels: List[int],
    output_dir: Path,
    seed: int = 42,
    segment_bias: Optional[str] = None,
    field_variation_mode: bool = False,
    vary_fields: Optional[List[str]] = None,
):
    print("\n" + "="*70)
    print("🚀 GENERATING CUSTOMER DATA")
    print("="*70)
    print(f"Base customers:       {n_customers}")
    print(f"Duplication levels:   {duplication_levels}")
    print(f"Output directory:     {output_dir}")
    print(f"Random seed:          {seed}")
    print(f"Segment bias (H6):    {segment_bias or 'uniform (none)'}")
    print(f"Field variation (H4): {field_variation_mode}")
    if field_variation_mode and vary_fields:
        print(f"  Varied fields:      {vary_fields}")
    print()

    agent_dir = output_dir / "agent"
    eval_dir = output_dir / "eval"
    metadata_dir = output_dir / "metadata"
    agent_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Generating base customers...")
    base_customers = generate_base_customers(n_customers, seed=seed)
    print(f"  ✅ Generated {len(base_customers)} base customers")
    print()

    save_canonical_customers(base_customers, eval_dir / "canonical_customers.json")
    print()

    for level in duplication_levels:
        print(f"📦 Generating {level}% duplication dataset...")

        all_records, cluster_map, stats = generate_duplicates_for_level(
            base_customers,
            duplication_pct=level,
            seed=seed + level,
            segment_bias=segment_bias,
            field_variation_mode=field_variation_mode,
            vary_fields=vary_fields,
        )

        agent_records = prepare_agent_records(all_records)

        save_agent_data(agent_records, agent_dir / f"customers_dup_{level}pct.jsonl")
        save_cluster_map(cluster_map, eval_dir / f"cluster_map_{level}pct.json")
        save_generation_stats(stats, metadata_dir / f"generation_stats_{level}pct.json")

        print(f"  📊 Stats:")
        print(f"     Total records:             {stats['total_records']}")
        print(f"     Duplicates added:          {stats['total_duplicates']}")
        print(f"     Avg cluster size:          {stats['avg_cluster_size']:.2f}")
        print(f"     Cluster size distribution: {dict(stats['cluster_size_distribution'])}")
        print(f"     Variation levels:          {dict(stats['variation_level_distribution'])}")
        if segment_bias:
            print(f"     Segment duplication rates: {stats['segment_duplication_rates']}")
        print()

    print("="*70)
    print("✅ DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput structure:")
    print(f"  {agent_dir}/")
    print(f"    └── customers_dup_{{level}}pct.jsonl")
    print(f"  {eval_dir}/")
    print(f"    ├── canonical_customers.json")
    print(f"    └── cluster_map_{{level}}pct.json")
    print(f"  {metadata_dir}/")
    print(f"    └── generation_stats_{{level}}pct.json")
    print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer data with controlled duplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python generate_customer_data.py --n 500 --out experiments_output

  # H4: Only vary tier-critical fields (tests decision sensitivity)
  python generate_customer_data.py --n 500 --out h4_tier_critical --field-variation-mode --vary-fields total_spend,lifetime_value_estimate,avg_order_value

  # H4: Only vary identity fields (tests if typos affect decisions)
  python generate_customer_data.py --n 500 --out h4_identity --field-variation-mode --vary-fields name,email,phone

  # H5: Fine-grained duplication levels
  python generate_customer_data.py --n 500 --out h5_granular --levels 0,5,10,15,20,25,30,40,50,75,100

  # H6: High-value segment preferentially duplicated
  python generate_customer_data.py --n 500 --out h6_high_value --segment-bias high_value

  # H6: At-risk segment preferentially duplicated
  python generate_customer_data.py --n 500 --out h6_at_risk --segment-bias at_risk

Available field groups for --vary-fields:
  identity_only:   name, email, phone, address
  tier_critical:   total_spend, lifetime_value_estimate, avg_order_value, total_purchases, customer_segment
  engagement:      nps_score, email_open_rate, last_login_days_ago, last_purchase_days_ago
  risk:            churn_risk_score, payment_failures, fraud_risk_score
  flags:           is_vip, is_at_risk, has_active_subscription
        """
    )

    parser.add_argument("--n", type=int, default=500,
        help="Number of base customers (default: 500)")
    parser.add_argument("--out", type=str, default="experiments_output",
        help="Output directory (default: experiments_output)")
    parser.add_argument("--levels", type=str, default="0,10,20,30,40,50,75,100",
        help="Comma-separated duplication levels in percent")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed (default: 42)")

    # H4 arguments
    parser.add_argument("--field-variation-mode", action="store_true",
        help="H4: Enable field-specific variation mode")
    parser.add_argument("--vary-fields", type=str, default=None,
        help="H4: Comma-separated fields to vary (e.g. 'name,email,total_spend')")

    # H6 argument
    parser.add_argument("--segment-bias", type=str, default=None,
        choices=SEGMENT_BIAS_OPTIONS,
        help="H6: Bias duplication toward a specific customer segment")

    args = parser.parse_args()

    duplication_levels = sorted(set(int(x.strip()) for x in args.levels.split(",")))

    for level in duplication_levels:
        if level < 0 or level > 100:
            print(f"❌ Error: Duplication level {level} must be between 0 and 100")
            return 1

    vary_fields = None
    if args.vary_fields:
        vary_fields = [f.strip() for f in args.vary_fields.split(",")]

    if args.field_variation_mode and not vary_fields:
        print("❌ Error: --field-variation-mode requires --vary-fields")
        return 1

    try:
        generate_all_datasets(
            n_customers=args.n,
            duplication_levels=duplication_levels,
            output_dir=Path(args.out),
            seed=args.seed,
            segment_bias=args.segment_bias,
            field_variation_mode=args.field_variation_mode,
            vary_fields=vary_fields,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

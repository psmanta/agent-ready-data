#!/usr/bin/env python3
"""
Enhanced Data Generator for Deduplication Experiment

Generates synthetic customer data with:
- Rich, correlated customer profiles (20+ fields)
- Variable duplication (1-5+ duplicates per customer)
- Controlled variation levels (minimal/moderate/significant)
- Skeleton key for cluster-aware evaluation
- Shuffled agent input (no clustering bias)

Usage:
    python generate_customer_data.py --n 500 --out experiments_output --levels 0,10,20,50,100

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
from typing import List, Dict, Any, Tuple
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
    
    Returns customer dict with 25+ fields including:
    - Identity (name, email, phone, DOB)
    - Behavioral signals (purchases, spend, frequency)
    - Engagement metrics (NPS, email open rate, logins)
    - Risk signals (churn, fraud, payment failures)
    - Segmentation (segment, channel, tenure)
    - Temporal context (dates, renewal)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Determine segment (affects all other fields)
    segment = generate_customer_segment()
    
    # Identity fields
    name = fake.name()
    email_local = name.lower().replace(' ', '.').replace("'", '')
    email = f"{email_local}@example.com"
    phone = fake.phone_number()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=75).isoformat()
    
    # Segment-correlated behavioral fields
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
    
    # Derived fields
    total_spend = round(total_purchases * avg_order_value * random.uniform(0.9, 1.1), 2)
    lifetime_value_estimate = round(total_spend * random.uniform(1.2, 2.0), 2)
    purchase_frequency_days = max(1, int(365 / (total_purchases / random.uniform(1, 3))))
    
    # Engagement fields
    tenure_months = random.randint(1, 60)
    last_purchase_days_ago = random.randint(1, min(365, tenure_months * 30))
    last_login_days_ago = random.randint(0, last_purchase_days_ago)
    support_tickets_closed = random.randint(support_tickets_open, support_tickets_open + 5)
    avg_resolution_time_hours = random.randint(2, 72)
    refund_rate = round(random.uniform(0.0, 0.15), 3)
    
    # Temporal fields
    account_created_date = (datetime.now() - timedelta(days=tenure_months * 30)).date().isoformat()
    last_purchase_date = (datetime.now() - timedelta(days=last_purchase_days_ago)).date().isoformat()
    next_renewal_date = (datetime.now() + timedelta(days=random.randint(30, 365))).date().isoformat()
    
    # Categorical fields
    acquisition_channel = random.choice(ACQUISITION_CHANNELS)
    preferred_categories = random.sample(PRODUCT_CATEGORIES, k=random.randint(1, 4))
    
    # Boolean flags
    has_active_subscription = random.random() < 0.4
    is_vip = segment == "high_value" and random.random() < 0.3
    is_at_risk = churn_risk_score > 0.6
    has_pending_order = random.random() < 0.2
    recently_contacted_support = support_tickets_open > 0
    
    # Address (for variation in duplicates)
    street_number = random.randint(1, 9999)
    street_name = fake.street_name()
    city = fake.city()
    state = fake.state_abbr()
    zip_code = fake.zipcode()
    address = f"{street_number} {street_name}, {city}, {state} {zip_code}"
    
    return {
        # Identity
        "customer_id": customer_id,
        "name": name,
        "email": email,
        "phone": phone,
        "dob": dob,
        "address": address,
        
        # Behavioral
        "total_purchases": total_purchases,
        "total_spend": total_spend,
        "avg_order_value": round(avg_order_value, 2),
        "purchase_frequency_days": purchase_frequency_days,
        "last_purchase_days_ago": last_purchase_days_ago,
        "lifetime_value_estimate": lifetime_value_estimate,
        
        # Engagement
        "support_tickets_open": support_tickets_open,
        "support_tickets_closed": support_tickets_closed,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        "nps_score": nps_score,
        "email_open_rate": round(email_open_rate, 3),
        "last_login_days_ago": last_login_days_ago,
        
        # Risk
        "churn_risk_score": round(churn_risk_score, 3),
        "payment_failures": payment_failures,
        "refund_rate": refund_rate,
        "fraud_risk_score": round(fraud_risk_score, 3),
        
        # Segmentation
        "customer_segment": segment,
        "acquisition_channel": acquisition_channel,
        "tenure_months": tenure_months,
        "preferred_categories": preferred_categories,
        
        # Temporal
        "account_created_date": account_created_date,
        "last_purchase_date": last_purchase_date,
        "next_renewal_date": next_renewal_date,
        
        # Flags
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
# DUPLICATE GENERATION WITH VARIATION
# ============================================================================

def vary_name(name: str) -> str:
    """Create name variation (typos, abbreviations)"""
    variations = [
        name,  # Keep original sometimes
        name.replace("a", "e"),  # Typo
        name.replace("i", "y"),  # Typo
        name.split()[0] + " " + name.split()[-1][0] + ".",  # First + Last initial
        name.upper(),  # All caps
        name.lower(),  # All lowercase
    ]
    return random.choice(variations)


def vary_email(email: str) -> str:
    """Create email variation"""
    local, domain = email.split("@")
    variations = [
        email,
        local.replace(".", "") + "@" + domain,  # Remove dots
        local + str(random.randint(1, 99)) + "@" + domain,  # Add number
        local.replace(".", "_") + "@" + domain,  # Underscore instead of dot
    ]
    return random.choice(variations)


def vary_phone(phone: str) -> str:
    """Create phone variation"""
    # Just change formatting
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
    """Create address variation"""
    # Minor changes to address
    variations = [
        address,
        address.replace("Street", "St").replace("Avenue", "Ave").replace("Road", "Rd"),
        address.upper(),
        address.replace(",", ""),
    ]
    return random.choice(variations)

def create_duplicate(base_customer: Dict[str, Any], variation_level: str, duplicate_num: int) -> Dict[str, Any]:
    """
    Create a duplicate of base_customer with controlled variation
    
    Args:
        base_customer: Original customer record
        variation_level: "minimal", "moderate", or "significant"
        duplicate_num: Which duplicate this is (1, 2, 3, etc.)
    
    Returns:
        Duplicate record with new record_id but same customer_id
    """
    duplicate = deepcopy(base_customer)
    
    # Generate unique record_id (but keep same customer_id)
    duplicate["record_id"] = f"REC_{uuid.uuid4().hex[:12].upper()}"
    
    if variation_level == "minimal":
        # Only PII varies (name typos, email variants, phone formatting)
        duplicate["name"] = vary_name(base_customer["name"])
        duplicate["email"] = vary_email(base_customer["email"])
        duplicate["phone"] = vary_phone(base_customer["phone"])
        duplicate["address"] = vary_address(base_customer["address"])
    
    elif variation_level == "moderate":
        # PII + some behavioral metrics vary slightly
        duplicate["name"] = vary_name(base_customer["name"])
        duplicate["email"] = vary_email(base_customer["email"])
        duplicate["phone"] = vary_phone(base_customer["phone"])
        
        # Slight variations in temporal/behavioral fields
        duplicate["last_purchase_days_ago"] = max(0, base_customer["last_purchase_days_ago"] + random.randint(-7, 7))
        duplicate["last_login_days_ago"] = max(0, base_customer["last_login_days_ago"] + random.randint(-3, 3))
        duplicate["nps_score"] = max(0, min(10, base_customer["nps_score"] + random.randint(-1, 1)))
        duplicate["email_open_rate"] = max(0.0, min(1.0, base_customer["email_open_rate"] + random.uniform(-0.05, 0.05)))
        duplicate["support_tickets_open"] = max(0, base_customer["support_tickets_open"] + random.randint(-1, 1))
    
    elif variation_level == "significant":
        # Multiple fields vary (tests agent robustness)
        duplicate["name"] = vary_name(base_customer["name"])
        duplicate["email"] = vary_email(base_customer["email"])
        duplicate["phone"] = vary_phone(base_customer["phone"])
        duplicate["address"] = vary_address(base_customer["address"])
        
        # More significant variations in behavioral fields
        duplicate["total_spend"] = round(base_customer["total_spend"] * random.uniform(0.9, 1.1), 2)
        duplicate["last_purchase_days_ago"] = max(0, base_customer["last_purchase_days_ago"] + random.randint(-30, 30))
        duplicate["churn_risk_score"] = max(0.0, min(1.0, base_customer["churn_risk_score"] + random.uniform(-0.15, 0.15)))
        duplicate["nps_score"] = max(0, min(10, base_customer["nps_score"] + random.randint(-2, 2)))
        duplicate["support_tickets_open"] = max(0, base_customer["support_tickets_open"] + random.randint(-2, 2))
        duplicate["payment_failures"] = max(0, base_customer["payment_failures"] + random.randint(-1, 1))
        duplicate["email_open_rate"] = max(0.0, min(1.0, base_customer["email_open_rate"] + random.uniform(-0.15, 0.15)))
        
        # Update derived flags based on changed values
        duplicate["is_at_risk"] = duplicate["churn_risk_score"] > 0.6
        duplicate["recently_contacted_support"] = duplicate["support_tickets_open"] > 0
    
    # Add metadata for evaluation (not visible to agent)
    duplicate["_variation_level"] = variation_level
    duplicate["_duplicate_number"] = duplicate_num
    duplicate["_canonical_customer_id"] = base_customer["customer_id"]
    
    return duplicate


def select_variation_level() -> str:
    """Select variation level based on distribution"""
    rand = random.random()
    cumulative = 0
    for level, prob in VARIATION_LEVELS.items():
        cumulative += prob
        if rand <= cumulative:
            return level
    return "minimal"


def select_num_duplicates() -> int:
    """Select number of duplicates based on distribution"""
    rand = random.random()
    cumulative = 0
    for num_dups, prob in DUPLICATION_DISTRIBUTION.items():
        cumulative += prob
        if rand <= cumulative:
            return num_dups
    return 1


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_duplicates_for_level(
    base_customers: List[Dict[str, Any]],
    duplication_pct: int,
    seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate duplicates for a given duplication percentage
    
    Args:
        base_customers: List of base customer records
        duplication_pct: Percentage of customers to duplicate (0-100)
        seed: Random seed
    
    Returns:
        Tuple of (all_records, cluster_map)
        - all_records: List of all records (base + duplicates)
        - cluster_map: Dict mapping customer_id to record_ids and metadata
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_customers = len(base_customers)
    n_to_duplicate = int(n_customers * (duplication_pct / 100.0))
    
    # Select which customers to duplicate
    customers_to_duplicate = random.sample(base_customers, n_to_duplicate) if n_to_duplicate > 0 else []
    
    all_records = []
    cluster_map = {}
    duplication_stats = {
        "total_customers": n_customers,
        "customers_duplicated": n_to_duplicate,
        "cluster_size_distribution": Counter(),
        "variation_level_distribution": Counter(),
    }
    
    # Add all base customers with record_ids
    for customer in base_customers:
        base_record = deepcopy(customer)
        base_record["record_id"] = f"REC_{uuid.uuid4().hex[:12].upper()}"
        all_records.append(base_record)
        
        # Initialize cluster map
        cluster_map[customer["customer_id"]] = {
            "customer_id": customer["customer_id"],
            "record_ids": [base_record["record_id"]],
            "cluster_size": 1,
            "canonical_record": base_record,
            "has_duplicates": False,
            "variation_levels": []
        }
    
    # Generate duplicates for selected customers
    for customer in customers_to_duplicate:
        customer_id = customer["customer_id"]
        
        # Determine how many duplicates to create
        num_duplicates = select_num_duplicates()
        
        # Create duplicates with varying levels
        for dup_num in range(1, num_duplicates + 1):
            variation_level = select_variation_level()
            duplicate = create_duplicate(customer, variation_level, dup_num)
            all_records.append(duplicate)
            
            # Update cluster map
            cluster_map[customer_id]["record_ids"].append(duplicate["record_id"])
            cluster_map[customer_id]["cluster_size"] += 1
            cluster_map[customer_id]["has_duplicates"] = True
            cluster_map[customer_id]["variation_levels"].append(variation_level)
            
            # Update stats
            duplication_stats["variation_level_distribution"][variation_level] += 1
        
        # Update cluster size stats
        duplication_stats["cluster_size_distribution"][cluster_map[customer_id]["cluster_size"]] += 1
    
    # Shuffle records (critical: removes clustering bias for agent)
    random.shuffle(all_records)
    
    # Calculate final stats
    duplication_stats["total_records"] = len(all_records)
    duplication_stats["total_duplicates"] = len(all_records) - n_customers
    duplication_stats["avg_cluster_size"] = len(all_records) / n_customers
    
    return all_records, cluster_map, duplication_stats


def prepare_agent_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare records for agent (remove customer_id and metadata)
    
    Agent should NOT see:
    - customer_id (would reveal clustering)
    - _variation_level, _duplicate_number, etc. (metadata)
    
    Agent SHOULD see:
    - record_id (for tracking decisions)
    - All business-relevant fields
    """
    agent_records = []
    
    for record in all_records:
        agent_record = {}
        
        for key, value in record.items():
            # Skip customer_id and metadata fields
            if key == "customer_id" or key.startswith("_"):
                continue
            agent_record[key] = value
        
        agent_records.append(agent_record)
    
    return agent_records


def prepare_eval_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare records for evaluation (keep all fields including metadata)
    """
    return deepcopy(all_records)


# ============================================================================
# FILE OUTPUT
# ============================================================================

def save_agent_data(agent_records: List[Dict[str, Any]], output_file: Path):
    """Save agent-facing data (JSONL format, shuffled, no customer_id)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for record in agent_records:
            f.write(json.dumps(record, default=str) + '\n')
    
    print(f"  ✅ Agent data: {output_file} ({len(agent_records)} records)")


def save_cluster_map(cluster_map: Dict[str, Any], output_file: Path):
    """Save skeleton key (cluster mapping for evaluation)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable_map = {}
    for customer_id, cluster_info in cluster_map.items():
        serializable_map[customer_id] = {
            "customer_id": cluster_info["customer_id"],
            "record_ids": cluster_info["record_ids"],
            "cluster_size": cluster_info["cluster_size"],
            "has_duplicates": cluster_info["has_duplicates"],
            "variation_levels": cluster_info["variation_levels"],
            # Don't include full canonical_record (too verbose)
            "canonical_customer_segment": cluster_info["canonical_record"]["customer_segment"],
            "canonical_total_spend": cluster_info["canonical_record"]["total_spend"],
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_map, f, indent=2, default=str)
    
    print(f"  ✅ Cluster map: {output_file} ({len(cluster_map)} clusters)")


def save_canonical_customers(base_customers: List[Dict[str, Any]], output_file: Path):
    """Save canonical customer data (ground truth)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(base_customers, f, indent=2, default=str)
    
    print(f"  ✅ Canonical customers: {output_file} ({len(base_customers)} customers)")


def save_generation_stats(stats: Dict[str, Any], output_file: Path):
    """Save generation statistics"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Counter objects to dicts
    serializable_stats = deepcopy(stats)
    if "cluster_size_distribution" in serializable_stats:
        serializable_stats["cluster_size_distribution"] = dict(serializable_stats["cluster_size_distribution"])
    if "variation_level_distribution" in serializable_stats:
        serializable_stats["variation_level_distribution"] = dict(serializable_stats["variation_level_distribution"])
    
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
    seed: int = 42
):
    """
    Generate all datasets for experiment
    
    Args:
        n_customers: Number of base customers
        duplication_levels: List of duplication percentages (e.g., [0, 10, 50, 100])
        output_dir: Base output directory
        seed: Random seed
    """
    print("\n" + "="*70)
    print("🚀 GENERATING CUSTOMER DATA")
    print("="*70)
    print(f"Base customers: {n_customers}")
    print(f"Duplication levels: {duplication_levels}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print()
    
    # Create output directories
    agent_dir = output_dir / "agent"
    eval_dir = output_dir / "eval"
    metadata_dir = output_dir / "metadata"
    
    agent_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base customers
    print("📊 Generating base customers...")
    base_customers = generate_base_customers(n_customers, seed=seed)
    print(f"  ✅ Generated {len(base_customers)} base customers")
    print()
    
    # Save canonical customers (ground truth)
    save_canonical_customers(base_customers, eval_dir / "canonical_customers.json")
    print()
    
    # Generate datasets for each duplication level
    for level in duplication_levels:
        print(f"📦 Generating {level}% duplication dataset...")
        
        # Generate duplicates
        all_records, cluster_map, stats = generate_duplicates_for_level(
            base_customers,
            duplication_pct=level,
            seed=seed + level
        )
        
        # Prepare agent and eval versions
        agent_records = prepare_agent_records(all_records)
        eval_records = prepare_eval_records(all_records)
        
        # Save files
        save_agent_data(agent_records, agent_dir / f"customers_dup_{level}pct.jsonl")
        save_cluster_map(cluster_map, eval_dir / f"cluster_map_{level}pct.json")
        save_generation_stats(stats, metadata_dir / f"generation_stats_{level}pct.json")
        
        # Print summary
        print(f"  📊 Stats:")
        print(f"     Total records: {stats['total_records']}")
        print(f"     Duplicates added: {stats['total_duplicates']}")
        print(f"     Avg cluster size: {stats['avg_cluster_size']:.2f}")
        print(f"     Cluster size distribution: {dict(stats['cluster_size_distribution'])}")
        print(f"     Variation levels: {dict(stats['variation_level_distribution'])}")
        print()
    
    print("="*70)
    print("✅ DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput structure:")
    print(f"  {agent_dir}/")
    print(f"    └── customers_dup_{{level}}pct.jsonl  (agent input, shuffled, no customer_id)")
    print(f"  {eval_dir}/")
    print(f"    ├── canonical_customers.json         (ground truth)")
    print(f"    └── cluster_map_{{level}}pct.json      (skeleton key for evaluation)")
    print(f"  {metadata_dir}/")
    print(f"    └── generation_stats_{{level}}pct.json (generation statistics)")
    print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer data with controlled duplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 500 customers with default duplication levels
  python generate_customer_data.py --n 500 --out experiments_output

  # Custom duplication levels
  python generate_customer_data.py --n 500 --out experiments_output --levels 0,25,50,75,100

  # Different random seed
  python generate_customer_data.py --n 500 --out experiments_output --seed 123

Output Structure:
  experiments_output/
  ├── agent/
  │   └── customers_dup_{level}pct.jsonl    # Agent input (no customer_id)
  ├── eval/
  │   ├── canonical_customers.json          # Ground truth
  │   └── cluster_map_{level}pct.json       # Skeleton key
  └── metadata/
      └── generation_stats_{level}pct.json  # Statistics
        """
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of base customers to generate (default: 500)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="experiments_output",
        help="Output directory (default: experiments_output)"
    )
    
    parser.add_argument(
        "--levels",
        type=str,
        default="0,10,20,30,40,50,75,100",
        help="Comma-separated duplication levels in percent (default: 0,10,20,30,40,50,75,100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Parse duplication levels
    duplication_levels = [int(x.strip()) for x in args.levels.split(",")]
    duplication_levels = sorted(set(duplication_levels))  # Remove duplicates and sort
    
    # Validate levels
    for level in duplication_levels:
        if level < 0 or level > 100:
            print(f"❌ Error: Duplication level {level} must be between 0 and 100")
            return 1
    
    # Convert output to Path
    output_dir = Path(args.out)
    
    # Generate datasets
    try:
        generate_all_datasets(
            n_customers=args.n,
            duplication_levels=duplication_levels,
            output_dir=output_dir,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


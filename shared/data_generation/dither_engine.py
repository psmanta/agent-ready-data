"""
Dither Engine — Shared Infrastructure
======================================
The Agentic Data Contract · Phase 1 · Shared Data Generation Layer

Applies controlled field-level perturbations to a clean customer dataset.
Used by Experiment 1b and any future experiment that requires dithering
as a data condition.

What is dithering?
------------------
Dithering introduces subtle errors into field values — not by duplicating
records (that's 1a) but by making individual field values quietly wrong.
The data looks plausible. The formats are valid. The values are in range.
But they no longer accurately reflect the customer they describe.

Two dither types are modelled:

  drift        — Asymmetric, directional change over time. last_purchase_days_ago
                 only grows. churn_risk tends upward for disengaged accounts.
                 Modelled on what happens to CRM data that isn't actively maintained.

  entry_error  — Modern automation artifacts. Not character-level typos (those
                 are rare in contemporary enterprise pipelines) but the failures
                 that automated systems produce: date format mismatches, unit
                 conversion misses, default value persistence, field truncation.

Architecture
------------
The engine is parameterised around five axes, all designed to map directly
to hypotheses in the 1b design document:

  fields          — which fields to dither (list of field names)
  magnitude       — how much to dither (scalar, per-field dict, or named tier)
  dither_type     — which error model(s) to apply (list)
  correlated      — drift fields together or independently
  segment_filter  — which customer segments to dither (list or None = all)

Usage:
    from dither_engine import DitherEngine, DitherConfig

    config = DitherConfig(
        fields=["churn_risk_score", "last_purchase_days_ago"],
        magnitude=0.15,
        dither_type=["drift"],
        correlated=True,
        segment_filter=["medium_value", "at_risk"],
        seed=42,
        condition_id="h1_churn_drift_15pct",
    )
    engine = DitherEngine(config)
    dithered_customers = engine.apply(base_customers)
"""

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ============================================================================
# FIELD METADATA
# ============================================================================

# Numeric fields: dithered by percentage magnitude
# drift_dir: +1 = tends to increase, -1 = tends to decrease, 0 = symmetric
NUMERIC_FIELD_META: Dict[str, Dict[str, Any]] = {
    "last_purchase_days_ago":    {"min": 1,    "max": 2160,   "drift_dir": +1, "type": "int"},
    "last_login_days_ago":       {"min": 0,    "max": 2160,   "drift_dir": +1, "type": "int"},
    "nps_score":                 {"min": 0,    "max": 10,     "drift_dir":  0, "type": "int"},
    "email_open_rate":           {"min": 0.0,  "max": 1.0,    "drift_dir": -1, "type": "float"},
    "churn_risk_score":          {"min": 0.0,  "max": 1.0,    "drift_dir": +1, "type": "float"},
    "fraud_risk_score":          {"min": 0.0,  "max": 1.0,    "drift_dir":  0, "type": "float"},
    "payment_failures":          {"min": 0,    "max": 20,     "drift_dir": +1, "type": "int"},
    "refund_rate":               {"min": 0.0,  "max": 1.0,    "drift_dir":  0, "type": "float"},
    "total_spend":               {"min": 0.01, "max": 500000, "drift_dir": +1, "type": "float"},
    "lifetime_value_estimate":   {"min": 0.01, "max": 500000, "drift_dir": +1, "type": "float"},
    "avg_order_value":           {"min": 0.01, "max": 50000,  "drift_dir":  0, "type": "float"},
    "total_purchases":           {"min": 1,    "max": 500,    "drift_dir": +1, "type": "int"},
    "purchase_frequency_days":   {"min": 1,    "max": 365,    "drift_dir": +1, "type": "int"},
    "support_tickets_open":      {"min": 0,    "max": 20,     "drift_dir": +1, "type": "int"},
    "support_tickets_closed":    {"min": 0,    "max": 100,    "drift_dir": +1, "type": "int"},
    "avg_resolution_time_hours": {"min": 1,    "max": 720,    "drift_dir": +1, "type": "int"},
    "tenure_months":             {"min": 1,    "max": 120,    "drift_dir":  0, "type": "int"},
}

# Categorical/identity fields: dithered by plausibility tier
CATEGORICAL_FIELD_META: Dict[str, List[str]] = {
    "name":                ["minimal", "moderate", "significant"],
    "email":               ["minimal", "moderate", "significant"],
    "phone":               ["minimal", "moderate", "significant"],
    "address":             ["minimal", "moderate"],
    "acquisition_channel": ["minimal", "moderate"],
}

# Entry error model targets
ENTRY_ERROR_MODELS = {
    "unit_conversion":     ["total_spend", "lifetime_value_estimate",
                            "avg_order_value", "avg_resolution_time_hours"],
    "default_persistence": ["nps_score", "email_open_rate", "churn_risk_score"],
    "date_format_mismatch":["last_purchase_days_ago", "last_login_days_ago",
                            "tenure_months"],
    "truncation":          ["name", "address", "email"],
}

# Named magnitude shortcuts
MAGNITUDE_TIERS = {
    "minimal":     0.05,
    "moderate":    0.15,
    "significant": 0.40,
    "extreme":     1.00,
}


# ============================================================================
# DITHER CONFIG
# ============================================================================

@dataclass
class DitherConfig:
    """
    Configuration for a single dither condition.
    Maps directly to the experimental axes in 1b_DESIGN.md.

    fields:
        List of field names to dither.

    magnitude:
        float scalar  — same magnitude for all fields (e.g. 0.15)
        dict          — per-field magnitudes (e.g. {"churn_risk_score": 0.15,
                        "last_purchase_days_ago": 0.10})
        str shortcut  — "minimal"(0.05), "moderate"(0.15),
                        "significant"(0.40), "extreme"(1.00)

    dither_type:
        List of error models. Options: "drift", "entry_error"

    correlated:
        True  = fields drift together in natural direction (H3 correlated arm)
        False = fields drift independently, breaking correlations (H3 uncorrelated arm)

    segment_filter:
        List of segment names to dither. None = all segments.
        e.g. ["high_value", "low_value"] — not medium_value or at_risk

    recompute_derived:
        True  (default) = recompute is_at_risk, recently_contacted_support
                          after dithering — record stays internally consistent
        False = leave derived fields as-is — creates internal contradictions
                Used for H3 internal consistency experiments.

    seed:          Random seed. Document in experiment metadata.
    condition_id:  Short label used in output filenames and reports.
    """
    fields:            List[str]
    magnitude:         Union[float, Dict[str, float], str] = 0.15
    dither_type:       List[str] = field(default_factory=lambda: ["drift"])
    correlated:        bool = True
    segment_filter:    Optional[List[str]] = None
    recompute_derived: bool = True
    seed:              int = 42
    condition_id:      str = "dither_condition"

    def __post_init__(self):
        if isinstance(self.magnitude, str):
            if self.magnitude not in MAGNITUDE_TIERS:
                raise ValueError(f"Unknown magnitude shortcut '{self.magnitude}'. "
                                 f"Valid: {list(MAGNITUDE_TIERS.keys())}")
            self.magnitude = MAGNITUDE_TIERS[self.magnitude]

        valid_types = {"drift", "entry_error"}
        for dt in self.dither_type:
            if dt not in valid_types:
                raise ValueError(f"Unknown dither_type '{dt}'. Valid: {valid_types}")

        all_known = set(NUMERIC_FIELD_META) | set(CATEGORICAL_FIELD_META)
        for f in self.fields:
            if f not in all_known:
                raise ValueError(f"Unknown field '{f}'. Check NUMERIC_FIELD_META "
                                 f"and CATEGORICAL_FIELD_META.")

    def get_magnitude(self, field_name: str) -> float:
        if isinstance(self.magnitude, dict):
            return self.magnitude.get(field_name, 0.15)
        return float(self.magnitude)

    def get_categorical_tier(self, field_name: str) -> str:
        mag = self.get_magnitude(field_name)
        if mag <= 0.05:   return "minimal"
        elif mag <= 0.15: return "moderate"
        else:             return "significant"


# ============================================================================
# FIELD DITHERERS
# ============================================================================

def _dither_numeric_drift(value, field_name, magnitude, direction, rng):
    meta = NUMERIC_FIELD_META[field_name]
    if direction == +1:
        sign = +1 if rng.random() < 0.85 else -1
    elif direction == -1:
        sign = -1 if rng.random() < 0.85 else +1
    else:
        sign = +1 if rng.random() < 0.5 else -1

    actual_magnitude = magnitude * rng.uniform(0.5, 1.5)
    new_value = value + (value * actual_magnitude * sign)
    new_value = max(meta["min"], min(meta["max"], new_value))

    return int(round(new_value)) if meta["type"] == "int" else round(new_value, 3)


def _dither_numeric_entry_error(value, field_name, magnitude, rng):
    meta = NUMERIC_FIELD_META[field_name]

    if field_name in ENTRY_ERROR_MODELS["unit_conversion"]:
        factors = [100, 0.01, 24, 1/24]
        rng.shuffle(factors)
        for factor in factors:
            candidate = value * factor
            if meta["min"] <= candidate <= meta["max"]:
                return round(candidate, 3) if meta["type"] == "float" else int(round(candidate))

    if field_name in ENTRY_ERROR_MODELS["default_persistence"]:
        defaults = {"nps_score": 5, "email_open_rate": 0.0, "churn_risk_score": 0.5}
        if field_name in defaults:
            return defaults[field_name]

    if field_name in ENTRY_ERROR_MODELS["date_format_mismatch"]:
        factor = rng.choice([3.5, 1/3.5])
        candidate = max(meta["min"], min(meta["max"], value * factor))
        return int(round(candidate)) if meta["type"] == "int" else round(candidate, 3)

    # Default: random non-directional perturbation
    sign = +1 if rng.random() < 0.5 else -1
    new_value = value + (value * magnitude * rng.uniform(0.5, 1.5) * sign)
    new_value = max(meta["min"], min(meta["max"], new_value))
    return int(round(new_value)) if meta["type"] == "int" else round(new_value, 3)


def _dither_email(email, tier, rng):
    try:
        local, domain = email.split("@")
    except ValueError:
        return email

    if tier == "minimal":
        if "." in local:
            return email.replace(".", "", 1)
        mid = len(local) // 2
        return f"{local[:mid]}.{local[mid:]}@{domain}"

    elif tier == "moderate":
        providers = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com"]
        new_domain = rng.choice([p for p in providers if p != domain])
        return f"{local}@{new_domain}"

    else:  # significant
        num = rng.randint(1000, 99999)
        handle = rng.choice(["user", "contact", "info", "hello", "me"])
        provider = rng.choice(["gmail.com", "yahoo.com", "hotmail.com"])
        return f"{handle}{num}@{provider}"


def _dither_name(name, tier, rng):
    if tier == "minimal":
        options = [name.lower(), name.upper(), name.title()]
        diff = [o for o in options if o != name]
        return diff[0] if diff else name.lower()
    elif tier == "moderate":
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return name
    else:
        parts = name.split()
        if len(parts) >= 2:
            rng.shuffle(parts)
            return " ".join(parts)
        return name[::-1]


def _dither_phone(phone, tier, rng):
    digits = re.sub(r'\D', '', phone)
    if tier == "minimal":
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return phone
    elif tier == "moderate":
        if len(digits) >= 10:
            new_last4 = str(rng.randint(1000, 9999))
            d = digits[:-4] + new_last4
            return f"({d[:3]}) {d[3:6]}-{d[6:10]}"
        return phone
    else:
        d = "".join([str(rng.randint(0, 9)) for _ in range(10)])
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"


def _dither_address(address, tier, rng):
    if tier == "minimal":
        for abbr, full in [("St,","Street,"),("Ave,","Avenue,"),("Blvd,","Boulevard,"),
                           ("Dr,","Drive,"),("Ln,","Lane,"),("Rd,","Road,")]:
            if abbr in address:
                return address.replace(abbr, full, 1)
        return address
    else:
        parts = address.split(" ", 1)
        if parts[0].isdigit():
            new_num = str(int(parts[0]) + rng.randint(-50, 50))
            if int(new_num) > 0:
                return f"{new_num} {parts[1]}"
        return address


def _dither_categorical(value, field_name, tier, rng):
    if field_name == "email":   return _dither_email(value, tier, rng)
    if field_name == "name":    return _dither_name(value, tier, rng)
    if field_name == "phone":   return _dither_phone(value, tier, rng)
    if field_name == "address": return _dither_address(value, tier, rng)
    return value


# ============================================================================
# DITHER ENGINE
# ============================================================================

class DitherEngine:
    """
    Applies controlled dither transformations to a customer dataset.

    Each call to apply() returns a new list — originals are not modified.
    Metadata fields (_dither_*) are stripped before writing agent input files.
    The agent never sees original values or dither metadata.
    """

    def __init__(self, config: DitherConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def apply(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for customer in customers:
            if self._should_dither(customer):
                dithered = self._apply_to_record(copy.deepcopy(customer))
            else:
                dithered = copy.deepcopy(customer)
                dithered["_dither_applied"]  = False
                dithered["_dither_fields"]   = []
                dithered["_dither_original"] = {}
                dithered["_dither_config"]   = self.config.condition_id
            results.append(dithered)
        return results

    def _should_dither(self, customer):
        if self.config.segment_filter is None:
            return True
        return customer.get("customer_segment") in self.config.segment_filter

    def _apply_to_record(self, customer):
        original_values = {}
        changed_fields = []
        directions = self._resolve_directions()

        for field_name in self.config.fields:
            if field_name not in customer:
                continue
            original_value = customer[field_name]
            magnitude = self.config.get_magnitude(field_name)
            current_value = original_value

            for dither_type in self.config.dither_type:
                current_value = self._dither_field(
                    current_value, field_name, magnitude,
                    dither_type, directions.get(field_name, 0)
                )

            if current_value != original_value:
                original_values[field_name] = original_value
                customer[field_name] = current_value
                changed_fields.append(field_name)

        if self.config.recompute_derived and changed_fields:
            self._recompute_derived(customer)

        customer["_dither_applied"]  = True
        customer["_dither_fields"]   = changed_fields
        customer["_dither_original"] = original_values
        customer["_dither_config"]   = self.config.condition_id
        return customer

    def _dither_field(self, value, field_name, magnitude, dither_type, direction):
        if field_name in NUMERIC_FIELD_META:
            if dither_type == "drift":
                return _dither_numeric_drift(value, field_name, magnitude, direction, self.rng)
            elif dither_type == "entry_error":
                return _dither_numeric_entry_error(value, field_name, magnitude, self.rng)
        elif field_name in CATEGORICAL_FIELD_META:
            tier = self.config.get_categorical_tier(field_name)
            return _dither_categorical(value, field_name, tier, self.rng)
        return value

    def _resolve_directions(self):
        directions = {}
        for f in self.config.fields:
            if f in NUMERIC_FIELD_META:
                nat = NUMERIC_FIELD_META[f]["drift_dir"]
                if self.config.correlated:
                    directions[f] = nat
                else:
                    directions[f] = nat if self.rng.random() > 0.5 else -nat if nat != 0 else 0
            else:
                directions[f] = 0
        return directions

    def _recompute_derived(self, customer):
        if "churn_risk_score" in customer:
            customer["is_at_risk"] = customer["churn_risk_score"] >= 0.60
        if "support_tickets_open" in customer:
            customer["recently_contacted_support"] = customer["support_tickets_open"] >= 1


# ============================================================================
# PREDEFINED CONDITIONS — maps to 1b hypotheses
# ============================================================================

def build_h1_conditions(seed: int = 42) -> List[DitherConfig]:
    """H1 — Field Importance: 5 individual top fields + identity aggregate + behavioral aggregate"""
    h4_top5 = ["last_purchase_days_ago", "churn_risk_score", "nps_score",
                "lifetime_value_estimate", "support_tickets_open"]
    conditions = [
        DitherConfig(fields=[f], magnitude=0.15, dither_type=["drift"],
                     correlated=True, seed=seed+i, condition_id=f"h1_individual_{f}")
        for i, f in enumerate(h4_top5)
    ]
    conditions.append(DitherConfig(
        fields=["name", "email", "phone"], magnitude=0.15, dither_type=["drift"],
        correlated=False, seed=seed+10, condition_id="h1_aggregate_identity"))
    conditions.append(DitherConfig(
        fields=["email_open_rate", "avg_order_value", "tenure_months",
                "payment_failures", "fraud_risk_score"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed+11, condition_id="h1_aggregate_behavioral"))
    return conditions


def build_h2_conditions(field: str = "churn_risk_score", seed: int = 42) -> List[DitherConfig]:
    """H2 — Magnitude Effects: four magnitude levels for one field"""
    return [
        DitherConfig(fields=[field], magnitude=mag, dither_type=["drift"],
                     correlated=True, seed=seed+i,
                     condition_id=f"h2_{field}_mag{int(mag*100)}pct")
        for i, mag in enumerate([0.05, 0.15, 0.40, 1.00])
    ]


def build_h3_conditions(seed: int = 42) -> List[DitherConfig]:
    """H3 — Internal Consistency: 3 pairs + 1 triplet, correlated vs uncorrelated"""
    field_sets = [
        (["churn_risk_score", "last_purchase_days_ago"], "pair1_churn_purchase"),
        (["total_spend", "lifetime_value_estimate"],     "pair2_spend_ltv"),
        (["support_tickets_open", "avg_resolution_time_hours"], "pair3_support"),
        (["total_spend", "last_purchase_days_ago", "support_tickets_open"], "triplet"),
    ]
    conditions = []
    for i, (fields, name) in enumerate(field_sets):
        for j, correlated in enumerate([True, False]):
            suffix = "correlated" if correlated else "uncorrelated"
            conditions.append(DitherConfig(
                fields=fields, magnitude=0.15, dither_type=["drift"],
                correlated=correlated,
                recompute_derived=not correlated,
                seed=seed + i*2 + j,
                condition_id=f"h3_{name}_{suffix}"))
    return conditions


def build_h4_conditions(field: str = "churn_risk_score", seed: int = 42) -> List[DitherConfig]:
    """H4 — Dither Type: drift vs entry_error at matched magnitude"""
    return [
        DitherConfig(fields=[field], magnitude=0.15, dither_type=[dt],
                     correlated=True, seed=seed+i,
                     condition_id=f"h4_{field}_{dt}")
        for i, dt in enumerate(["drift", "entry_error"])
    ]


# ============================================================================
# FILE OUTPUT UTILITIES
# ============================================================================

def save_dithered_condition(
    customers: List[Dict[str, Any]],
    output_path: Path,
    strip_metadata: bool = True,
) -> None:
    """
    Save dithered customers to JSONL.
    strip_metadata=True  → agent input file (no _dither_* fields)
    strip_metadata=False → evaluation reference file (keeps originals)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for customer in customers:
            record = copy.deepcopy(customer)
            if strip_metadata:
                for key in list(record.keys()):
                    if key.startswith("_dither"):
                        del record[key]
            f.write(json.dumps(record, default=str) + "\n")

    n_dithered = sum(1 for c in customers if c.get("_dither_applied", False))
    print(f"  Saved: {output_path} ({len(customers)} records, {n_dithered} dithered)")


def save_dither_reference(customers: List[Dict[str, Any]], output_path: Path) -> None:
    """Save evaluation reference file with dither metadata intact."""
    save_dithered_condition(customers, output_path, strip_metadata=False)

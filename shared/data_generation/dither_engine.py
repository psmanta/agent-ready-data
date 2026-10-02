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

# Boolean fields: dithered by flip probability, not percentage magnitude.
# magnitude directly IS the probability a given customer's boolean gets
# flipped — 0.15 magnitude means a 15% chance of flip, full stop. This
# keeps "magnitude" meaning the same thing (frequency/severity of
# corruption) across every field type in the engine, rather than scaling
# flip likelihood by how many alternative values exist.
BOOLEAN_FIELDS: List[str] = [
    "is_vip",
    "has_active_subscription",
    "has_pending_order",
]

# Protected fields: fields the engine will NEVER dither directly, for two
# structurally distinct reasons documented here so the reason is visible
# to anyone reading this file, not just in the design amendment.
#
# DERIVED fields are computed FROM other fields after generation (and
# re-derived after any dither, per _recompute_derived below). Dithering
# a derived field directly would create an incoherent record — e.g.
# is_at_risk=True with a churn_risk_score that doesn't support it — and
# would confound any experiment without a dedicated hypothesis built
# specifically to study derived-field mismatch, which does not currently
# exist. See RESEARCH_NOTES.md / 1b amendment for full reasoning.
#
# UPSTREAM fields are used DURING generation to condition the sampling
# distributions of OTHER fields (customer_segment determines the ranges
# total_spend, churn_risk_score, etc. are drawn from). Dithering the
# upstream field itself — changing the label after the profile was
# already generated to match the ORIGINAL label — breaks the
# segment/profile relationship from the top down with no defined
# semantics. This is different from, and excluded in favor of, dithering
# the downstream numeric fields far enough that they no longer match
# their original segment's typical range (the "segment/profile mismatch"
# analysis lens) — that is a valid and interesting bottom-up effect,
# computed as an analysis enrichment, not built as a dither mechanism.
PROTECTED_FIELDS: Dict[str, str] = {
    "is_at_risk":                 "derived",   # computed from churn_risk_score
    "recently_contacted_support": "derived",   # computed from support_tickets_open
    "customer_segment":           "upstream",  # conditions other fields' sampling
}

# Entry error model targets
# H4's plausible/implausible operator matrix — locked design, see
# h4_working_doc.md. Distinct from ENTRY_ERROR_MODELS below (which is
# generic, used by other conditions on other fields) — H4 needs
# explicit, individually-designed operators for exactly these 3 fields,
# not the generic dispatch.
H4_OPERATOR_MATRIX = {
    "churn_risk_score": {
        "plausible":   {"up": ("constant", 0.85),  "down": ("constant", 0.15)},
        "implausible": {"up": ("multiply", 100.0), "down": ("multiply", -1.0)},
    },
    "total_spend": {
        "plausible":   {"up": ("multiply", 2.0),   "down": ("multiply", 0.1)},
        "implausible": {"up": ("multiply", 100.0), "down": ("multiply", -1.0)},
    },
    "tenure_months": {
        "plausible":   {"up": ("multiply", 1.3),   "down": ("multiply", 1/1.3)},
        "implausible": {"up": ("multiply", 30.0),  "down": ("multiply", -1.0)},
    },
}


def _dither_h4_entry_error(value, field_name, plausibility, rng):
    """
    H4's plausible/implausible entry-error operator. Direction is a fair
    coin per customer, drawn here purely to select WHICH transform to
    apply — the coin's label is intentionally NOT what gets logged as
    this customer's direction. A constant-target operator (churn_risk_score's
    plausible resets) can produce the opposite of its "intended" direction
    for a customer who already sits on the far side of the target (e.g.
    orig=0.92, coin says "up" -> target 0.85 -> actually a decrease).
    Verified: roughly 20% of customers sit far enough from the midpoint
    for this to matter in practice. The caller derives the REAL direction
    from the observed before/after values, not from this coin.

    Plausible arm respects the field's normal min/max bounds -- that
    contract-respecting property is what makes it "plausible" at all.
    Implausible arm skips the bounds clip entirely (the caller must pass
    allow_out_of_bounds=True for this dither_type).

    Returns (new_value, operator_label, mechanism_style). mechanism_style
    is "constant" or "personalized" -- tracked as metadata because
    churn_risk_score's plausible cells are unavoidably constant (a real
    default-persistence bug IS a fixed value) while its implausible cells
    are naturally personalized (a real scale-mismatch bug multiplies
    whatever the true value is) -- an asymmetry that's a property of the
    real failure modes being modeled, not an engine inconsistency, and is
    tracked rather than forced into false uniformity.
    """
    coin = "up" if rng.random() < 0.5 else "down"
    style, param = H4_OPERATOR_MATRIX[field_name][plausibility][coin]

    if style == "constant":
        new_value = param
    else:
        new_value = value * param

    meta = NUMERIC_FIELD_META[field_name]
    if plausibility == "plausible":
        new_value = max(meta["min"], min(meta["max"], new_value))
    # implausible: no clipping -- caller has set allow_out_of_bounds

    new_value = int(round(new_value)) if meta["type"] == "int" else round(new_value, 4)
    operator_label = f"{plausibility}_{style}_{coin}"
    return new_value, operator_label, style


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
    coupling_signs:    Optional[Dict[str, int]] = None
    entry_error_plausibility: Optional[str] = None  # "plausible" or "implausible" -- H4 only
    seed:              int = 42
    condition_id:      str = "dither_condition"

    def __post_init__(self):
        if isinstance(self.magnitude, str):
            if self.magnitude not in MAGNITUDE_TIERS:
                raise ValueError(f"Unknown magnitude shortcut '{self.magnitude}'. "
                                 f"Valid: {list(MAGNITUDE_TIERS.keys())}")
            self.magnitude = MAGNITUDE_TIERS[self.magnitude]

        valid_types = {"drift", "entry_error", "h4_entry_error"}
        for dt in self.dither_type:
            if dt not in valid_types:
                raise ValueError(f"Unknown dither_type '{dt}'. Valid: {valid_types}")

        all_known = set(NUMERIC_FIELD_META) | set(CATEGORICAL_FIELD_META) | set(BOOLEAN_FIELDS)
        for f in self.fields:
            if f in PROTECTED_FIELDS:
                reason = PROTECTED_FIELDS[f]
                if reason == "derived":
                    raise ValueError(
                        f"'{f}' is a derived field (computed from another field "
                        f"after generation) and cannot be dithered directly. "
                        f"Dithering it would create an internally incoherent "
                        f"record with no defined semantics. See PROTECTED_FIELDS."
                    )
                elif reason == "upstream":
                    raise ValueError(
                        f"'{f}' is an upstream field that conditions the sampling "
                        f"distributions of other fields during generation. "
                        f"Dithering it directly breaks the segment/profile "
                        f"relationship from the top down with no defined "
                        f"semantics. To study segment/profile mismatch, dither "
                        f"the downstream numeric fields far enough that they no "
                        f"longer match this field's original value — see the "
                        f"segment/profile mismatch analysis lens in the "
                        f"evaluator instead. See PROTECTED_FIELDS."
                    )
            if f not in all_known:
                raise ValueError(f"Unknown field '{f}'. Check NUMERIC_FIELD_META, "
                                 f"CATEGORICAL_FIELD_META, and BOOLEAN_FIELDS.")

        if self.entry_error_plausibility is not None:
            if self.entry_error_plausibility not in ("plausible", "implausible"):
                raise ValueError(
                    f"entry_error_plausibility must be 'plausible' or "
                    f"'implausible', got {self.entry_error_plausibility!r}")
            if self.dither_type != ["h4_entry_error"]:
                raise ValueError(
                    "entry_error_plausibility is only defined for "
                    "dither_type=['h4_entry_error']")
            bad = [f for f in self.fields if f not in H4_OPERATOR_MATRIX]
            if bad:
                raise ValueError(
                    f"entry_error_plausibility fields must be in "
                    f"H4_OPERATOR_MATRIX, got unsupported fields: {bad}")

        if self.coupling_signs is not None:
            if set(self.coupling_signs) != set(self.fields):
                raise ValueError(
                    f"coupling_signs keys {sorted(self.coupling_signs)} must "
                    f"exactly match fields {sorted(self.fields)}")
            bad = {f: s for f, s in self.coupling_signs.items() if s not in (+1, -1)}
            if bad:
                raise ValueError(f"coupling_signs values must be +1 or -1, got {bad}")
            non_numeric = [f for f in self.fields if f not in NUMERIC_FIELD_META]
            if non_numeric:
                raise ValueError(f"coupling requires numeric fields; got {non_numeric}")
            if self.dither_type != ["drift"]:
                raise ValueError("coupling is only defined for dither_type=['drift']")

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

def _dither_numeric_drift(value, field_name, magnitude, direction, rng, strict=False):
    """
    strict=False (default, all non-H3 conditions): direction is a BIAS —
        +1/-1 means 85% probability of moving that way, 0 means 50/50.
    strict=True (H3 coupled conditions only): direction is EXACT — the
        sign has already been decided upstream by the coupling logic in
        _resolve_directions(), and must not be re-randomized here, or the
        85/15 bias would leak back in and the two H3 arms would stop
        having matched marginal odds.

    MINIMUM ±1 STEP FOR INTEGER FIELDS (added 2026-08 after an exposure
    audit): drift is multiplicative, so a small count like 1 or 2 times
    ~15% rounds straight back to itself, and 0 times anything stays 0.
    At 15% magnitude, support_tickets_open changed for only 7.9% of
    customers and payment_failures for only 2.8% — versus ~100% for
    continuous fields — so drift rates across fields were silently
    measuring different amounts of exposure, not different agent
    sensitivity. Any perturbed integer now moves at least one unit in the
    drawn direction. Remaining limit, reported rather than hidden: a value
    already at its floor that is drawn to move DOWN (e.g. 0 tickets) is
    clipped and cannot change — the generator's validity report
    quantifies this per condition.
    """
    meta = NUMERIC_FIELD_META[field_name]
    if strict:
        if direction not in (+1, -1):
            raise ValueError(f"strict mode requires direction +1 or -1, got {direction}")
        sign = direction
    elif direction == +1:
        sign = +1 if rng.random() < 0.85 else -1
    elif direction == -1:
        sign = -1 if rng.random() < 0.85 else +1
    else:
        sign = +1 if rng.random() < 0.5 else -1

    actual_magnitude = magnitude * rng.uniform(0.5, 1.5)
    new_value = value + (value * actual_magnitude * sign)
    new_value = max(meta["min"], min(meta["max"], new_value))

    if meta["type"] == "int":
        new_int = int(round(new_value))
        if new_int == value:
            new_int = int(max(meta["min"], min(meta["max"], value + sign)))
        blocked = new_int == value
        return new_int, blocked
    return round(new_value, 3), False


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

    # Default: random non-directional perturbation. Same minimum-step
    # fix as _dither_numeric_drift (added 2026-08) — this fallback is the
    # same multiplicative computation and had the identical bug: a small
    # integer times ~15% rounds back to itself. Found by checking whether
    # the pattern that broke drift also existed elsewhere in the file.
    sign = +1 if rng.random() < 0.5 else -1
    new_value = value + (value * magnitude * rng.uniform(0.5, 1.5) * sign)
    new_value = max(meta["min"], min(meta["max"], new_value))
    if meta["type"] == "int":
        new_int = int(round(new_value))
        if new_int == value:
            new_int = int(max(meta["min"], min(meta["max"], value + sign)))
        return new_int
    return round(new_value, 3)


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


def _dither_boolean(value: bool, magnitude: float, rng: random.Random) -> bool:
    """
    Dither a boolean field by flip probability.

    magnitude IS the flip probability directly — 0.15 magnitude means this
    customer's boolean has a 15% chance of being flipped. This keeps
    "magnitude" meaning the same thing (how often/severely this customer's
    data gets corrupted) across every field type in the engine — numeric
    percentage drift, categorical plausibility tier, and boolean flip
    probability are all just different implementations of "magnitude
    controls how disruptive the corruption is," not three unrelated
    concepts wearing the same parameter name.

    For a true/false field there is only one other value to flip to, so
    unlike an N-way categorical field, "which value does it become" has
    no additional decision to make — it simply becomes not(value).
    """
    if rng.random() < magnitude:
        return not value
    return value


def _dither_acquisition_channel(value, tier, rng):
    """
    Single-select categorical: swap to a uniformly-chosen DIFFERENT
    channel (the rule locked earlier for single-select categoricals —
    once a value is corrupted, pick uniformly among the other N-1
    options). The minimal/moderate tier distinction has no natural
    meaning for an unordered channel label, so both tiers behave the
    same; documented rather than invented.

    Channel list imported from the base generator rather than
    duplicated, so the two can never drift out of sync.
    """
    from base_customer_generator import ACQUISITION_CHANNELS
    others = [c for c in ACQUISITION_CHANNELS if c != value]
    return rng.choice(others) if others else value


def _dither_categorical(value, field_name, tier, rng):
    if field_name == "email":   return _dither_email(value, tier, rng)
    if field_name == "name":    return _dither_name(value, tier, rng)
    if field_name == "phone":   return _dither_phone(value, tier, rng)
    if field_name == "address": return _dither_address(value, tier, rng)
    if field_name == "acquisition_channel":
        return _dither_acquisition_channel(value, tier, rng)
    # Fail loudly rather than silently return the value unchanged — a
    # silent fall-through here is exactly how acquisition_channel went
    # undithered at every magnitude without anything noticing.
    raise ValueError(
        f"No categorical dither implementation for '{field_name}'. It is "
        f"listed in CATEGORICAL_FIELD_META but has no branch here."
    )


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
                dithered["_dither_blocked"]  = []
                dithered["_dither_original"] = {}
                dithered["_dither_config"]   = self.config.condition_id
                dithered["_dither_operator"] = {}
                dithered["_dither_direction"] = {}
                dithered["_dither_mechanism_style"] = {}
            results.append(dithered)
        return results

    def _should_dither(self, customer):
        if self.config.segment_filter is None:
            return True
        return customer.get("customer_segment") in self.config.segment_filter

    def _apply_to_record(self, customer):
        original_values = {}
        changed_fields = []
        blocked_fields = []
        operators = {}          # field_name -> operator_label (H4 only)
        mechanism_styles = {}   # field_name -> "constant"/"personalized" (H4 only)
        directions_observed = {}  # field_name -> "up"/"down", DERIVED from
                                  # before/after values, never from the
                                  # coin that selected an operator -- see
                                  # _dither_h4_entry_error's docstring for
                                  # why the coin's label can't be trusted
                                  # directly (a constant-target operator can
                                  # move a customer the opposite way from
                                  # what the coin intended).
        directions = self._resolve_directions()

        for field_name in self.config.fields:
            if field_name not in customer:
                continue
            original_value = customer[field_name]
            magnitude = self.config.get_magnitude(field_name)
            current_value = original_value
            field_blocked = False
            field_operator = None
            field_mechanism_style = None

            for dither_type in self.config.dither_type:
                current_value, was_blocked, operator_label, mechanism_style = self._dither_field(
                    current_value, field_name, magnitude,
                    dither_type, directions.get(field_name, 0)
                )
                field_blocked = field_blocked or was_blocked
                field_operator = operator_label if operator_label is not None else field_operator
                field_mechanism_style = mechanism_style if mechanism_style is not None else field_mechanism_style

            if current_value != original_value:
                original_values[field_name] = original_value
                customer[field_name] = current_value
                changed_fields.append(field_name)
                if field_operator is not None:
                    operators[field_name] = field_operator
                    mechanism_styles[field_name] = field_mechanism_style
                    directions_observed[field_name] = "up" if current_value > original_value else "down"
            elif field_blocked:
                # Targeted for dithering, and structurally could not move —
                # value already sat at the boundary the drawn direction
                # pushed toward (e.g. payment_failures=0, direction=down).
                # Distinct from a field the mechanism simply failed to
                # move, which is what the validity check exists to catch.
                blocked_fields.append(field_name)

        if self.config.recompute_derived and changed_fields:
            self._recompute_derived(customer)

        customer["_dither_applied"]   = True
        customer["_dither_fields"]    = changed_fields
        customer["_dither_blocked"]   = blocked_fields
        customer["_dither_original"]  = original_values
        customer["_dither_config"]    = self.config.condition_id
        customer["_dither_operator"]  = operators
        customer["_dither_direction"] = directions_observed
        customer["_dither_mechanism_style"] = mechanism_styles
        return customer

    def _dither_field(self, value, field_name, magnitude, dither_type, direction):
        """Returns (new_value, was_blocked, operator_label, mechanism_style).
        was_blocked is only ever True/False for numeric drift (the only
        path where a structurally-impossible move applies). operator_label
        and mechanism_style are only ever non-None for H4's dedicated
        entry-error path -- every other path returns None for both, since
        no other dither_type tracks a specific named operator today."""
        if field_name in NUMERIC_FIELD_META:
            if dither_type == "drift":
                new_value, blocked = _dither_numeric_drift(
                    value, field_name, magnitude, direction, self.rng,
                    strict=self.config.coupling_signs is not None)
                return new_value, blocked, None, None
            elif dither_type == "entry_error":
                return _dither_numeric_entry_error(value, field_name, magnitude, self.rng), False, None, None
            elif dither_type == "h4_entry_error":
                new_value, operator_label, mechanism_style = _dither_h4_entry_error(
                    value, field_name, self.config.entry_error_plausibility, self.rng)
                return new_value, False, operator_label, mechanism_style
        elif field_name in CATEGORICAL_FIELD_META:
            tier = self.config.get_categorical_tier(field_name)
            return _dither_categorical(value, field_name, tier, self.rng), False, None, None
        elif field_name in BOOLEAN_FIELDS:
            # Boolean fields ignore dither_type (drift/entry_error) — a
            # flip is a flip regardless of "mechanism," since there's no
            # meaningful distinction between a boolean decaying over time
            # vs. an automation artifact flipping it. Both dither_type
            # values route to the same flip-probability logic.
            return _dither_boolean(value, magnitude, self.rng), False, None, None
        return value, False, None, None

    def _resolve_directions(self):
        """
        Two modes.

        COUPLED (coupling_signs set — H3 only). Replaces the original
        definition of "correlated" as "each field drifts in its own
        natural aging direction," which an audit showed does NOT produce
        coherent joint shifts: Pair 1 (churn+nps) was 52.5% vs 52.0%
        coherent across arms — identical — because nps has no natural
        direction, and the triplet's "correlated" arm was LESS coherent
        (10.8%) than its uncorrelated arm (27.3%), since pushing spend and
        churn both "up" contradicts their r=-0.58. The natural-direction
        mechanism also gave the arms different MARGINAL odds (85/15 vs
        50/50), confounding coherence with directional bias.

          correlated=True:  one fair coin per customer sets a shared
              direction; each field follows it times its coupling sign
              (its empirically verified correlation sign relative to the
              set). Fields move coherently with each other.
          correlated=False: each field gets its own independent fair
              coin. Coherence is left to chance.

        Both arms give every field exactly 50/50 odds of moving up or
        down, so the ONLY thing that differs between arms is whether the
        fields move together — which is the thing H3 tests. For reference
        sets (fields with no real relationship), "correlated" simply means
        synchronized, so any reference delta isolates the mechanical
        effect of synchronized movement — exactly what the GEE
        difference-in-differences subtracts.

        NATURAL (coupling_signs None — every other condition): unchanged
        original behavior, preserved so nothing outside H3 moves.
        """
        if self.config.coupling_signs is not None:
            if self.config.correlated:
                shared = +1 if self.rng.random() < 0.5 else -1
                return {f: shared * s for f, s in self.config.coupling_signs.items()}
            return {f: (+1 if self.rng.random() < 0.5 else -1) for f in self.config.fields}

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
# PREDEFINED CONDITIONS — maps to 1b hypotheses, per 1b_DESIGN_AMENDMENT_1.md
# ============================================================================
#
# Condition counts per the amendment (H4 reduced from 6 to 3 — see below):
#   H1: 14   H2: 12   H3: 15   H4: 6   H7: 4   H8a: 2   H8b: 0-1 (conditional)
#   Total: 50-51 (H1 +2 comparison fields; H3 +4 after the field-composition
#   correction, the payment_failures individual condition, and the reference
#   sets gaining the correlated arms the GEE DiD requires)

def build_h1_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H1 — Field Importance Validation (restructured per amendment).

    12 conditions: 5 individual H4-top-5 fields, 6 category-level
    conditions (field lists corrected against actual engine capability —
    see PROTECTED_FIELDS and the amendment's "A Note on Protected Fields
    and Boolean Support"), and 1 distributed condition.
    """
    conditions = []

    # --- 5 individual fields (1a's H4 top-5) ---
    h4_top5 = ["last_purchase_days_ago", "churn_risk_score", "nps_score",
               "lifetime_value_estimate", "support_tickets_open"]
    for i, f in enumerate(h4_top5):
        conditions.append(DitherConfig(
            fields=[f], magnitude=0.15, dither_type=["drift"], correlated=True,
            seed=seed + i, condition_id=f"h1_individual_{f}"))

    # --- 2 comparison-group individual fields (amendment: H1 12 -> 14) ---
    # email (Identity) and is_vip (Account Status) close the only two
    # categories with zero individually-tested fields. Both are predicted-
    # null COMPARISON fields for Question A, not top-5 fields. These were
    # in the amended design but missing from this builder until an audit
    # caught it — evaluate_h1.py loads both.
    for i, f in enumerate(["email", "is_vip"]):
        conditions.append(DitherConfig(
            fields=[f], magnitude=0.15, dither_type=["drift"], correlated=True,
            seed=seed + 10 + i, condition_id=f"h1_individual_{f}"))

    # --- 6 category-level conditions (engine-verified field lists) ---
    # dob excluded from identity — not currently dither-capable (no defined
    # magnitude/tier behavior for a date-of-birth field)
    conditions.append(DitherConfig(
        fields=["name", "email", "phone", "address"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 20, condition_id="h1_category_identity"))

    conditions.append(DitherConfig(
        fields=["total_purchases", "total_spend", "avg_order_value",
                "purchase_frequency_days", "last_purchase_days_ago",
                "lifetime_value_estimate"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 21, condition_id="h1_category_purchase_behavior"))

    conditions.append(DitherConfig(
        fields=["nps_score", "email_open_rate", "last_login_days_ago",
                "support_tickets_open", "support_tickets_closed",
                "avg_resolution_time_hours"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 22, condition_id="h1_category_engagement"))

    conditions.append(DitherConfig(
        fields=["churn_risk_score", "payment_failures", "fraud_risk_score",
                "refund_rate"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 23, condition_id="h1_category_risk_factors"))

    # customer_segment excluded (protected, upstream field)
    # preferred_categories excluded (variable-length list, deferred)
    conditions.append(DitherConfig(
        fields=["acquisition_channel", "tenure_months"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 24, condition_id="h1_category_segmentation"))

    # is_at_risk, recently_contacted_support excluded (protected, derived)
    conditions.append(DitherConfig(
        fields=["is_vip", "has_active_subscription", "has_pending_order"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 25, condition_id="h1_category_account_status"))

    # --- 1 distributed condition — deliberately excludes H4 top-5 ---
    conditions.append(DitherConfig(
        fields=["email", "avg_order_value", "last_login_days_ago",
                "refund_rate", "acquisition_channel", "has_pending_order"],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed + 30, condition_id="h1_distributed"))

    return conditions


def build_h2_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H2 — Magnitude Effects (restructured per amendment).

    12 conditions: 3 fields (churn_risk_score, total_spend, tenure_months)
    x 4 magnitude levels (5%, 15%, 40%, 100%).

    Seed scheme is deliberately structured (base + field_offset + mag_index)
    so that each field's mag15pct seed is stable and referenceable — H4's
    drift-type conditions for these same 3 fields are NOT regenerated
    separately; they reuse these exact mag15pct conditions by condition_id.
    See build_h4_conditions() docstring.
    """
    fields = ["churn_risk_score", "total_spend", "tenure_months"]
    magnitudes = [0.05, 0.15, 0.40, 1.00]

    conditions = []
    for field_idx, field in enumerate(fields):
        for mag_idx, mag in enumerate(magnitudes):
            conditions.append(DitherConfig(
                fields=[field], magnitude=mag, dither_type=["drift"],
                correlated=True,
                seed=seed + field_idx * 10 + mag_idx,
                condition_id=f"h2_{field}_mag{int(mag*100)}pct"))
    return conditions


def build_h3_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H3 — Internal Consistency (restructured; field composition corrected
    and coupling mechanism redefined, 2026-08).

    15 conditions:
      - 4 test sets x 2 arms (8): three pairs + one triplet, each field set
        verified against the base generator to correlate through genuine
        common-cause structure (or, for Pair 2, documented direct
        derivation). Original Pair 1, Pair 3, and the triplet's third
        member were found to have r~0 — indistinguishable from the null
        reference — and were replaced. See "A Note on Assumed vs. Forced
        Correlation" in the design amendment.
      - 3 individual-field conditions completing 2x2 directionality where
        no individual condition exists elsewhere (avg_resolution_time_hours,
        refund_rate, payment_failures). All other H3 fields already have
        individual conditions in H1/H2.
      - 2 reference sets x 2 arms (4): a 2-field and a 3-field null set
        (all pairwise |r| < 0.022, verified). Both arms are REQUIRED —
        the GEE difference-in-differences estimates the group x arm
        interaction, which is not estimable without a reference-correlated
        cell. The 3-field reference matches the triplet's perturbation
        volume (k=3), removing the confound of comparing a 3-field
        perturbation against a 2-field reference.

    Coupling: every test and reference set uses coupling_signs (see
    DitherEngine._resolve_directions) — correlated arm = shared direction
    times each field's verified correlation sign; uncorrelated arm =
    independent fair coins; identical 50/50 marginal odds in both arms.

    recompute_derived=True in BOTH arms. The original design left derived
    fields stale only in the uncorrelated arm, which added a second
    difference between arms (e.g. is_at_risk inconsistent with churn) and
    contradicted the later protected-fields decision that no hypothesis
    studies derived-field mismatch.
    """
    # Coupling signs relative to a shared latent direction, from verified
    # correlations (n=2000, seed=42):
    #   churn_risk_score <-> nps_score:              r = -0.72
    #   total_spend <-> lifetime_value_estimate:     r = +0.99 (direct derivation)
    #   support_tickets_open <-> payment_failures:   r = +0.51
    #   triplet: spend<->tickets -0.28, spend<->churn -0.58, tickets<->churn +0.59
    #     (sign structure is internally consistent: "disengaging account" =
    #      spend down, tickets up, churn up)
    #   references: all pairwise |r| < 0.022 -> synchronized (+1 all)
    SETS = [
        ("pair1_churn_nps",        {"churn_risk_score": +1, "nps_score": -1}),
        ("pair2_spend_ltv",        {"total_spend": +1, "lifetime_value_estimate": +1}),
        ("pair3_support_payment",  {"support_tickets_open": +1, "payment_failures": +1}),
        ("triplet",                {"total_spend": -1, "support_tickets_open": +1,
                                    "churn_risk_score": +1}),
        ("reference_pair",         {"avg_resolution_time_hours": +1, "refund_rate": +1}),
        ("reference_triplet",      {"avg_resolution_time_hours": +1, "refund_rate": +1,
                                    "tenure_months": +1}),
    ]

    conditions = []
    for i, (name, signs) in enumerate(SETS):
        for j, correlated in enumerate([True, False]):
            suffix = "correlated" if correlated else "uncorrelated"
            conditions.append(DitherConfig(
                fields=list(signs), magnitude=0.15, dither_type=["drift"],
                correlated=correlated, coupling_signs=dict(signs),
                recompute_derived=True,
                seed=seed + i * 2 + j,
                condition_id=f"h3_{name}_{suffix}"))

    for k, f in enumerate(["avg_resolution_time_hours", "refund_rate", "payment_failures"]):
        conditions.append(DitherConfig(
            fields=[f], magnitude=0.15, dither_type=["drift"], correlated=True,
            seed=seed + 100 + k, condition_id=f"h3_individual_{f}"))

    return conditions


def build_h4_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H4 — Dither Type Effects (restructured per h4_working_doc.md's full
    design session: matched-definition decision, direction-confound fix,
    mechanism-style tracking).

    6 NEW conditions: one plausible and one implausible entry-error
    condition per field, across the same 3 fields H2 uses
    (churn_risk_score, total_spend, tenure_months). Each condition
    internally applies a fair coin per customer to select an up or down
    variant from H4_OPERATOR_MATRIX — direction is NOT a separate
    condition axis; it's derived per-customer from the observed
    before/after values (see _apply_to_record), never logged from the
    coin itself, since a constant-target operator can produce the
    opposite of its "intended" direction for a customer already on the
    far side of the target.

    The "drift" arm of H4's comparison is NOT regenerated — it is served
    directly by H2's existing 15%-magnitude conditions for these same
    fields (h2_churn_risk_score_mag15pct, h2_total_spend_mag15pct,
    h2_tenure_months_mag15pct). Generating a separate "drift" condition
    here would mean running the agent against two statistically-
    equivalent-but-not-identical datasets purely to answer the same
    comparison twice. The evaluator's H4 analysis references
    h2_{field}_mag15pct's decisions directly as the "drift" data point
    for each field, alongside these 6 new plausible/implausible
    conditions.

    Locked at 100% prevalence (matches H2's drift arm, maximizes McNemar's
    power) with realistic, uncapped intensity -- see "The matched
    decision" in h4_working_doc.md for why matched-prevalence and
    matched-MAPE alternatives were both rejected.
    """
    fields = ["churn_risk_score", "total_spend", "tenure_months"]
    conditions = []
    for i, f in enumerate(fields):
        for j, plausibility in enumerate(["plausible", "implausible"]):
            conditions.append(DitherConfig(
                fields=[f], magnitude=0.15, dither_type=["h4_entry_error"],
                correlated=True, entry_error_plausibility=plausibility,
                seed=seed + i * 2 + j,
                condition_id=f"h4_{f}_{plausibility}"))
    return conditions



def build_h7_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H7 — Breadth Effects (new hypothesis per amendment).

    4 conditions: a fixed accumulation ladder (1 field -> 3 -> 6 -> 11),
    each step adding to the previous rather than swapping to unrelated
    fields, so results form a genuine accumulation curve.

    h7_breadth_all's 11-field list (locked at build time, per the
    amendment's deferred placeholder) deliberately spans all six prompt
    categories rather than concentrating within the fields already used
    in the 6-field step — so "breadth" measures touching many kinds of
    data, not just piling up more fields within the same category:
      Risk:              churn_risk_score
      Purchase Behavior: last_purchase_days_ago, lifetime_value_estimate,
                         total_spend
      Engagement:        nps_score, support_tickets_open,
                         avg_resolution_time_hours
      Identity:          email
      Segmentation:      tenure_months, acquisition_channel
      Account Status:    is_vip
    """
    return [
        DitherConfig(
            fields=["churn_risk_score"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed, condition_id="h7_breadth_1field"),

        DitherConfig(
            fields=["churn_risk_score", "last_purchase_days_ago", "nps_score"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed + 1, condition_id="h7_breadth_3fields"),

        DitherConfig(
            fields=["churn_risk_score", "last_purchase_days_ago", "nps_score",
                    "lifetime_value_estimate", "support_tickets_open", "total_spend"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed + 2, condition_id="h7_breadth_6fields"),

        DitherConfig(
            fields=["churn_risk_score", "last_purchase_days_ago", "nps_score",
                    "lifetime_value_estimate", "support_tickets_open", "total_spend",
                    "email", "tenure_months", "acquisition_channel", "is_vip",
                    "avg_resolution_time_hours"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed + 3, condition_id="h7_breadth_all"),
    ]


def build_h8a_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    H8a — Category-Level Interaction (new hypothesis per amendment).

    2 conditions. Field lists mirror the corresponding H1 category
    conditions exactly, so the "alone" comparison points for the additive-
    baseline formula come directly from h1_category_identity,
    h1_category_account_status, h1_category_purchase_behavior, and
    h1_category_risk_factors — no separate individual-category conditions
    needed here.

    Pair 1 (Identity + Account Status): predicted null / negative control.
    Pair 2 (Purchase Behavior + Risk Factors): predicted possible
    super-additive interaction.
    """
    return [
        DitherConfig(
            fields=["name", "email", "phone", "address",
                    "is_vip", "has_active_subscription", "has_pending_order"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed, condition_id="h8a_pair1_identity_account_status"),

        DitherConfig(
            fields=["total_purchases", "total_spend", "avg_order_value",
                    "purchase_frequency_days", "last_purchase_days_ago",
                    "lifetime_value_estimate",
                    "churn_risk_score", "payment_failures",
                    "fraud_risk_score", "refund_rate"],
            magnitude=0.15, dither_type=["drift"], correlated=False,
            seed=seed + 1, condition_id="h8a_pair2_purchase_risk"),
    ]


def build_h8b_condition(field_a: str, field_b: str, seed: int = 42) -> DitherConfig:
    """
    H8b — Field-Level Interaction (new hypothesis per amendment; CONDITIONAL).

    Unlike every other build_h*_conditions() function, this does not run
    automatically as part of full-pipeline generation. Per the amendment's
    locked decision rule: h8a_pair2_purchase_risk must be run and analyzed
    FIRST. Only if its combined drift rate exceeds the additive-baseline
    prediction (rate_A + rate_B - rate_A*rate_B) by more than 20% relative
    excess does this condition get generated at all — and field_a/field_b
    are then the empirically top-drifting individual fields from H1/H2
    within Purchase Behavior and Risk Factors respectively, NOT a
    pre-registered guess.

    This function takes the winning fields as parameters rather than
    hard-coding them, since the whole point of the decision rule is that
    the fields are determined by H8a's results, not chosen in advance.

    Args:
        field_a: top-drifting field from Purchase Behavior (from H1/H2 data)
        field_b: top-drifting field from Risk Factors (from H1/H2 data)
        seed: random seed for this condition
    """
    return DitherConfig(
        fields=[field_a, field_b],
        magnitude=0.15, dither_type=["drift"], correlated=False,
        seed=seed,
        condition_id=f"h8b_{field_a}_{field_b}")


def build_all_conditions(seed: int = 42) -> List[DitherConfig]:
    """
    Assemble every unconditional condition (H1, H2, H3, H4, H7, H8a) into
    one list. H8b is deliberately excluded — it is conditional on H8a's
    analyzed results and must be generated separately, later, by the
    orchestration script (generate_dithered_data.py) after checking the
    additive-baseline threshold.

    Returns 53 conditions total (14+12+15+6+4+2). H8b (0-1 more) is
    handled outside this function.
    """
    return (
        build_h1_conditions(seed=seed)
        + build_h2_conditions(seed=seed)
        + build_h3_conditions(seed=seed)
        + build_h4_conditions(seed=seed)
        + build_h7_conditions(seed=seed)
        + build_h8a_conditions(seed=seed)
    )


def validate_condition_ids_unique(conditions: List[DitherConfig]) -> None:
    """
    Sanity check: no two conditions share a condition_id. Run this before
    generating any data — a collision would silently overwrite one
    condition's output files with another's.
    """
    ids = [c.condition_id for c in conditions]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dupes = [id_ for id_, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate condition_id(s) found: {dupes}")


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

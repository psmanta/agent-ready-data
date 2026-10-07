#!/usr/bin/env python3
"""
Field redundancy table -- Experiment 1b (H4 redundancy hypothesis; reusable by H5).

Which dithered fields have a sibling in the same record that an agent could
cross-check against, and does the engine make that sibling agree with the
corrupted value? This is a DESIGN artifact, derived only from the data
generator's field formulas and the engine's propagation list, and fixed BEFORE
the full run. It must never be edited after seeing which fields produced
detections: that would make the redundancy hypothesis circular.

Relationship tiers (strongest first):
  strict       the sibling is an exact deterministic function of the field (or
               vice versa) in the generator. A contradiction is unambiguous.
  bounded      a hard constraint (inequality or implication) holding in every
               clean record, inferable from what the fields mean.
  approximate  a generator formula with a bounded random factor; checkable as
               a ratio band.
  soft         customer_segment sets a hard band on the field in the generator,
               but the band is never stated to the agent: only general
               expectations ("high-value customers rarely churn") can be used.
  (none)       no generator-defined link.

Propagation: with recompute_derived=True (the default for every condition
except the two isolated-churn arms) the engine recomputes some siblings after
dithering, so they AGREE with the corrupted value and the contradiction is
removed. A sibling the engine recomputes is "propagated"; the remaining
siblings are "uncorrected", and only those can expose the corruption.

A second, independent attribute is recorded per field: whether the agent's system prompt
STATES a value range for it (only nps_score, email_open_rate, churn_risk_score and
fraud_risk_score). It matters for H4: an "implausible" churn value violates a range the
agent was told, while an "implausible" total_spend or tenure_months breaks no stated rule,
so field-to-field differences in implausible-value handling are partly confounded with
whether a range was stated. The ranges are parsed from the agent file itself, so they
cannot drift from what the agent is actually told.

Direction matters: strict relations expose every change to the field, but bounded,
approximate and soft relations expose a corruption only when it pushes the value
outside the region the relationship allows (e.g. support_tickets_open dithered
DOWNWARD never violates closed >= open; is_vip flipped to False never violates
'is_vip implies high_value'). The table says which siblings EXIST, not that every
corruption is exposed by them.

Run `python3 field_redundancy.py` to print the table and verify every claim
against the generator and the engine (exit code 1 on any failure).
"""
import re
import sys
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "shared" / "data_generation"))

TIER_ORDER = ("strict", "bounded", "approximate", "soft")


class Edge(NamedTuple):
    a: str
    b: str
    tier: str
    rule: str
    group: Optional[str] = None   # edges that only work jointly (three-way relations)


# Every field the 55 conditions can dither. Verified against the engine at run time.
DITHERED_FIELDS = (
    "acquisition_channel", "address", "avg_order_value", "avg_resolution_time_hours",
    "churn_risk_score", "email", "email_open_rate", "fraud_risk_score",
    "has_active_subscription", "has_pending_order", "is_vip", "last_login_days_ago",
    "last_purchase_days_ago", "lifetime_value_estimate", "name", "nps_score",
    "payment_failures", "phone", "purchase_frequency_days", "refund_rate",
    "support_tickets_closed", "support_tickets_open", "tenure_months",
    "total_purchases", "total_spend",
)

SEGMENT_DRIVEN = ("total_purchases", "avg_order_value", "nps_score", "churn_risk_score",
                  "email_open_rate", "support_tickets_open", "payment_failures", "fraud_risk_score")

EDGES: List[Edge] = [
    # --- strict ---
    Edge("tenure_months", "account_created_date", "strict", "account_created_date = snapshot - 30 * tenure_months days"),
    Edge("last_purchase_days_ago", "last_purchase_date", "strict", "last_purchase_date = snapshot - last_purchase_days_ago days"),
    Edge("churn_risk_score", "is_at_risk", "strict", "is_at_risk = churn_risk_score >= 0.60"),
    Edge("support_tickets_open", "recently_contacted_support", "strict", "recently_contacted_support = support_tickets_open >= 1"),
    Edge("name", "email", "strict", "email = name lowercased, spaces -> '.', apostrophes removed, + '@example.com'"),
    # --- bounded ---
    Edge("support_tickets_open", "support_tickets_closed", "bounded", "support_tickets_closed >= support_tickets_open"),
    Edge("last_purchase_days_ago", "tenure_months", "bounded", "1 <= last_purchase_days_ago <= min(365, 30 * tenure_months)"),
    Edge("last_login_days_ago", "last_purchase_days_ago", "bounded", "max(0, purchase - 7) <= login <= purchase + 30"),
    Edge("last_login_days_ago", "tenure_months", "bounded", "last_login_days_ago <= 30 * tenure_months"),
    Edge("is_vip", "customer_segment", "bounded", "is_vip implies customer_segment == high_value"),
    # --- approximate ---
    Edge("total_spend", "total_purchases", "approximate", "total_spend ~ total_purchases * avg_order_value * U(0.90, 1.10)", "spend_triad"),
    Edge("total_spend", "avg_order_value", "approximate", "total_spend ~ total_purchases * avg_order_value * U(0.90, 1.10)", "spend_triad"),
    Edge("total_purchases", "avg_order_value", "approximate", "total_spend ~ total_purchases * avg_order_value * U(0.90, 1.10)", "spend_triad"),
    Edge("total_spend", "lifetime_value_estimate", "approximate", "lifetime_value_estimate / total_spend lies in the customer_segment's multiplier band"),
    Edge("total_purchases", "purchase_frequency_days", "approximate", "purchase_frequency_days ~ 365 / (total_purchases / U(1, 3)), floor 1"),
] + [Edge(f, "customer_segment", "soft", "hard band by segment in the generator, never stated to the agent") for f in SEGMENT_DRIVEN]

# What the engine recomputes when a field is dithered with recompute_derived=True.
PROPAGATED: Dict[str, List[str]] = {
    "churn_risk_score": ["is_at_risk"],
    "support_tickets_open": ["recently_contacted_support"],
}

# Segment bands copied from base_customer_generator._generate_segment_fields (inclusive).
# Verified empirically against generated data; a mismatch means the generator changed.
SEGMENT_BANDS = {
    "high_value":   {"total_purchases": (30, 100), "avg_order_value": (200, 800), "nps_score": (7, 10), "churn_risk_score": (0.0, 0.25),
                     "email_open_rate": (0.60, 0.95), "support_tickets_open": (0, 1), "payment_failures": (0, 0), "fraud_risk_score": (0.0, 0.08)},
    "medium_value": {"total_purchases": (10, 35), "avg_order_value": (50, 250), "nps_score": (5, 8), "churn_risk_score": (0.20, 0.55),
                     "email_open_rate": (0.35, 0.70), "support_tickets_open": (0, 2), "payment_failures": (0, 1), "fraud_risk_score": (0.0, 0.15)},
    "low_value":    {"total_purchases": (1, 12), "avg_order_value": (10, 80), "nps_score": (3, 6), "churn_risk_score": (0.35, 0.62),
                     "email_open_rate": (0.15, 0.45), "support_tickets_open": (0, 3), "payment_failures": (0, 2), "fraud_risk_score": (0.0, 0.25)},
    "at_risk":      {"total_purchases": (5, 25), "avg_order_value": (30, 150), "nps_score": (1, 4), "churn_risk_score": (0.65, 0.95),
                     "email_open_rate": (0.03, 0.25), "support_tickets_open": (2, 6), "payment_failures": (1, 4), "fraud_risk_score": (0.10, 0.40)},
}
LTV_BANDS = {"high_value": (2.5, 4.0), "medium_value": (1.5, 2.5), "low_value": (1.0, 1.5), "at_risk": (0.8, 1.2)}

# Value ranges the agent's system prompt states (parsed from the agent file and verified).
STATED_RANGES: Dict[str, Tuple[float, float]] = {
    "nps_score": (0.0, 10.0), "email_open_rate": (0.0, 1.0),
    "churn_risk_score": (0.0, 1.0), "fraud_risk_score": (0.0, 1.0),
}
AGENT_FILE = Path(__file__).resolve().parent / "business_decision_agent.py"

# Fields with no generator-defined link to any other field.
INDEPENDENT_FIELDS = ("avg_resolution_time_hours", "refund_rate", "acquisition_channel", "phone",
                      "address", "has_pending_order", "has_active_subscription")


def siblings(field: str) -> Dict[str, List[str]]:
    """tier -> sorted sibling names, from every edge touching `field`."""
    out: Dict[str, set] = defaultdict(set)
    for e in EDGES:
        if e.a == field:
            out[e.tier].add(e.b)
        elif e.b == field:
            out[e.tier].add(e.a)
    return {t: sorted(out[t]) for t in TIER_ORDER if out.get(t)}


def redundancy_profile(field: str, recompute_derived: bool = True) -> Dict[str, Any]:
    """Per-field summary. 'class' is the strongest tier among UNCORRECTED siblings:
    strict | bounded | approximate | soft_only | none."""
    sib = siblings(field)
    propagated = set(PROPAGATED.get(field, [])) if recompute_derived else set()
    uncorrected = {t: [s for s in names if s not in propagated] for t, names in sib.items()}
    uncorrected = {t: names for t, names in uncorrected.items() if names}
    cls = next((t for t in TIER_ORDER if t in uncorrected), None)
    return {"field": field, "siblings": sib, "propagated": sorted(propagated), "uncorrected": uncorrected,
            "stated_range": STATED_RANGES.get(field),
            "class": {"soft": "soft_only"}.get(cls, cls or "none")}


def format_table() -> str:
    rows = ["field | strict | bounded | approximate | soft | engine propagates | class (default) | class (isolated) | range stated to agent", "-" * 135]
    for f in DITHERED_FIELDS:
        p, q = redundancy_profile(f), redundancy_profile(f, recompute_derived=False)
        s = p["siblings"]
        rows.append(" | ".join([f, ",".join(s.get("strict", [])) or "-", ",".join(s.get("bounded", [])) or "-",
                                ",".join(s.get("approximate", [])) or "-", "segment" if "soft" in s else "-",
                                ",".join(p["propagated"]) or "-", p["class"], q["class"],
                                "{}-{}".format(*p["stated_range"]) if p["stated_range"] else "-"]))
    return "\n".join(rows)


# ============================================================================
# VERIFICATION: every claim above is checked against the generator and engine
# ============================================================================

def _strict_checks(c, snap) -> Dict[str, bool]:
    return {
        "tenure_months~account_created_date": c["account_created_date"] == (snap - timedelta(days=c["tenure_months"] * 30)).date().isoformat(),
        "last_purchase_days_ago~last_purchase_date": c["last_purchase_date"] == (snap - timedelta(days=c["last_purchase_days_ago"])).date().isoformat(),
        "churn_risk_score~is_at_risk": c["is_at_risk"] == (c["churn_risk_score"] >= 0.60),
        "support_tickets_open~recently_contacted_support": c["recently_contacted_support"] == (c["support_tickets_open"] >= 1),
        "name~email": c["email"] == c["name"].lower().replace(" ", ".").replace("'", "") + "@example.com",
    }


def _bounded_checks(c) -> Dict[str, bool]:
    t30, lp, lg = c["tenure_months"] * 30, c["last_purchase_days_ago"], c["last_login_days_ago"]
    return {
        "support_tickets_closed>=open": c["support_tickets_closed"] >= c["support_tickets_open"],
        "last_purchase<=min(365,30*tenure)": 1 <= lp <= min(365, t30),
        "login_vs_purchase": max(0, lp - 7) <= lg <= lp + 30,
        "login<=30*tenure": lg <= t30,
        "is_vip=>high_value": (not c["is_vip"]) or c["customer_segment"] == "high_value",
    }


def verify_against_generator(n: int = 3000, seed: int = 42) -> Dict[str, Any]:
    from base_customer_generator import generate_base_customers, SNAPSHOT_DATE, CONSISTENCY_RULES
    customers = generate_base_customers(n=n, seed=seed)
    failures: Dict[str, int] = defaultdict(int)
    observed: Dict[str, List[float]] = {"spend_ratio": [], "ltv_ratio": []}
    for c in customers:
        for name, ok in {**_strict_checks(c, SNAPSHOT_DATE), **_bounded_checks(c)}.items():
            failures[name] += (not ok)
        ratio = c["total_spend"] / (c["total_purchases"] * c["avg_order_value"])
        observed["spend_ratio"].append(ratio)
        failures["spend~purchases*aov in [0.90,1.10]"] += not (0.90 - 0.002 <= ratio <= 1.10 + 0.002)
        lo, hi = LTV_BANDS[c["customer_segment"]]
        lr = c["lifetime_value_estimate"] / c["total_spend"]
        observed["ltv_ratio"].append(lr)
        failures["ltv/spend in segment band"] += not (lo - 0.002 <= lr <= hi + 0.002)
        tp = c["total_purchases"]
        f_lo, f_hi = int(365 / max(1, tp)), max(1, int(365 / max(1, tp / 3.0)))
        failures["frequency~365/(purchases/U(1,3))"] += not (max(1, f_lo) - 1 <= c["purchase_frequency_days"] <= f_hi + 1)
        for field, (lo_, hi_) in SEGMENT_BANDS[c["customer_segment"]].items():
            pad = 0.0006 if isinstance(lo_, float) or isinstance(hi_, float) else 0
            failures[f"segment band: {field}"] += not (lo_ - pad <= c[field] <= hi_ + pad)
    rules_ok = CONSISTENCY_RULES["ltv_multiplier"] == {k: v for k, v in LTV_BANDS.items()}
    return {"n": n, "failures": {k: v for k, v in failures.items() if v}, "ltv_bands_match_generator": rules_ok,
            "checked": sorted(failures), "spend_ratio_range": (min(observed["spend_ratio"]), max(observed["spend_ratio"])),
            "ltv_ratio_range": (min(observed["ltv_ratio"]), max(observed["ltv_ratio"]))}


def parse_stated_ranges(agent_source: str) -> Dict[str, Tuple[float, float]]:
    """Value ranges the agent's SYSTEM_PROMPT glossary states, e.g. '- nps_score: ... (0-10, ...'."""
    start = agent_source.index('SYSTEM_PROMPT = """')
    block = agent_source[start: agent_source.index('"""', start + 20)]
    found = {}
    for line in block.splitlines():
        m = re.match(r"- (\w+): .*?\((\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", line)
        if m:
            found[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return found


def verify_stated_ranges(n: int = 3000, seed: int = 42, agent_source: Optional[str] = None) -> Dict[str, Any]:
    """STATED_RANGES must equal what the agent file's prompt says, and clean generated data must respect them."""
    from base_customer_generator import generate_base_customers
    problems = []
    if agent_source is None:
        if not AGENT_FILE.exists():
            return {"problems": [f"agent file not found at {AGENT_FILE}"], "parsed": {}}
        agent_source = AGENT_FILE.read_text()
    parsed = parse_stated_ranges(agent_source)
    if parsed != STATED_RANGES:
        problems.append(f"STATED_RANGES out of date: prompt says {parsed}, table says {STATED_RANGES}")
    for c in generate_base_customers(n=n, seed=seed):
        for f, (lo, hi) in STATED_RANGES.items():
            if not (lo <= c[f] <= hi):
                problems.append(f"clean {c['customer_id']} {f}={c[f]} outside its stated range {lo}-{hi}")
                break
        if len(problems) > 3:
            break
    return {"parsed": parsed, "problems": problems}


def verify_independence(n: int = 3000, seed: int = 42, threshold: float = 0.20) -> Dict[str, Any]:
    """Numeric 'independent' fields should show no within-segment rank correlation with any other numeric field."""
    from base_customer_generator import generate_base_customers
    from scipy.stats import spearmanr
    customers = generate_base_customers(n=n, seed=seed)
    numeric = [k for k, v in customers[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    worst = {}
    for f in (x for x in INDEPENDENT_FIELDS if x in numeric):
        w = 0.0
        for seg in {c["customer_segment"] for c in customers}:
            rows = [c for c in customers if c["customer_segment"] == seg]
            for g in numeric:
                if g == f:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")   # constant columns (e.g. payment_failures in high_value) give NaN
                    rho = spearmanr([r[f] for r in rows], [r[g] for r in rows])[0]
                if rho == rho:
                    w = max(w, abs(rho))
        worst[f] = round(float(w), 3)
    return {"threshold": threshold, "worst_within_segment_abs_rho": worst, "failures": {f: w for f, w in worst.items() if w >= threshold}}


def verify_propagation_against_engine(n: int = 600, seed: int = 42) -> Dict[str, Any]:
    """For each dithered field: dither it with recompute_derived=True and confirm the engine changes
    NO field outside {the field itself} + PROPAGATED[field], and that every declared propagation
    actually occurs for at least one customer."""
    from base_customer_generator import generate_base_customers
    from dither_engine import DitherConfig, DitherEngine, build_all_conditions
    engine_fields = {f for cond in build_all_conditions() for f in cond.fields}
    base = generate_base_customers(n=n, seed=seed)
    problems, undeclared, fired = [], {}, defaultdict(set)
    if engine_fields != set(DITHERED_FIELDS):
        problems.append(f"DITHERED_FIELDS out of date: engine-only={sorted(engine_fields - set(DITHERED_FIELDS))} "
                        f"table-only={sorted(set(DITHERED_FIELDS) - engine_fields)}")
    for f in DITHERED_FIELDS:
        cfg = DitherConfig(fields=[f], magnitude=0.5, dither_type=["drift"], correlated=True, seed=7, condition_id=f"_probe_{f}")
        out = DitherEngine(cfg).apply(base)
        allowed = {f} | set(PROPAGATED.get(f, []))
        for before, after in zip(base, out):
            changed = {k for k, v in before.items() if k in after and after[k] != v}
            extra = changed - allowed
            if extra:
                undeclared.setdefault(f, set()).update(extra)
            for s in changed & set(PROPAGATED.get(f, [])):
                fired[f].add(s)
    for f, extra in undeclared.items():
        problems.append(f"{f}: engine also changed undeclared field(s) {sorted(extra)}")
    for f, declared in PROPAGATED.items():
        missing = set(declared) - fired.get(f, set())
        if missing:
            problems.append(f"{f}: declared propagation to {sorted(missing)} never occurred in {n} customers")
    return {"n": n, "fields_probed": len(DITHERED_FIELDS), "problems": problems, "propagation_observed": {k: sorted(v) for k, v in fired.items()}}


def main() -> int:
    print("FIELD REDUNDANCY TABLE (pre-specified; derived from the generator and the engine)\n")
    print(format_table())
    print("\nclass = strongest tier among siblings the engine does NOT recompute.")
    print("'default' = recompute_derived=True (all conditions but two); 'isolated' = recompute_derived=False.\n")
    ok = True
    g = verify_against_generator()
    print(f"generator check ({g['n']} customers): {len(g['checked'])} relationships checked, failures: {g['failures'] or 'none'}")
    print(f"  spend / (purchases x aov) observed range {g['spend_ratio_range'][0]:.3f}-{g['spend_ratio_range'][1]:.3f}; "
          f"LTV / spend observed {g['ltv_ratio_range'][0]:.2f}-{g['ltv_ratio_range'][1]:.2f}; LTV bands match generator: {g['ltv_bands_match_generator']}")
    ok &= not g["failures"] and g["ltv_bands_match_generator"]
    r = verify_stated_ranges()
    print(f"stated-range check: prompt states {r['parsed']}; problems: {r['problems'] or 'none'}")
    ok &= not r["problems"]
    i = verify_independence()
    print(f"independence check: worst within-segment |rho| per 'independent' numeric field: {i['worst_within_segment_abs_rho']} (threshold {i['threshold']})")
    ok &= not i["failures"]
    p = verify_propagation_against_engine()
    print(f"engine propagation check ({p['n']} customers, {p['fields_probed']} fields probed): problems: {p['problems'] or 'none'}; observed: {p['propagation_observed']}")
    ok &= not p["problems"]
    print("\nALL CLAIMS VERIFIED" if ok else "\nVERIFICATION FAILED: the table and the generator/engine disagree")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

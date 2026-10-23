#!/usr/bin/env python3
"""
Evaluate H2 — Magnitude Effects
====================================
The Agentic Data Contract · Pillar 1: Authoritative · Experiment 1b

Tests whether decision drift scales with dither magnitude, across a
deliberate 3-field spread (churn_risk_score — H4 top-5 anchor,
total_spend — important but not self-reported, tenure_months — expected
lower decision weight), each dithered at 5%, 15%, 40%, and 100%
magnitude. The magnitude is the SIZE of each change relative to the value,
applied to every customer; it is not the share of records affected.

STRATIFICATION (Revision 3, R3-9 / D54). Every analysis is reported for three
strata, using the same helper as H4 and H5 (evaluate_core.stability_stratum):
    stable    customers whose five clean baseline runs all agreed  (PRIMARY)
    boundary  lightly_boundary + deeply_boundary + tied_no_majority (reported, never dropped)
    all       every customer, including any without a tier           (the unstratified number)
A boundary customer flips at its own clean rate (about 40% for deeply
boundary) with no corruption at all, so mixing them in inflates absolute
drift and costs power. Contrasts between arms stay valid either way; the
pre-registered H2 test (below) is defined on the stable stratum only.

THE PRE-REGISTERED SHAPE TEST (Revision 3, R3-3 / D56, with the materiality
line M of D57). Stable stratum only. Population: the stable customers whose
field value actually changed at every rung and who are present in the clean
re-run control, so s and F are computed on the same customers. Per field:
    s = drift at 5% minus the control's drift
    F = the largest drift over 15%, 40% and 100%, minus the control's drift
    floor:  assessed only if F >= M; otherwise "non-material for this decision: no shape claim"
    cliff:  s >= 0.5 F AND s above the control by a paired exact McNemar test (Holm across the three fields, 0.05)
    smooth: s < 0.25 F
    mixed:  anything else above the floor
The cliff candidate qualifies if at least two of the three fields are
cliffs AND its effect (the median s of the cliff fields) replicates in both
customer halves (same sign, at least half the pooled effect; R3-4). Its
effect for Claim 2 is that median s. s/F carries a customer-level bootstrap
interval; a non-monotone ladder is flagged; classification is also
tabulated at M = 5 and 15 points as a sensitivity analysis (headlines use
M = 10). Boundary and all-customer curves are reported net of their own
stratum's control and are never used for qualification.

DESCRIPTIVE AND SECONDARY ANALYSES (unchanged methods, now per stratum):
1. Cochran's Q (omnibus) across the 4 rungs per field, with adjacent-step
   McNemar's only if the omnibus is significant.
2. Extreme comparison (5% vs 100%), a labelled "total range effect" anchor.
3. Spearman's rank correlation on the 4-point curve, DESCRIPTIVE only.
4. Within-segment breakdown for churn_risk_score and total_spend only, on
   the stable and all strata.
"""

import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_core import (  # noqa: E402
    load_condition, compute_drift_rate, mcnemar_paired_test,
    cochrans_q_test, align_drift_by_customer, align_drift_by_customer_multi,
    descriptive_trend_correlation, stability_stratum,
)
import customer_halves  # noqa: E402


MAGNITUDE_LEVELS = [5, 15, 40, 100]  # percent
FIELDS = ["churn_risk_score", "total_spend", "tenure_months"]
WITHIN_SEGMENT_FIELDS = {"churn_risk_score", "total_spend"}  # NOT tenure_months
WITHIN_SEGMENT_STRATA = ("stable", "all")

STRATA = ("stable", "boundary", "all")
PRIMARY_STRATUM = "stable"
CONTROL_CONDITION = "h0_clean_rerun_1"

MATERIALITY_M = 0.10                 # D57: 10 points of net drift, stable stratum
MATERIALITY_SENSITIVITY = (0.05, 0.15)
CLIFF_RATIO = 0.5                    # D56: s >= 0.5 F
SMOOTH_RATIO = 0.25                  # D56: s < 0.25 F
ALPHA = 0.05
MIN_CLIFF_FIELDS = 2                 # D56: at least two of three fields
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261023            # fixed, so the interval is reproducible


def condition_id_for(field: str, magnitude_pct: int) -> str:
    return f"h2_{field}_mag{magnitude_pct}pct"


def in_stratum(record: Dict[str, Any], stratum: str) -> bool:
    return True if stratum == "all" else stability_stratum(record) == stratum


def _finite(x: Optional[float]) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# ----------------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------------

def load_all_h2_conditions(
    experiments_output_dir: Path,
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Load all 12 H2 conditions, organized as {field: {magnitude_pct: records}}
    for easy per-field ladder access.
    """
    conditions_dir = experiments_output_dir / "conditions"
    loaded: Dict[str, Dict[int, List[Dict[str, Any]]]] = {f: {} for f in FIELDS}

    for field in FIELDS:
        for mag in MAGNITUDE_LEVELS:
            cid = condition_id_for(field, mag)
            loaded[field][mag] = load_condition(conditions_dir / cid, ground_truth)
            print(f"  Loaded {cid}: {len(loaded[field][mag])} records")

    return loaded


def load_control(experiments_output_dir: Path, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    cdir = experiments_output_dir / "conditions" / CONTROL_CONDITION
    if not (cdir / "control.json").exists():
        raise FileNotFoundError(
            f"{CONTROL_CONDITION} (the clean re-run control) is required: H2's shape test is net of the control's drift. "
            f"Generate it with generate_dithered_data.py (it is written by default) and run it in the same blocks.")
    records = load_condition(cdir, ground_truth)
    if any(r["dither_fields"] for r in records):
        raise ValueError(f"{CONTROL_CONDITION} has dithered fields: it is not a clean control")
    print(f"  Loaded {CONTROL_CONDITION} (control): {len(records)} records")
    return records


# ----------------------------------------------------------------------------------------------------------------------
# Descriptive ladder (per stratum)
# ----------------------------------------------------------------------------------------------------------------------

def analyze_field_ladder(
    field: str,
    magnitude_records: Dict[int, List[Dict[str, Any]]],
    control_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Magnitude-ladder analysis for one field on the records given (already
    restricted to one stratum): Cochran's Q omnibus gate, conditional
    adjacent-step McNemar's, extreme-comparison anchor, the descriptive
    Spearman's trend curve, and the drift curve net of the control.
    """
    n = min(len(magnitude_records[m]) for m in MAGNITUDE_LEVELS)
    if n == 0:
        return {"field": field, "n_customers": 0, "note": "no customers in this stratum"}

    drift_rates = {mag: compute_drift_rate(magnitude_records[mag]) for mag in MAGNITUDE_LEVELS}

    # --- Cochran's Q omnibus gate ---
    ordered_records = [magnitude_records[mag] for mag in MAGNITUDE_LEVELS]
    binary_matrix, shared_customers = align_drift_by_customer_multi(ordered_records)
    omnibus = cochrans_q_test(binary_matrix) if binary_matrix else {"Q_statistic": None, "df": 3, "p_value": None,
                                                                    "note": "no customer is present at every rung"}
    omnibus_significant = omnibus["p_value"] is not None and omnibus["p_value"] < ALPHA

    # --- Conditional adjacent-step McNemar's ---
    adjacent_results = None
    if omnibus_significant:
        adjacent_results = []
        for i in range(len(MAGNITUDE_LEVELS) - 1):
            mag_a, mag_b = MAGNITUDE_LEVELS[i], MAGNITUDE_LEVELS[i + 1]
            drift_a, drift_b, n_shared = align_drift_by_customer(
                magnitude_records[mag_a], magnitude_records[mag_b])
            mc = mcnemar_paired_test(drift_a, drift_b)
            adjacent_results.append({
                "step": f"{mag_a}pct -> {mag_b}pct",
                "drift_rate_a": round(drift_rates[mag_a], 4),
                "drift_rate_b": round(drift_rates[mag_b], 4),
                "n_shared_customers": n_shared,
                **mc,
            })

    # --- Extreme comparison (5% vs 100%) — separate labeled anchor ---
    drift_5, drift_100, n_shared_extreme = align_drift_by_customer(
        magnitude_records[5], magnitude_records[100])
    extreme_comparison = {
        "drift_rate_5pct":   round(drift_rates[5], 4),
        "drift_rate_100pct": round(drift_rates[100], 4),
        "total_range_effect": round(drift_rates[100] - drift_rates[5], 4),
        "n_shared_customers": n_shared_extreme,
        **mcnemar_paired_test(drift_5, drift_100),
        "note": "Supplementary anchor for stakeholder communication — not "
                "part of the omnibus/adjacent test family, not subject to "
                "the same multiple-comparisons correction, since it makes "
                "a different claim (total range) than a localized change.",
    }

    # --- Descriptive trend curve ---
    trend = descriptive_trend_correlation(
        MAGNITUDE_LEVELS, [drift_rates[m] for m in MAGNITUDE_LEVELS])
    trend["rho"], trend["p_value"] = _finite(trend.get("rho")), _finite(trend.get("p_value"))

    # --- Net of the control, on the same customers ---
    net = None
    if control_records is not None:
        ids = set(shared_customers)
        ctrl = [r for r in control_records if r["customer_id"] in ids]
        if ctrl:
            c = compute_drift_rate(ctrl)
            net = {"control_drift_rate": round(c, 4), "n_customers": len(ctrl),
                   "net_drift_by_magnitude": {str(m): round(_rate(magnitude_records[m], ids) - c, 4) for m in MAGNITUDE_LEVELS}}

    return {
        "field": field,
        "n_customers": len(shared_customers),
        "drift_rates_by_magnitude": {str(m): round(drift_rates[m], 4) for m in MAGNITUDE_LEVELS},
        "net_of_control": net,
        "cochrans_q_omnibus": omnibus,
        "omnibus_significant": omnibus_significant,
        "adjacent_step_mcnemar": adjacent_results,
        "adjacent_tests_note": (
            "Adjacent-step tests run and reported below."
            if omnibus_significant else
            "Cochran's Q was not significant — no adjacent-step tests run. "
            "This null finding is reported as complete on its own, not as "
            "grounds to search for a localized effect the omnibus gate "
            "didn't detect."
        ),
        "extreme_comparison_5v100": extreme_comparison,
        "descriptive_trend": trend,
    }


def within_segment_breakdown(
    magnitude_records: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Drift rate computed separately per customer_segment, at each
    magnitude level — separates "the field intrinsically matters" from
    "the field is a proxy for segment membership." Only called for
    churn_risk_score and total_spend, per the amendment's scope.
    """
    result = {}
    for mag in MAGNITUDE_LEVELS:
        by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for r in magnitude_records[mag]:
            by_segment.setdefault(r["customer_segment"], []).append(r)
        result[str(mag)] = {segment: round(compute_drift_rate(recs), 4) for segment, recs in by_segment.items()}
    return result


# ----------------------------------------------------------------------------------------------------------------------
# The shape test (stable stratum)
# ----------------------------------------------------------------------------------------------------------------------

def _rate(records: List[Dict[str, Any]], ids: Set[str]) -> float:
    sel = [r["drifted"] for r in records if r["customer_id"] in ids]
    return sum(sel) / len(sel) if sel else float("nan")


def shape_population(field: str, magnitude_records: Dict[int, List[Dict[str, Any]]],
                     control_records: List[Dict[str, Any]]) -> List[str]:
    """Stable customers whose `field` actually changed at EVERY rung and who are in the control."""
    sets = []
    for m in MAGNITUDE_LEVELS:
        sets.append({r["customer_id"] for r in magnitude_records[m]
                     if in_stratum(r, PRIMARY_STRATUM) and field in (r.get("dither_fields") or [])})
    sets.append({r["customer_id"] for r in control_records if in_stratum(r, PRIMARY_STRATUM)})
    return sorted(set.intersection(*sets))


def _s_and_F(drift: Dict[Any, Dict[str, bool]], control: Dict[str, bool], ids: List[str]):
    """s, F and the peak rung on exactly these customers (ids may repeat in a bootstrap resample)."""
    n = len(ids)
    c = sum(control[i] for i in ids) / n
    d = {m: sum(drift[m][i] for i in ids) / n for m in MAGNITUDE_LEVELS}
    peak = max(MAGNITUDE_LEVELS[1:], key=lambda m: d[m])
    return d[5] - c, d[peak] - c, peak, d, c


def classify(s: float, F: float, p_holm: Optional[float], M: float) -> str:
    if F < M:
        return "below_floor"
    if s >= CLIFF_RATIO * F and p_holm is not None and p_holm < ALPHA and s > 0:
        return "cliff"
    if s < SMOOTH_RATIO * F:
        return "smooth"
    return "mixed"


def _holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjustment; None p-values (no test possible) stay None and do not count toward the family size."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def shape_field_stats(loaded: Dict[str, Dict[int, List[Dict[str, Any]]]],
                      control_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Everything about each field's curve that does not depend on M (computed once; M only changes classification)."""
    control = {r["customer_id"]: r["drifted"] for r in control_records}
    per_field: Dict[str, Dict[str, Any]] = {}
    for field in FIELDS:
        ids = shape_population(field, loaded[field], control_records)
        if not ids:
            per_field[field] = {"n_customers": 0, "classification": "no_population",
                                "note": "no stable customer had this field changed at every rung"}
            continue
        drift = {m: {r["customer_id"]: r["drifted"] for r in loaded[field][m]} for m in MAGNITUDE_LEVELS}
        s, F, peak, d, c = _s_and_F(drift, control, ids)
        mc = mcnemar_paired_test([drift[5][i] for i in ids], [control[i] for i in ids])
        # bootstrap of s/F over customers
        rng = random.Random(f"{BOOTSTRAP_SEED}|{field}")
        ratios = []
        for _ in range(BOOTSTRAP_REPS):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            bs, bF, *_ = _s_and_F(drift, control, sample)
            if bF > 0:
                ratios.append(bs / bF)
        ratios.sort()
        ci = ([round(ratios[int(0.025 * (len(ratios) - 1))], 4), round(ratios[int(0.975 * (len(ratios) - 1))], 4)]
              if len(ratios) >= 100 else None)
        halves = {}
        for h, hids in customer_halves.split(ids).items():
            if hids:
                hs, hF, *_ = _s_and_F(drift, control, hids)
                halves[str(h)] = {"n_customers": len(hids), "s": round(hs, 4), "F": round(hF, 4)}
            else:
                halves[str(h)] = {"n_customers": 0, "s": None, "F": None}
        rungs = [d[m] for m in MAGNITUDE_LEVELS]
        per_field[field] = {
            "n_customers": len(ids),
            "control_drift_rate": round(c, 4),
            "drift_by_magnitude": {str(m): round(d[m], 4) for m in MAGNITUDE_LEVELS},
            "s": round(s, 4), "F": round(F, 4), "F_rung_pct": peak,
            "s_over_F": round(s / F, 4) if F > 0 else None,
            "s_over_F_ci95_bootstrap": ci,
            "non_monotone": any(b < a for a, b in zip(rungs, rungs[1:])),
            "mcnemar_5pct_vs_control": {"a_only_5pct": mc["b_x_only"], "b_only_control": mc["c_y_only"],
                                        "n_discordant": mc["n_discordant"], "p_value": mc["p_value"]},
            "halves": halves,
        }
    names = [f for f in FIELDS if per_field[f].get("n_customers")]
    adj = _holm([per_field[f]["mcnemar_5pct_vs_control"]["p_value"] for f in names])   # Holm across the three fields
    for f, p in zip(names, adj):
        per_field[f]["mcnemar_5pct_vs_control"]["p_holm"] = p
    return per_field


def shape_test(stats: Dict[str, Dict[str, Any]], M: float = MATERIALITY_M) -> Dict[str, Any]:
    import copy
    per_field = copy.deepcopy(stats)
    names = [f for f in FIELDS if per_field[f].get("n_customers")]
    for f in names:
        per_field[f]["classification"] = classify(per_field[f]["s"], per_field[f]["F"],
                                                  per_field[f]["mcnemar_5pct_vs_control"]["p_holm"], M)

    cliffs = [f for f in FIELDS if per_field[f].get("classification") == "cliff"]
    effect = statistics.median(per_field[f]["s"] for f in cliffs) if cliffs else None
    halves_effect, replicated = {}, None
    if len(cliffs) >= MIN_CLIFF_FIELDS:
        replicated = True
        for h in ("0", "1"):
            vals = [per_field[f]["halves"][h]["s"] for f in cliffs if per_field[f]["halves"][h]["s"] is not None]
            he = statistics.median(vals) if len(vals) == len(cliffs) else None
            halves_effect[h] = None if he is None else round(he, 4)
            if he is None or he <= 0 or he < 0.5 * effect:
                replicated = False
    qualified = bool(len(cliffs) >= MIN_CLIFF_FIELDS and replicated)
    return {
        "M": M,
        "per_field": per_field,
        "cliff_fields": cliffs,
        "n_cliff_fields": len(cliffs),
        "candidate_effect_median_s": None if effect is None else round(effect, 4),
        "effect_by_half": halves_effect or None,
        "replicated_in_both_halves": replicated,
        "qualified": qualified,
        "effect_at_least_M": bool(effect is not None and effect >= M),
    }


def shape_test_report(loaded, control_records) -> Dict[str, Any]:
    stats = shape_field_stats(loaded, control_records)
    primary = shape_test(stats, MATERIALITY_M)
    sens = {}
    for m in MATERIALITY_SENSITIVITY:
        t = shape_test(stats, m)
        sens[f"M_{int(round(m * 100))}"] = {"classification": {f: t["per_field"][f].get("classification") for f in FIELDS},
                                            "qualified": t["qualified"], "candidate_effect_median_s": t["candidate_effect_median_s"]}
    primary["sensitivity"] = sens
    primary["rules"] = {
        "stratum": PRIMARY_STRATUM, "population": "stable customers changed at every rung and present in the control",
        "M": MATERIALITY_M, "cliff": f"s >= {CLIFF_RATIO} F and McNemar (5% vs control) p_holm < {ALPHA}, s > 0",
        "smooth": f"s < {SMOOTH_RATIO} F", "floor": "F >= M", "qualification": f">= {MIN_CLIFF_FIELDS} cliff fields and the median s "
        f"replicates in both customer halves ({customer_halves.PARTITION}: same sign, at least half the pooled effect)",
        "bootstrap": f"{BOOTSTRAP_REPS} customer-level resamples, seed {BOOTSTRAP_SEED}, percentile 95%",
    }
    return primary


# ----------------------------------------------------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------------------------------------------------

def evaluate_h2(
    experiments_output_dir: Path,
    finalized_ground_truth_path: Path,
) -> Dict[str, Any]:
    with open(finalized_ground_truth_path) as f:
        ground_truth = json.load(f)

    print("Loading all 12 H2 conditions and the clean re-run control...")
    loaded = load_all_h2_conditions(experiments_output_dir, ground_truth)
    control = load_control(experiments_output_dir, ground_truth)

    results: Dict[str, Any] = {
        "hypothesis": "H2",
        "primary_stratum": PRIMARY_STRATUM,
        "strata_definitions": {
            "stable": "stability_tier == stable (all five clean baseline runs agreed); PRIMARY",
            "boundary": "lightly_boundary, deeply_boundary, tied_no_majority (reported, never dropped)",
            "all": "every customer, including any without a tier (the unstratified number)",
        },
        "stratum_sizes": {s: len({r["customer_id"] for r in control if in_stratum(r, s)}) for s in STRATA},
        "control_condition": CONTROL_CONDITION,
        "fields": {},
    }

    for field in FIELDS:
        print(f"\nAnalyzing magnitude ladder for {field}...")
        per = {}
        for s in STRATA:
            recs = {m: [r for r in loaded[field][m] if in_stratum(r, s)] for m in MAGNITUDE_LEVELS}
            ctrl = [r for r in control if in_stratum(r, s)]
            per[s] = analyze_field_ladder(field, recs, ctrl)
            if field in WITHIN_SEGMENT_FIELDS and s in WITHIN_SEGMENT_STRATA and per[s].get("n_customers"):
                per[s]["within_segment_breakdown"] = within_segment_breakdown(recs)
        results["fields"][field] = per

    print("\nRunning the pre-registered shape test (stable stratum)...")
    results["shape_test"] = shape_test_report(loaded, control)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_output", type=str, default="experiments_output")
    parser.add_argument("--finalized_ground_truth", type=str,
        default="experiments_output/finalized_ground_truth.json")
    parser.add_argument("--out", type=str,
        default="experiments_output/evaluation/h2_results.json")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("H2 Evaluation — Magnitude Effects")
    print(f"{'='*60}\n")

    results = evaluate_h2(Path(args.experiments_output), Path(args.finalized_ground_truth))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

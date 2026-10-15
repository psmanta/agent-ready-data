#!/usr/bin/env python3
"""
H5 evaluator -- Detection Awareness (analysis stage) -- Experiment 1b
=====================================================================
The Agentic Data Contract . Pillar 1: Authoritative

Implements exactly the analysis the amendment fixes for H5 ("H5 -- Detection Awareness"), using the
instruments computed in load_condition(): the frozen 25-pattern keyword scan and the deterministic value
echo check, plus the clean-baseline floors, the Jaccard machinery, and (optionally) the blind judge.
It adds no new statistics; every output block carries a `status` so a reader can never mistake an
exploratory table for a pre-registered one.

  PRIMARY      detection_crosstab: per condition, per stratum, the three-way table
               (detected AND the decision changed / detected, no change / not detected), the keyword
               detection rate with a Wilson interval, and the clean-baseline keyword floor read beside it.
               (The five paired tests per family pre-specified under H4 are produced by evaluate_h4.py.)
  EXPLORATORY  echo_classes: what the reasoning did to each dithered value, with its clean floor and the
               drift rate by class.
               coherence: the Jaccard condition-level shift vs the customer's own baseline wobble, and the
               descriptive reasoning-inertia count (flat coherence AND a flipped decision).
               redundancy_classes: detection by redundancy class across single-field conditions, and the
               negative controls (fields with no partner).
  PROVISIONAL  judge: category tables for conditions that have blind-judge output (the eight H4 conditions),
               until the human audit calibrates the judge.

Definitions fixed here (disclosed in the amendment):
  behavior changed   the decision differs from the reference decision, OR the dithered confidence lies outside
                     the customer's own baseline confidence range (min..max over their baseline runs). A customer
                     with no recorded range counts on the decision alone and is counted as "confidence unknown".
                     The decision-only version is reported beside it (it is what the pre-specified tests use).
  reasoning inertia  needs >= 3 matching baseline texts (with only two there is a single baseline pair, so "flat"
                     is nearly impossible and would bias the count down); customers with fewer are excluded from
                     the inertia counts but still contribute to the coherence shift.

All tables are over PERTURBED customers (those whose record has a dithered field), as in H1's exposure-
adjusted analysis: an unperturbed customer has nothing to detect and would dilute every rate. Each
condition reports its exposure beside its tables.

Strata: `stable` customers give the clean attribution of drift to corruption; `boundary` customers
(lightly_boundary, deeply_boundary, tied_no_majority) are reported separately, never discarded; `all` is
also reported. Customers with no recognized tier appear in `all` only.

Floors come from five baseline runs per customer that are NOT independent (about 44% of clean texts are
exact duplicates at temperature 0): they are point estimates and carry no intervals.

Usage:
    python evaluate_h5.py --experiments_output experiments_output \\
        --finalized_ground_truth experiments_output/finalized_ground_truth.json \\
        --baseline_reference experiments_output/baseline/baseline_reference.json \\
        [--judge_dir classifier_output] --out experiments_output/evaluation/h5_results.json
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "shared" / "data_generation"))
sys.path.insert(0, str(_HERE))

from evaluate_core import (  # noqa: E402
    clean_baseline_echo_floor, clean_baseline_keyword_floor, jaccard_condition_level_shift,
    jaccard_similarity, load_baseline_customers, load_condition, mean_pairwise_jaccard,
    outcome_rate_by_group, stability_stratum, wilson_interval, H5_KEYWORD_PATTERNS, ECHO_SCALE_FACTORS,
)
import field_redundancy as fr  # noqa: E402
try:
    from classifier_prompt import PROMPT_ID as FROZEN_JUDGE_PROMPT_ID  # noqa: E402
except ImportError:           # the stamp then records null rather than guessing
    FROZEN_JUDGE_PROMPT_ID = None

STRATA = ("stable", "boundary", "all")
HAS_PARTNER = ("strict", "bounded", "approximate")        # primary grouping, decided by the owner:
NO_PARTNER = ("soft_only", "none")                        # soft_only is ALSO reported on its own
STATUS = {"detection_crosstab": "primary", "echo_classes": "exploratory", "coherence": "exploratory",
          "redundancy_classes": "exploratory", "judge": "provisional"}


def in_stratum(record: Dict[str, Any], stratum: str) -> bool:
    return True if stratum == "all" else stability_stratum(record) == stratum


def rate_block(k: int, n: int) -> Dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci": None}
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}


def drift_matrix(table: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """An explicit [group] x [drift status] count matrix from an {n, k, ...} rate table (k = drifted)."""
    return {g: {"not_drifted": b["n"] - b["k"], "drifted": b["k"]} for g, b in table.items()}


def sample_sizes(records: List[Dict[str, Any]], perturbed: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(stability_stratum(r) for r in perturbed)
    return {"total_n": len(records), "n_perturbed": len(perturbed), "n_stable": counts["stable"],
            "n_boundary": counts["boundary"], "n_unclassified": counts["unknown"]}


def judge_prompt_ids(judge_dir: Optional[Path], condition_id: str) -> Optional[List[str]]:
    """The prompt id(s) found in a condition's judge output file; None if there is no judge file."""
    if judge_dir is None:
        return None
    path = Path(judge_dir) / f"{condition_id}.jsonl"
    if not path.exists():
        return None
    return sorted({json.loads(line).get("prompt_id") for line in open(path) if line.strip()} - {None})


def audit_stamp(tables: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance stamped into every H5 output: which frozen instruments produced these numbers."""
    judged = sorted(c for c, t in tables.items() if t.get("judge_prompt_ids") is not None)
    found = sorted({i for t in tables.values() for i in (t.get("judge_prompt_ids") or [])})
    return {
        "keyword_scan": {"n_patterns": len(H5_KEYWORD_PATTERNS),
                         "sha256": hashlib.sha256("\n".join(H5_KEYWORD_PATTERNS).encode()).hexdigest()},
        "echo_check": {"scale_factors": [[n, v] for n, v in ECHO_SCALE_FACTORS]},
        "judge": {"frozen_prompt_id": FROZEN_JUDGE_PROMPT_ID, "prompt_ids_found": found, "conditions_judged": judged,
                  "all_match_frozen": (None if not judged else (FROZEN_JUDGE_PROMPT_ID is not None and found == [FROZEN_JUDGE_PROMPT_ID]))},
        "primary_stratum": "stable",
    }


def discover_conditions(out_dir: Path) -> List[str]:
    root = out_dir / "conditions"
    # a control (control.json, such as the zero-dither h0_clean_rerun_<k>) is not a hypothesis condition
    return sorted(p.name for p in root.iterdir()
                  if (p / "decisions.jsonl").exists() and (p / "dither_reference.json").exists()
                  and not (p / "control.json").exists())


def single_field(records: List[Dict[str, Any]]) -> Optional[str]:
    fields = {f for r in records for f in r["dither_fields"]}
    return next(iter(fields)) if len(fields) == 1 else None


# ----------------------------------------------------------------------------
# PRIMARY: the three-way detection table
# ----------------------------------------------------------------------------

def confidence_outside(record: Dict[str, Any], ground_truth: Dict[str, Any]) -> Optional[bool]:
    """True/False: is the dithered confidence outside the customer's baseline range? None if unknowable."""
    c = record.get("dithered_confidence")
    rng = (ground_truth.get(record["customer_id"]) or {}).get("confidence_range")
    if c is None or not rng or len(rng) != 2 or rng[0] is None or rng[1] is None:
        return None
    return bool(c < rng[0] or c > rng[1])


def detection_crosstab(records: List[Dict[str, Any]], baseline: Dict[str, Any],
                       ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for s in STRATA:
        rs = [r for r in records if in_stratum(r, s)]
        det = [r for r in rs if r["h5_keyword_detected"]]
        nd = [r for r in rs if not r["h5_keyword_detected"]]
        dc = sum(bool(r["drifted"]) for r in det)
        ndc = sum(bool(r["drifted"]) for r in nd)
        conf = [confidence_outside(r, ground_truth) for r in rs]
        changed = [bool(r["drifted"]) or bool(c) for r, c in zip(rs, conf)]
        det_flags = [bool(r["h5_keyword_detected"]) for r in rs]
        floor = clean_baseline_keyword_floor(baseline, [r["customer_id"] for r in rs])
        out[s] = {
            "n": len(rs),
            "detected_changed": dc, "detected_unchanged": len(det) - dc,
            "not_detected_changed": ndc, "not_detected_unchanged": len(nd) - ndc,
            "three_way": {"detected_and_changed": sum(d and c for d, c in zip(det_flags, changed)),
                          "detected_no_change": sum(d and not c for d, c in zip(det_flags, changed)),
                          "not_detected": len(nd)},
            "three_way_decision_only": {"detected_and_changed": dc, "detected_no_change": len(det) - dc, "not_detected": len(nd)},
            "detection_by_drift_matrix": {"detected": {"not_drifted": len(det) - dc, "drifted": dc},
                                          "not_detected": {"not_drifted": len(nd) - ndc, "drifted": ndc}},
            "behavior_changed_definition": "decision differs from the reference OR confidence outside the customer's baseline range",
            "confidence": {"n_outside_baseline_range": sum(bool(c) for c in conf), "n_unknown": sum(c is None for c in conf)},
            "keyword_detection": rate_block(len(det), len(rs)),
            "clean_baseline_floor": {k: floor[k] for k in ("n_customers", "n_runs", "n_hits", "hit_rate")},
        }
    return out


# ----------------------------------------------------------------------------
# EXPLORATORY: echo classes
# ----------------------------------------------------------------------------

def echo_classes(records: List[Dict[str, Any]], baseline: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for f in sorted({f for r in records for f in r["dither_fields"]}):
        per = {}
        for s in STRATA:
            rs = [r for r in records if f in r["dither_fields"] and f in r["value_echo"] and in_stratum(r, s)]
            floor = clean_baseline_echo_floor(rs, f, baseline)
            floor.pop("customers_with_any_run", None)
            dbe = outcome_rate_by_group(rs, lambda r, f=f: r["value_echo"][f]["echo_class"])
            per[s] = {"n": len(rs),
                      "class_counts": dict(sorted(Counter(r["value_echo"][f]["echo_class"] for r in rs).items())),
                      "drift_by_echo_class": dbe,
                      "echo_by_drift_matrix": drift_matrix(dbe),
                      "clean_baseline_floor": floor}
        out[f] = per
    return out


# ----------------------------------------------------------------------------
# EXPLORATORY: Jaccard coherence shift and reasoning inertia
# ----------------------------------------------------------------------------

def coherence(records: List[Dict[str, Any]], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Mirrors H3's method: a customer's dithered reasoning is compared with their baseline reasoning that
    reached the same (ground-truth) decision. A customer with fewer than two such texts is skipped (counted).
    Reasoning inertia (descriptive, a threshold-based flag, not a test): the dithered text's mean Jaccard
    similarity lies within the range of the customer's own baseline pairwise similarities (flat) AND the
    decision flipped."""
    per_customer = []
    for r in records:
        entry = baseline.get(r["customer_id"])
        texts = [x["decision_reasoning"] for x in entry["run_details"]
                 if entry and x.get("business_decision") == r["ground_truth_decision"] and x.get("decision_reasoning")] if entry else []
        if len(texts) < 2 or not r["dithered_reasoning"]:
            per_customer.append(None)
            continue
        d = mean_pairwise_jaccard(r["dithered_reasoning"], texts)
        b = [mean_pairwise_jaccard(texts[i], texts[:i] + texts[i + 1:]) for i in range(len(texts))]
        b = [x for x in b if x is not None]
        pair = [jaccard_similarity(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        if d is None or not b or not pair:
            per_customer.append(None)
            continue
        per_customer.append({"d": d, "b": sum(b) / len(b), "flat": min(pair) <= d <= max(pair), "drifted": bool(r["drifted"]),
                             "n_texts": len(texts)})
    out = {"definition": {"baseline_texts": "the customer's baseline reasoning that reached the ground-truth decision (at least 2 for the coherence shift; at least 3 for the inertia counts)",
                          "flat": "dithered mean Jaccard within [min, max] of the customer's own baseline pairwise Jaccard scores",
                          "inertia": "flat AND the decision flipped, among customers with >= 3 matching baseline texts (descriptive count, not a test)"}}
    for s in STRATA:
        rows = [(r, c) for r, c in zip(records, per_customer) if in_stratum(r, s)]
        elig = [c for _, c in rows if c is not None]
        iel = [c for c in elig if c["n_texts"] >= 3]
        block = {"n": len(rows), "n_eligible": len(elig), "n_skipped": len(rows) - len(elig),
                 "n_inertia_eligible": len(iel), "n_excluded_from_inertia_fewer_than_3_texts": len(elig) - len(iel),
                 "n_drifted_inertia_eligible": sum(c["drifted"] for c in iel),
                 "flat_coherence": sum(c["flat"] for c in iel),
                 "inertia": sum(c["flat"] and c["drifted"] for c in iel)}
        if len(elig) >= 2:
            block["condition_level_shift"] = jaccard_condition_level_shift([c["d"] for c in elig], [c["b"] for c in elig])
        else:
            block["condition_level_shift"] = {"p_value": None, "n_pairs": len(elig), "note": "fewer than 2 eligible customers"}
        out[s] = block
    return out


# ----------------------------------------------------------------------------
# PROVISIONAL: judge tables
# ----------------------------------------------------------------------------

def judge_tables(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not any("judge_category" in r for r in records):
        return None
    out = {}
    for s in STRATA:
        rs = [r for r in records if "judge_category" in r and in_stratum(r, s)]
        matrix: Dict[str, Dict[str, int]] = {}
        for r in rs:
            f = r["dither_fields"][0]
            cls = r["value_echo"].get(f, {}).get("echo_class", "not_applicable")
            matrix.setdefault(cls, {}).setdefault(r["judge_category"], 0)
            matrix[cls][r["judge_category"]] += 1
        dbj = outcome_rate_by_group(rs, lambda r: r["judge_category"])
        out[s] = {"n": len(rs),
                  "category_counts": dict(sorted(Counter(r["judge_category"] for r in rs).items())),
                  "explicit_concern": rate_block(sum(r["judge_category"] == "explicit_concern" for r in rs), len(rs)),
                  "drift_by_judge_category": dbj,
                  "judge_by_drift_matrix": drift_matrix(dbj),
                  "echo_by_judge": {k: dict(sorted(v.items())) for k, v in sorted(matrix.items())}}
    return out


# ----------------------------------------------------------------------------
# EXPLORATORY: redundancy classes and negative controls
# ----------------------------------------------------------------------------

def redundancy_classes(loaded: Dict[str, List[Dict[str, Any]]], tables: Dict[str, Any]) -> Dict[str, Any]:
    rows, pooled = [], {}
    for cid, recs in loaded.items():
        f = single_field(recs)
        if f is None:
            continue
        cls = fr.redundancy_profile(f, recompute_derived=not cid.endswith("_isolated"))["class"]
        stable = tables[cid]["detection_crosstab"]["stable"]
        det = stable["keyword_detection"]
        rows.append({"condition_id": cid, "field": f, "class": cls, "n_stable": det["n"], "detected_stable": det["k"],
                     "keyword_detection_stable": det["rate"], "clean_baseline_floor_stable": stable["clean_baseline_floor"]["hit_rate"]})
        for key in (cls, "has_partner" if cls in HAS_PARTNER else "no_partner"):
            p = pooled.setdefault(key, [0, 0])
            p[0] += det["k"]; p[1] += det["n"]
    controls = [r for r in rows if r["field"] in fr.INDEPENDENT_FIELDS]
    return {
        "grouping": {"has_partner": list(HAS_PARTNER), "no_partner": list(NO_PARTNER), "soft_only": "also reported on its own"},
        "caveat": "Descriptive only. The same customers appear in every condition, so pooled records are not independent, and fields "
                  "differ in many ways besides redundancy (operator, magnitude, type, stated range).",
        "per_condition": sorted(rows, key=lambda r: r["condition_id"]),
        "pooled_stable": {k: rate_block(v[0], v[1]) for k, v in sorted(pooled.items())},
        "negative_controls": sorted(controls, key=lambda r: r["condition_id"]),
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def evaluate_h5(out_dir: Path, gt_path: Path, baseline_reference_path: Path, judge_dir: Optional[Path] = None,
                condition_ids: Optional[List[str]] = None, allow_incomplete: bool = False) -> Dict[str, Any]:
    ground_truth = json.load(open(gt_path))
    baseline = load_baseline_customers(baseline_reference_path)
    generated = sorted(p.name for p in (out_dir / "conditions").iterdir()
                       if (p / "dither_reference.json").exists() and not (p / "control.json").exists())   # controls are not hypothesis conditions
    missing = [c for c in generated if not (out_dir / "conditions" / c / "decisions.jsonl").exists()]
    if condition_ids is None and missing and not allow_incomplete:
        raise SystemExit(f"H5 reads every dithered condition, but {len(missing)} have no decisions yet: "
                         f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}. Run them through the agent, or pass "
                         f"--allow_incomplete to analyze what exists (the output lists what is missing).")
    ids = condition_ids if condition_ids is not None else discover_conditions(out_dir)
    loaded = {cid: load_condition(out_dir / "conditions" / cid, ground_truth) for cid in ids}
    judge_joined: Dict[str, int] = {}
    if judge_dir is not None and Path(judge_dir).exists():
        from evaluate_h4 import attach_judge_results
        judge_joined = attach_judge_results(loaded, Path(judge_dir))
    tables: Dict[str, Any] = {}
    for cid, recs in loaded.items():
        pert = [r for r in recs if r["dither_fields"]]
        t = {"exposure": {"n_records": len(recs), "n_perturbed": len(pert), "exposure": (len(pert) / len(recs)) if recs else None},
             "detection_crosstab": detection_crosstab(pert, baseline, ground_truth), "echo_classes": echo_classes(pert, baseline),
             "coherence": coherence(pert, baseline)}
        j = judge_tables(pert)
        if j is not None:
            t["judge"] = j
        t["sample_sizes"] = sample_sizes(recs, pert)
        t["judge_prompt_ids"] = judge_prompt_ids(judge_dir, cid) if j is not None else None
        tables[cid] = t
    return {
        "hypothesis": "H5",
        "status_by_section": STATUS,
        "strata_note": "stable = clean attribution; boundary reported separately (never discarded); all also reported",
        "floor_note": "Clean-baseline floors are point estimates (five non-independent runs per customer); no intervals.",
        "n_conditions": len(loaded),
        "conditions_missing_decisions": missing if condition_ids is None else [],
        "judge_records_joined": judge_joined,
        "audit_stamp": audit_stamp(tables),
        "conditions": tables,
        "redundancy_classes": redundancy_classes(loaded, tables),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_output", default="experiments_output")
    ap.add_argument("--finalized_ground_truth", default="experiments_output/finalized_ground_truth.json")
    ap.add_argument("--baseline_reference", default="experiments_output/baseline/baseline_reference.json")
    ap.add_argument("--judge_dir", default=None, help="classifier_output directory (judge results for the H4 conditions)")
    ap.add_argument("--allow_incomplete", action="store_true", help="analyze only the conditions that have decisions")
    ap.add_argument("--out", default="experiments_output/evaluation/h5_results.json")
    a = ap.parse_args()
    res = evaluate_h5(Path(a.experiments_output), Path(a.finalized_ground_truth), Path(a.baseline_reference),
                      Path(a.judge_dir) if a.judge_dir else None,
                      allow_incomplete=a.allow_incomplete)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(f"H5: {res['n_conditions']} conditions evaluated -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
H1 baseline replication check and conditional rule -- Experiment 1b
====================================================================
The Agentic Data Contract . Pillar 1: Authoritative

Implements the H1 conditional rule exactly as pre-registered in
1b_DESIGN_AMENDMENT_1.md. Run it AFTER the five clean baseline runs are
complete and BEFORE any dithered condition has been run through the agent.

What it does
  1. Counts how often each schema field is cited in `key_factors` across all
     baseline decisions (5 runs x all customers).
  2. Ranks the fields and compares the 1b ranking with 1a's top 5
     (last_purchase_days_ago, churn_risk_score, nps_score,
     lifetime_value_estimate, support_tickets_open): top-5 and top-8 overlap,
     where each 1a field landed, a bootstrap over customers for the stability
     of the overlap, and a Spearman correlation if a full 1a ranking is given.
  3. Applies the trigger. If the point-estimate top-5 overlap is 0, 1 or 2,
     it lists the individual conditions to add: one for every field in the 1b
     top 5 (eligible fields only) that does not already have an individual
     15% condition. Otherwise it adds nothing.

It never reads a drift outcome. Selection depends only on clean-baseline
citations, so the rule adds no forking path.

The output is a FROZEN MANIFEST (h1_replication_manifest.json), the single source
of truth for everything downstream:
  * the decision-bearing content (input hashes, the complete citation table, the
    overlap, the trigger, the conditions to add, and the pre-registered Question A
    groups) is covered by `decision_sha256`; it contains no paths, no timestamps
    and no library-dependent randomness, so the same inputs give the same hash on
    any machine;
  * the `context` block (bootstrap, optional Spearman, cost estimate, sequencing
    warning) is NOT covered by the hash;
  * it is write-once: re-running with identical decision content leaves the file
    untouched; different content is refused (delete the file deliberately if a
    re-run is truly intended);
  * the generator and evaluate_h1.py must verify `decision_sha256` and consume
    the manifest; neither re-evaluates the trigger.

The rule's constants below are FROZEN with the pre-registration. Do not edit
them after the baseline has run.

Measurement rules (pre-registered)
  * A cited item counts only if, trimmed and lowercased, it exactly matches a
    schema field name (the keys the agent sees, minus record_id). Anything
    else is unmatched: not assigned to any field, and its share is reported.
  * A field counts at most once per decision. Rate = decisions citing the field
    / decisions used. PARSE_ERROR decisions and decisions whose key_factors is
    not a list are excluded and counted.
  * Rank by integer citation count, descending; ties broken by field name,
    ascending (strictly deterministic, independent of hash randomization). The
    rate margin between ranks 5 and 6 is reported.
  * The bootstrap resamples CUSTOMERS (all of a customer's runs together). It
    is context for the reader; the trigger uses the point estimate.
  * Fields the engine cannot dither (customer_segment, is_at_risk,
    recently_contacted_support, dates, ...) are skipped, in rank order, until
    five eligible fields are found; the skipped fields are reported.

Usage
    python h1_baseline_replication.py \
        --decisions_dir experiments_output/baseline/decisions \
        --baseline_input experiments_output/baseline/agent_input/baseline_customers.jsonl \
        --canonical experiments_output/ground_truth/canonical_customers.json \
        --out experiments_output/evaluation/h1_replication_manifest.json

If it triggers, generate the new conditions with
generate_h1_replication_conditions.py --manifest <this file>, then run them
through the agent.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "shared" / "data_generation"))
sys.path.insert(0, str(_HERE))

from dither_engine import build_all_conditions  # noqa: E402
from field_redundancy import DITHERED_FIELDS  # noqa: E402

# ----------------------------------------------------------------------------
# FROZEN WITH THE PRE-REGISTRATION
# ----------------------------------------------------------------------------
RULE_VERSION = "h1-replication-v2"
MANIFEST_VERSION = "h1-replication-manifest-v1"
A1A_TOP5 = ("last_purchase_days_ago", "churn_risk_score", "nps_score",
            "lifetime_value_estimate", "support_tickets_open")
TRIGGER_MAX_OVERLAP = 2          # trigger when top-5 overlap <= this
TOP_K, TOP_K_WIDE = 5, 8
N_EXPECTED_RUNS = 5
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261009
ELIGIBLE_FIELDS = frozenset(DITHERED_FIELDS)   # exactly the fields the engine can dither
CONDITION_PREFIX = "h1_replication_"
# The six comparison fields the amendment fixes for Question A on the 1a list.
LEGACY_COMPARISON = ("email", "is_vip", "total_spend", "tenure_months",
                     "avg_resolution_time_hours", "refund_rate")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_fields(baseline_input: Path) -> List[str]:
    """The field names the agent sees: keys of the first baseline record, minus record_id."""
    with open(baseline_input) as f:
        first = json.loads(f.readline())
    return sorted(k for k in first if k != "record_id")


def load_runs(decisions_dir: Path, n_runs: int = N_EXPECTED_RUNS) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    """All n_runs baseline files; refuses to evaluate a partial or inconsistent baseline."""
    runs = []
    for i in range(1, n_runs + 1):
        p = decisions_dir / f"run{i}.decisions.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}: the rule is evaluated only on the complete baseline "
                                    f"(expected run1..run{n_runs})")
        runs.append((p, [json.loads(l) for l in open(p) if l.strip()]))
    ids0 = [r["record_id"] for r in runs[0][1]]
    if len(ids0) != len(set(ids0)):
        raise ValueError(f"{runs[0][0].name}: duplicate record_ids")
    for p, rows in runs[1:]:
        ids = [r["record_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(ids0):
            raise ValueError(f"{p.name}: record_ids differ from run1 (incomplete or different input)")
    return runs


# ----------------------------------------------------------------------------
# Counting and ranking
# ----------------------------------------------------------------------------

def count_citations(runs, fields: List[str]) -> Dict[str, Any]:
    lookup = {f.lower(): f for f in fields}
    per_customer: Dict[str, Counter] = defaultdict(Counter)
    n_dec: Counter = Counter()
    unmatched: Counter = Counter()
    n_items = n_unmatched = excluded = 0
    cost_sum, cost_n = 0.0, 0
    for _, rows in runs:
        for r in rows:
            kf = r.get("key_factors")
            if r.get("business_decision") in (None, "PARSE_ERROR") or not isinstance(kf, list):
                excluded += 1
                continue
            rid = r["record_id"]
            n_dec[rid] += 1
            if r.get("cost_usd"):
                cost_sum += r["cost_usd"]; cost_n += 1
            cited = set()
            for item in kf:
                n_items += 1
                f = lookup.get(item.strip().lower()) if isinstance(item, str) else None
                if f is None:
                    n_unmatched += 1
                    unmatched[str(item)] += 1
                else:
                    cited.add(f)
            for f in cited:
                per_customer[rid][f] += 1
    return {"per_customer": per_customer, "n_dec": n_dec, "excluded": excluded, "n_items": n_items,
            "n_unmatched": n_unmatched, "unmatched": unmatched,
            "mean_cost": (cost_sum / cost_n) if cost_n else None}


def build_matrix(counts: Dict[str, Any], fields: List[str]):
    ids = sorted(counts["n_dec"])
    idx = {f: j for j, f in enumerate(fields)}
    C = np.zeros((len(ids), len(fields)), dtype=np.int64)
    n = np.zeros(len(ids), dtype=np.int64)
    for i, rid in enumerate(ids):
        n[i] = counts["n_dec"][rid]
        for f, c in counts["per_customer"][rid].items():
            C[i, idx[f]] = c
    return ids, C, n


def rank_order(citations: np.ndarray) -> np.ndarray:
    """Field indices best-first: citation count DESC, then field name ASC (fields are stored
    alphabetically, so the index IS the name order). Integer keys: no float comparison."""
    return np.lexsort((np.arange(len(citations)), -citations))


def eligible_top(order_fields: List[str], k: int) -> Tuple[List[str], List[str]]:
    chosen, skipped = [], []
    for f in order_fields:
        if f in ELIGIBLE_FIELDS:
            chosen.append(f)
        else:
            skipped.append(f)
        if len(chosen) == k:
            break
    return chosen, skipped


def bootstrap_overlap(C: np.ndarray, n: np.ndarray, fields: List[str], B: int, seed: int) -> Dict[str, Any]:
    """Resample customers with replacement (all of a customer's runs together). Uses Python's
    random.Random, whose stream is stable across versions, with exact integer arithmetic."""
    rng = random.Random(seed)
    a1a = {fields.index(f) for f in A1A_TOP5}
    n_cust = len(n)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n_cust) for _ in range(n_cust)]
        top = rank_order(C[idx].sum(0))[:TOP_K]
        vals.append(len(set(top.tolist()) & a1a))
    arr = np.array(vals)
    dist = {int(k): int((arr == k).sum()) for k in range(TOP_K + 1)}
    return {"B": B, "seed": seed, "overlap_distribution": dist,
            "p_overlap_le_trigger": float((arr <= TRIGGER_MAX_OVERLAP).mean()),
            "overlap_ci95": [int(np.percentile(arr, 2.5)), int(np.percentile(arr, 97.5))],
            "overlap_median": float(np.median(arr))}


def spearman_vs_1a(fields: List[str], rate_of: Dict[str, float], a1a_full: Optional[List[str]]) -> Dict[str, Any]:
    if not a1a_full:
        return {"value": None, "note": "not computed: 1a reports only its top 5, so a full-ranking correlation "
                                       "needs the full 1a ranking (--a1a_ranking). See 'a1a_landing' instead."}
    from scipy.stats import spearmanr
    cited = {f for f in fields if rate_of[f] > 0}
    universe = sorted(cited | set(a1a_full))
    s1b = [rate_of.get(f, 0.0) for f in universe]
    s1a = [-(a1a_full.index(f)) if f in a1a_full else -len(a1a_full) for f in universe]   # unlisted: tied last
    rho = spearmanr(s1b, s1a)[0]
    return {"value": float(rho), "n_fields": len(universe), "note": "over every field cited in either ranking"}


def existing_individual_fields() -> Dict[str, str]:
    """field -> condition_id for every single-field condition with the same parameters as the
    existing individual 15% conditions (read from the engine, never hard-coded)."""
    from dataclasses import fields as dc_fields
    conds = build_all_conditions()
    skip = {"condition_id", "fields", "seed", "description"}
    names = [f.name for f in dc_fields(conds[0]) if f.name not in skip]
    sig = lambda c: json.dumps({n: getattr(c, n) for n in names}, sort_keys=True, default=str)
    template = next(c for c in conds if c.condition_id == "h1_individual_nps_score")
    out: Dict[str, str] = {}
    for c in conds:
        if len(c.fields) == 1 and sig(c) == sig(template):
            out.setdefault(c.fields[0], c.condition_id)
    return out


# ----------------------------------------------------------------------------
# Manifest integrity
# ----------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_decision_sha256(manifest: Dict[str, Any]) -> str:
    """Hash of everything except the context block and the hash itself."""
    body = {k: v for k, v in manifest.items() if k not in ("context", "decision_sha256")}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def verify_manifest(manifest: Dict[str, Any]) -> bool:
    return manifest.get("decision_sha256") == manifest_decision_sha256(manifest)


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load a manifest and refuse one that was edited after it was frozen."""
    m = json.load(open(path))
    if not verify_manifest(m):
        raise ValueError(f"{path}: decision_sha256 does not match the manifest's content: it was edited "
                         f"after it was frozen. Restore the original file.")
    return m


def question_a_groups(dither_list: List[str], existing: Dict[str, str], rank_of: Dict[str, int],
                      triggered: bool) -> Dict[str, Any]:
    """The pre-registered Question A groups, so evaluate_h1.py never re-derives them."""
    entry = lambda f, cid: {"field": f, "condition_id": cid}
    legacy = {"role": "legacy comparison" if triggered else "primary",
              "top": [entry(f, existing[f]) for f in A1A_TOP5],
              "comparison": [entry(f, existing[f]) for f in LEGACY_COMPARISON]}
    primary = None
    if triggered:
        top_ids = {f: existing.get(f, f"{CONDITION_PREFIX}{f}") for f in dither_list}
        comp = [f for f in sorted(existing) if f not in dither_list]
        sens = [f for f in comp if rank_of[f] > TOP_K_WIDE]
        primary = {"role": "primary",
                   "top": [entry(f, top_ids[f]) for f in dither_list],
                   "comparison": [entry(f, existing[f]) for f in comp],
                   "comparison_sensitivity_excluding_1b_ranks_6_to_8": [entry(f, existing[f]) for f in sens]}
    return {"primary_list": "1b" if triggered else "legacy_1a", "legacy_1a": legacy, "primary_1b": primary}


# ----------------------------------------------------------------------------
# The analysis
# ----------------------------------------------------------------------------

def analyze(decisions_dir: Path, baseline_input: Path, canonical: Optional[Path] = None,
            a1a_ranking: Optional[List[str]] = None, agent_results_dir: Optional[Path] = None,
            bootstrap_b: int = BOOTSTRAP_B) -> Dict[str, Any]:
    fields = schema_fields(baseline_input)                      # alphabetical
    missing = set(A1A_TOP5) - set(fields)
    if missing:
        raise ValueError(f"1a fields not in the baseline schema: {sorted(missing)}")
    runs = load_runs(decisions_dir)
    counts = count_citations(runs, fields)
    ids, C, n = build_matrix(counts, fields)
    total_n = int(n.sum())
    cites = C.sum(0)                                            # integer citation counts per field
    order = rank_order(cites)
    ranked = [fields[j] for j in order]
    rank_of = {f: i + 1 for i, f in enumerate(ranked)}
    cnt_of = {f: int(cites[fields.index(f)]) for f in fields}
    rate_of = {f: cnt_of[f] / total_n for f in fields}

    top5, top8 = ranked[:TOP_K], ranked[:TOP_K_WIDE]
    overlap5 = sorted(set(top5) & set(A1A_TOP5))
    overlap8 = sorted(set(top8) & set(A1A_TOP5))
    margin = (cnt_of[ranked[4]] - cnt_of[ranked[5]]) / total_n

    chosen, skipped = eligible_top(ranked, TOP_K)
    existing = existing_individual_fields()
    triggered = len(overlap5) <= TRIGGER_MAX_OVERLAP
    to_add = []
    if triggered:
        for f in chosen:
            if f not in existing:
                to_add.append({"condition_id": f"{CONDITION_PREFIX}{f}", "field": f,
                               "citation_rank": rank_of[f], "citation_rate": rate_of[f],
                               "seed": 500 + sorted(DITHERED_FIELDS).index(f)})
    per_cond_cost = (counts["mean_cost"] * len(ids)) if counts["mean_cost"] else None
    dithered_present = bool(agent_results_dir and Path(agent_results_dir).exists() and
                            any(p.stat().st_size > 0 for p in Path(agent_results_dir).glob("*.decisions.jsonl")))
    canon_ok = canonical is not None and Path(canonical).exists()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "rule_version": RULE_VERSION,
        "frozen": {"a1a_top5": list(A1A_TOP5), "legacy_comparison": list(LEGACY_COMPARISON),
                   "trigger_max_overlap": TRIGGER_MAX_OVERLAP, "top_k": TOP_K, "top_k_wide": TOP_K_WIDE,
                   "bootstrap_B": BOOTSTRAP_B, "bootstrap_seed": BOOTSTRAP_SEED,
                   "n_expected_runs": N_EXPECTED_RUNS, "eligible_fields": sorted(ELIGIBLE_FIELDS)},
        "inputs": {"run_files": {p.name: sha256_file(p) for p, _ in runs},
                   "baseline_input": {"name": Path(baseline_input).name, "sha256": sha256_file(baseline_input)},
                   "canonical": ({"name": Path(canonical).name, "sha256": sha256_file(canonical)} if canon_ok else None)},
        "counts": {"n_customers": len(ids), "n_runs": len(runs), "n_decisions_used": total_n,
                   "n_decisions_excluded": counts["excluded"], "n_items": counts["n_items"],
                   "n_unmatched_items": counts["n_unmatched"],
                   "unmatched_share": (counts["n_unmatched"] / counts["n_items"]) if counts["n_items"] else 0.0},
        "citation_table": [{"rank": i + 1, "field": f, "citations": cnt_of[f], "rate": rate_of[f],
                            "eligible": f in ELIGIBLE_FIELDS, "in_1a_top5": f in A1A_TOP5,
                            "existing_condition": existing.get(f)} for i, f in enumerate(ranked)],
        "replication": {
            "top5_1b": top5, "top5_overlap": len(overlap5), "top5_overlap_fields": overlap5,
            "top8_overlap": len(overlap8), "top8_overlap_fields": overlap8,
            "margin_rank5_rank6": margin,
            "a1a_landing": {f: {"rank_1a": i + 1, "rank_1b": rank_of[f], "rate_1b": rate_of[f]}
                            for i, f in enumerate(A1A_TOP5)}},
        "decision": {"triggered": triggered,
                     "rule": f"top-5 overlap <= {TRIGGER_MAX_OVERLAP}: add an individual 15% condition for every "
                             f"eligible 1b top-5 field without one (at most {TOP_K}); otherwise add nothing",
                     "dither_list_1b": chosen, "skipped_ineligible": skipped,
                     "covered_by_existing": {f: existing[f] for f in chosen if f in existing},
                     "conditions_to_add": to_add},
        "question_a_groups": question_a_groups(chosen, existing, rank_of, triggered),
    }
    manifest["decision_sha256"] = manifest_decision_sha256(manifest)
    manifest["context"] = {
        "note": "Not covered by decision_sha256.",
        "bootstrap": bootstrap_overlap(C, n, fields, bootstrap_b, BOOTSTRAP_SEED),
        "frozen_defaults_used": bootstrap_b == BOOTSTRAP_B,
        "spearman": spearman_vs_1a(fields, rate_of, a1a_ranking),
        "top_unmatched": counts["unmatched"].most_common(8),
        "estimated_cost_per_condition_usd": per_cond_cost,
        "estimated_total_cost_usd": (per_cond_cost * len(to_add)) if per_cond_cost else None,
        "sequencing": {"dithered_decisions_already_present": dithered_present,
                       "note": ("WARNING: dithered-condition decisions already exist. The rule is meant to be "
                                "evaluated before any are run; record this in the write-up."
                                if dithered_present else "no dithered-condition decisions found (as intended)")},
    }
    return manifest


def print_report(res: Dict[str, Any]) -> None:
    r, d, c, ctx = res["replication"], res["decision"], res["counts"], res["context"]
    print(f"\n{'=' * 64}\nH1 BASELINE REPLICATION CHECK ({res['rule_version']})\n{'=' * 64}")
    print(f"customers {c['n_customers']} x runs {c['n_runs']} -> {c['n_decisions_used']} decisions used, "
          f"{c['n_decisions_excluded']} excluded; unmatched citations {c['unmatched_share']:.1%} "
          f"({c['n_unmatched_items']} of {c['n_items']} items)")
    print("\n1b top 8 by citation count (ties: field name A-Z):")
    for row in res["citation_table"][:TOP_K_WIDE]:
        tag = ("  [1a top 5]" if row["in_1a_top5"] else "") + ("" if row["eligible"] else "  [cannot be dithered]")
        print(f"  {row['rank']:2d}. {row['field']:26s} {row['rate']:6.1%}{tag}")
    print(f"  rate margin between rank 5 and rank 6: {r['margin_rank5_rank6']:.2%}")
    print(f"\ntop-5 overlap with 1a: {r['top5_overlap']} of 5  {r['top5_overlap_fields']}")
    print(f"top-8 overlap with 1a: {r['top8_overlap']} of 5")
    print("where 1a's five landed in 1b:")
    for f, v in r["a1a_landing"].items():
        print(f"  1a #{v['rank_1a']} {f:26s} -> 1b #{v['rank_1b']:<3d} ({v['rate_1b']:.1%} of decisions)")
    b = ctx["bootstrap"]
    print(f"bootstrap over customers (B={b['B']}): overlap median {b['overlap_median']:.0f}, 95% interval "
          f"{b['overlap_ci95']}, P(overlap <= {TRIGGER_MAX_OVERLAP}) = {b['p_overlap_le_trigger']:.1%}")
    sp = ctx["spearman"]
    print(f"spearman vs 1a: {sp['value'] if sp['value'] is not None else sp['note']}")
    print(f"\nTRIGGER (overlap <= {TRIGGER_MAX_OVERLAP}): {'FIRED' if d['triggered'] else 'not fired'}")
    if d["skipped_ineligible"]:
        print(f"  skipped (cannot be dithered): {d['skipped_ineligible']}")
    if d["triggered"]:
        print(f"  1b list for Question A (primary): {d['dither_list_1b']}")
        print(f"  already covered by an existing condition: {d['covered_by_existing']}")
        print(f"  NEW conditions to generate ({len(d['conditions_to_add'])}): {[x['field'] for x in d['conditions_to_add']]}")
        if ctx["estimated_total_cost_usd"]:
            print(f"  estimated cost: ${ctx['estimated_total_cost_usd']:.2f} at standard pricing")
        print("  next: python generate_h1_replication_conditions.py --manifest <this file> --out experiments_output")
    else:
        print("  Question A stands as designed (a replication under a changed instrument). Nothing to generate.")
    print(f"\ndecision_sha256: {res['decision_sha256']}")
    print(f"{ctx['sequencing']['note']}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions_dir", required=True, type=Path)
    ap.add_argument("--baseline_input", required=True, type=Path)
    ap.add_argument("--canonical", type=Path, default=None,
                    help="canonical_customers.json: its hash is recorded so later generation can verify it is unchanged")
    ap.add_argument("--a1a_ranking", type=Path, default=None,
                    help="optional JSON list of ALL fields in 1a's rank order, for a full Spearman correlation")
    ap.add_argument("--agent_results_dir", type=Path, default=Path("experiments_output/agent_results/decisions"))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--out", type=Path, default=Path("experiments_output/evaluation/h1_replication_manifest.json"))
    a = ap.parse_args()
    ranking = json.load(open(a.a1a_ranking)) if a.a1a_ranking else None
    res = analyze(a.decisions_dir, a.baseline_input, a.canonical, ranking, a.agent_results_dir, a.bootstrap)
    if a.out.exists():
        try:
            old = load_manifest(a.out)
        except ValueError as e:
            print(f"\n❌ {e}")
            return 1
        if old["decision_sha256"] == res["decision_sha256"]:
            print_report(res)
            print(f"Manifest already frozen with IDENTICAL decision content ({a.out}); left untouched.")
            return 0
        print(f"\n❌ A frozen manifest already exists at {a.out} with DIFFERENT decision content "
              f"({old['decision_sha256'][:12]}... vs {res['decision_sha256'][:12]}...). Refusing to overwrite: the "
              f"trigger must not be re-evaluated. If this is a deliberate re-run (e.g. a rehearsal), delete the "
              f"manifest first.")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print_report(res)
    print(f"Saved (frozen): {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

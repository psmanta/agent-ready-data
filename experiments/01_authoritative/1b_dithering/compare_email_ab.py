#!/usr/bin/env python3
"""
Email A/B smoke test -- step 2: did the email change move the agent's behavior beyond ordinary run-to-run noise?
==================================================================================================================
Arms (5 runs of the same 30 customers each):
    OLD   the original smoke baseline, run weeks ago           (old_dir/baseline/decisions/run1..5)
    A     the old-email input, RE-RUN NOW                      (out_dir/old_emails_rerun/decisions/run1..5)   <- the control
    B     the new-email input, run now                         (out_dir/new_emails/decisions/run1..5)

The agent is not perfectly repeatable, so two runs of the SAME input sometimes disagree. That is the noise floor.
The email mattered only if runs on DIFFERENT emails disagree MORE than runs on the same emails.

    noise floor  = how often two runs disagree WITHIN an arm (A with A, B with B)
    cross        = how often a run in A disagrees with a run in B
    excess       = cross - noise floor            (about 0 if the email did not matter)

A customer-level bootstrap gives an interval for the excess. With 30 customers this can only detect LARGE effects;
the interval's half-width says how large. OLD vs A is a drift check (did the model or prompt change since the first smoke).

Usage (from 1b_dithering/):
    python3 compare_email_ab.py --old_dir experiments_output --out_dir experiments_output_email_ab
"""
import argparse
import itertools
import json
import random
import re
import sys
from math import comb
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

LANG = re.compile(r"example\.com|\bdomain\b|\bsynthetic\b|\bfake\b|\bplaceholder\b|\bdemo\b|\btest (data|record|customer)\b", re.I)


def load_arm(run_dir: Path) -> Dict[str, List[dict]]:
    """record_id -> the (up to 5) valid decision rows from run1..run5."""
    per: Dict[str, List[dict]] = {}
    files = sorted(run_dir.glob("run*.decisions.jsonl"))
    if not files:
        raise SystemExit(f"❌ no run*.decisions.jsonl in {run_dir}")
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("business_decision") in ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"):
                per.setdefault(row["record_id"], []).append(row)
    return per


def disagree(xs: List[str], ys: List[str], same_arm: bool) -> Optional[float]:
    pairs = list(itertools.combinations(range(len(xs)), 2)) if same_arm else list(itertools.product(range(len(xs)), range(len(ys))))
    if not pairs:
        return None
    return sum((xs[i] != (xs[j] if same_arm else ys[j])) for i, j in pairs) / len(pairs)


def majority(ds: List[str]) -> str:
    top = sorted(((ds.count(d), d) for d in set(ds)), reverse=True)
    return "TIE" if len(top) > 1 and top[0][0] == top[1][0] else top[0][1]


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval for k of n: the honest way to bound a rare event like '1 customer of 30'."""
    def cdf(kk: int, p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))
    low, high = 0.0, 1.0
    if k > 0:                                    # smallest p at which seeing k or more still has probability alpha/2
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if 1 - cdf(k - 1, mid) < alpha / 2 else (lo, mid)
        low = hi
    if k < n:                                    # largest p at which seeing k or fewer still has probability alpha/2
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cdf(k, mid) > alpha / 2 else (lo, mid)
        high = lo
    return low, high


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def compare(X: Dict[str, List[dict]], Y: Dict[str, List[dict]], seed: int = 20260101, boot: int = 4000) -> dict:
    ids = sorted(set(X) & set(Y))
    rows = []
    for rid in ids:
        dx, dy = [r["business_decision"] for r in X[rid]], [r["business_decision"] for r in Y[rid]]
        if len(dx) < 2 or len(dy) < 2:
            continue
        wx, wy, c = disagree(dx, dx, True), disagree(dy, dy, True), disagree(dx, dy, False)
        rows.append({"within": (wx + wy) / 2, "cross": c, "x_major": majority(dx), "y_major": majority(dy)})
    n = len(rows)
    if n == 0:
        return {"n": 0}
    excess = [r["cross"] - r["within"] for r in rows]
    rng = random.Random(seed)
    boots = sorted(mean([excess[rng.randrange(n)] for _ in range(n)]) for _ in range(boot))
    return {"n": n, "within": mean([r["within"] for r in rows]), "cross": mean([r["cross"] for r in rows]),
            "excess": mean(excess), "lo": boots[int(0.025 * boot)], "hi": boots[int(0.975 * boot) - 1],
            "majority_changed": sum(r["x_major"] != r["y_major"] for r in rows)}


MENTION = re.compile(r"\be-?mail (address|domain|provider)\b|\bdomain\b|@|example\.com", re.I)


def changed_customers(X: Dict[str, List[dict]], Y: Dict[str, List[dict]], label_x: str, label_y: str, extra: Dict[str, Dict[str, List[dict]]]):
    """Customers whose 5-run majority decision differs between two arms: votes, confidence, wording."""
    out, count = [], 0
    for rid in sorted(set(X) & set(Y)):
        dx, dy = [r["business_decision"] for r in X[rid]], [r["business_decision"] for r in Y[rid]]
        if len(dx) < 2 or len(dy) < 2 or majority(dx) == majority(dy):
            continue
        count += 1
        def votes(ds): return ", ".join(f"{d.replace('_PRIORITY', '')} x{n}" for d, n in Counter(ds).most_common())
        def conf(rows): return mean([r.get("agent_confidence") for r in rows])
        def says(rows): return sum(bool(MENTION.search(str(r.get("decision_reasoning", "")))) for r in rows)
        out.append(f"  {rid}")
        out.append(f"     {label_x:<10} {votes(dx):<28} mean conf {conf(X[rid]):.2f} | email/domain words in {says(X[rid])} of {len(dx)} reasonings")
        out.append(f"     {label_y:<10} {votes(dy):<28} mean conf {conf(Y[rid]):.2f} | email/domain words in {says(Y[rid])} of {len(dy)} reasonings")
        for lab, per in ((label_x, X), (label_y, Y)):
            out.append(f"     {lab} first reasoning: {str(per[rid][0].get('decision_reasoning', ''))[:230]}")
    return out, count


def provenance(X: Dict[str, List[dict]], Y: Dict[str, List[dict]]):
    """(records compared, same input_hash, same input_tokens): is anything else different between two arms?"""
    n = same_hash = same_tok = 0
    for rid in sorted(set(X) & set(Y)):
        hx, hy = {r.get("input_hash") for r in X[rid]}, {r.get("input_hash") for r in Y[rid]}
        tx, ty = {r.get("input_tokens") for r in X[rid]}, {r.get("input_tokens") for r in Y[rid]}
        n += 1
        same_hash += bool(hx == hy and None not in hx)
        same_tok += bool(tx == ty and None not in tx)
    return n, same_hash, same_tok


def arm_meta(per: Dict[str, List[dict]]):
    rows = [r for v in per.values() for r in v]
    models = sorted({str(r.get("model")) for r in rows if r.get("model")})
    ts = sorted(str(r["timestamp"]) for r in rows if r.get("timestamp"))
    lat = mean([r.get("processing_time_ms") for r in rows])
    return ", ".join(models) or "?", (ts[0][:16] if ts else "?"), (ts[-1][:16] if ts else "?"), lat


def arm_stats(per: Dict[str, List[dict]]) -> dict:
    rows = [r for v in per.values() for r in v]
    return {"decisions": len(rows),
            "mean_confidence": mean([r.get("agent_confidence") for r in rows]),
            "mean_output_tokens": mean([r.get("output_tokens") for r in rows]),
            "mean_cost": mean([r.get("cost_usd") for r in rows]),
            "test_or_domain_language": sum(bool(LANG.search(str(r.get("decision_reasoning", "")))) for r in rows)}


def verdict(res: dict) -> str:
    if res["n"] == 0:
        return "not enough data"
    lo, hi, ex = res["lo"], res["hi"], res["excess"]
    if lo > 0:
        return f"DIFFERENT EMAILS DISAGREE MORE THAN THE NOISE (excess {ex:+.1%}, 95% interval {lo:+.1%} to {hi:+.1%}): the email change probably mattered; look closer."
    if hi < 0:
        return f"Different emails disagree LESS than the noise (excess {ex:+.1%}); unexpected but not a concern for the email question."
    size = (hi - lo) / 2
    if abs(ex) < 0.05:
        k, n = res["majority_changed"], res["n"]
        elo, ehi = clopper_pearson(k, n)
        return (f"NO SIGN THE EMAIL MATTERED (excess {ex:+.1%}, 95% interval {lo:+.1%} to {hi:+.1%}). "
                f"{k} of {n} customers changed their decision; the exact 95% range for the share the email could change is {elo:.1%} to {ehi:.1%}, "
                f"so a modest effect cannot be ruled out with only {n} customers.")
    return (f"INCONCLUSIVE: excess {ex:+.1%} with an interval ({lo:+.1%} to {hi:+.1%}) that includes zero. Too few customers to tell; "
            f"treat as possible. A larger smoke run would settle it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old_dir", type=Path, default=Path("experiments_output"))
    ap.add_argument("--out_dir", type=Path, default=Path("experiments_output_email_ab"))
    a = ap.parse_args()
    OLD = load_arm(a.old_dir / "baseline" / "decisions")
    A = load_arm(a.out_dir / "old_emails_rerun" / "decisions")
    B = load_arm(a.out_dir / "new_emails" / "decisions")

    print("=" * 96 + "\nEMAIL A/B: did replacing example.com with realistic emails change the agent's behavior?\n" + "=" * 96)
    print("\nPER-ARM SNAPSHOT (30 customers x 5 runs)")
    print(f"  {'arm':<26}{'decisions':>10}{'mean conf':>11}{'out tokens':>12}{'mean cost':>11}{'test/domain words':>19}")
    for name, per in (("OLD smoke (weeks ago)", OLD), ("A: old emails, re-run now", A), ("B: NEW emails", B)):
        s = arm_stats(per)
        print(f"  {name:<26}{s['decisions']:>10}{s['mean_confidence']:>11.3f}{s['mean_output_tokens']:>12.0f}{s['mean_cost']:>11.5f}{s['test_or_domain_language']:>19}")

    drift = compare(OLD, A)
    main_ = compare(A, B)
    print("\nDRIFT CHECK: OLD smoke vs A (same emails, different day). Should be about 0; if not, the model/prompt moved.")
    if drift["n"]:
        print(f"  noise floor {drift['within']:.1%} | cross {drift['cross']:.1%} | excess {drift['excess']:+.1%} ({drift['lo']:+.1%} to {drift['hi']:+.1%}) | customers whose majority decision changed: {drift['majority_changed']} of {drift['n']}")
    print("\nTHE QUESTION: A (old emails) vs B (new emails), both run now")
    if main_["n"]:
        print(f"  runs on the SAME emails disagree on   {main_['within']:.1%} of customers' decisions (the noise floor)")
        print(f"  runs on DIFFERENT emails disagree on  {main_['cross']:.1%}")
        print(f"  excess over noise                     {main_['excess']:+.1%}  (95% interval {main_['lo']:+.1%} to {main_['hi']:+.1%})")
        print(f"  customers whose 5-run majority decision changed: {main_['majority_changed']} of {main_['n']}")
    print("\nVERDICT\n  " + verdict(main_))
    print("\nPROVENANCE: is anything other than the email different between the arms?")
    print(f"  {'arm':<26}{'model':<30}{'first run':<18}{'last run':<18}{'mean ms/call':>13}")
    for name, per in (("OLD smoke (weeks ago)", OLD), ("A: old emails, re-run now", A), ("B: NEW emails", B)):
        m, t0, t1, lat = arm_meta(per)
        print(f"  {name:<26}{m:<30}{t0:<18}{t1:<18}{lat:>13.0f}")
    n1, h1, t1_ = provenance(OLD, A)
    n2, h2, t2_ = provenance(A, B)
    print(f"  OLD vs A : identical input for {h1}/{n1} records; identical input token count for {t1_}/{n1}")
    print(f"  A vs B   : identical input for {h2}/{n2} records (expected 0: the email differs); identical token count for {t2_}/{n2} (can differ: the new email is a different length)")
    if n1:
        same_model = arm_meta(OLD)[0] == arm_meta(A)[0]
        if h1 != n1:
            print("  Reading: the INPUT differs between OLD and A for some records. They are not a clean repeat; find out why.")
        elif t1_ != n1:
            print("  Reading: the same input produced a DIFFERENT token count, so the prompt or code changed between OLD and A. Check git history for business_decision_agent.py.")
        elif not same_model:
            print(f"  Reading: same input and token count, but the model string differs ({arm_meta(OLD)[0]} versus {arm_meta(A)[0]}). That alone could explain different decisions.")
        else:
            print("  Reading: same input text, same token count (so the same prompt text) and same model string. That rules out a change in our prompt or data.")
            print("           It does NOT rule out things the rows do not record: temperature and other settings, the libraries that build the request,")
            print("           or how the model was served. Check those (git history, library versions, how the first run was launched) before blaming the model.")
    print("\nWHO CHANGED (customers whose 5-run majority decision differs; votes show whether the agent was already torn)")
    for title, X, Y, lx, ly in (("A vs B (the email change)", A, B, "A old", "B new"), ("OLD smoke vs A (NO change at all, just a different day)", OLD, A, "OLD", "A")):
        lines, count = changed_customers(X, Y, lx, ly, {})
        print(f"\n {title}: {count} customer(s)")
        print("\n".join(lines) if lines else "  none")
    if drift["n"] and drift["lo"] > 0:
        print("  CAUTION: the drift check shows the agent moved since the first smoke run, so trust A vs B, not OLD vs B.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

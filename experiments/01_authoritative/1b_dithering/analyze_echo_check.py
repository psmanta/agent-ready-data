#!/usr/bin/env python3
"""
Echo check over the H4 implausible conditions, plus the clean-baseline floor.

For each dithered customer: what did the agent do to the value it was shown
(echo / echo with % added / converted / sign dropped / omitted)? Then the same
check on that customer's CLEAN baseline runs against the original value, which
is the floor: agents legitimately convert ("4+ years" for 48 months) and skip
fields in clean data too, so a conversion or omission is only notable relative
to that rate.

Also prints the four hand-checked customers as a sanity check against what we
found by reading shown value vs. written text.

Run from experiments/01_authoritative/1b_dithering/ (free, no API calls):
  python3 analyze_echo_check.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from evaluate_core import classify_value_echo, clean_baseline_echo_floor

CONDITIONS = {
    "h4_churn_risk_score_implausible": "churn_risk_score",
    "h4_total_spend_implausible": "total_spend",
    "h4_tenure_months_implausible": "tenure_months",
}
# What we found by hand, comparing shown value to written text.
KNOWN = {
    ("h4_churn_risk_score_implausible", "CUST_000003"): "echo",
    ("h4_churn_risk_score_implausible", "CUST_000022"): "echo",
    ("h4_churn_risk_score_implausible", "CUST_000009"): "echo_pct_marker",
    ("h4_tenure_months_implausible", "CUST_000007"): "converted",
}
CLASSES = ["echo", "echo_pct_marker", "converted", "magnitude_only", "omitted"]

baseline = json.load(open("experiments_output/baseline/baseline_reference.json"))["customers"]


def wilson(k, n, z=1.96):
    from math import sqrt
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fmt(counter, n):
    return "  ".join(f"{k}={counter.get(k, 0)}" for k in CLASSES) + f"   (n={n})"


def fmt_ci(counter, n):
    """Rates with 95% Wilson intervals (valid for the dithered rows: one
    observation per customer)."""
    parts = []
    for k in CLASSES:
        if counter.get(k, 0):
            lo, hi = wilson(counter[k], n)
            parts.append(f"{k} {counter[k]/n:.0%} [{lo:.0%}-{hi:.0%}]")
    return "   ".join(parts)


print("=" * 78)
print("DITHERED CONDITIONS: what the agent did to the value it was shown")
print("=" * 78)
known_seen = {}
floor = {}
for cond, field in CONDITIONS.items():
    ref = json.load(open(f"experiments_output/conditions/{cond}/dither_reference.json"))
    decisions = {}
    with open(f"experiments_output/conditions/{cond}/decisions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            decisions[d["record_id"]] = d

    dith, dith_conv, examples = Counter(), Counter(), []
    floor_recs = []
    n_d = 0
    for r in ref:
        if not r.get("_dither_applied") or field not in r.get("_dither_fields", []):
            continue
        d = decisions.get(r["record_id"])
        if d is None:
            continue
        res = classify_value_echo(r[field], d.get("decision_reasoning"))
        dith[res["echo_class"]] += 1
        n_d += 1
        if res["echo_class"] == "converted":
            dith_conv[(res["factor"], res["unit_after"], bool(res["low_specificity"]))] += 1
            text = d.get("decision_reasoning") or ""
            i = text.find(res["matched_text"])
            examples.append(f'{r["customer_id"]} shown={r[field]} {res["factor"]} '
                            f'unit={res["unit_after"]} low_spec={res["low_specificity"]}  '
                            f'...{text[max(0, i-45):i+35]}...')
        if (cond, r["customer_id"]) in KNOWN:
            known_seen[(cond, r["customer_id"])] = res

        floor_recs.append({"customer_id": r["customer_id"], "dither_fields": r["_dither_fields"],
                           "dither_original": r["_dither_original"]})

    floor = clean_baseline_echo_floor(floor_recs, field, baseline)
    base, n_b = floor["class_counts"], floor["n_runs"]

    print(f"\n{cond}")
    print(f"  dithered  : {fmt(dith, n_d)}")
    print(f"              {fmt_ci(dith, n_d)}")
    if dith_conv:
        print(f"    converted breakdown (factor, unit_after, low_specificity): {dict(dith_conv)}")
        for line in examples[:15]:
            print(f"      {line}")
    print(f"  CLEAN floor (same customers, baseline runs vs. original value):")
    print(f"              {fmt(base, n_b)}")
    print(f"              (5 runs per customer, not independent: treat these rates as point estimates only)")
    if floor["converted_breakdown"]:
        print(f"    clean converted breakdown (factor|unit_after): {floor['converted_breakdown']}")

print("\n" + "=" * 78)
print("HAND-CHECKED CUSTOMERS vs. the echo check")
print("=" * 78)
for key, expected in KNOWN.items():
    res = known_seen.get(key)
    if res is None:
        print(f"  {key[1]} [{key[0]}]: not found in this run")
        continue
    ok = "✅" if res["echo_class"] == expected else "❌"
    print(f"  {ok} {key[1]} [{key[0]}]: expected {expected}, got {res['echo_class']} "
          f"(matched {res['matched_text']!r})")

print("\nHow to read this: 'converted' and 'omitted' only mean something relative")
print("to the CLEAN floor above. For conversions, unit_after separates a real unit")
print("change (/12 with 'years') from a rescale that kept the old label (/12 with")
print("'months'). low_specificity is a 'look at it' flag, not a verdict -- whole-")
print("number years on tenure trip it routinely.")

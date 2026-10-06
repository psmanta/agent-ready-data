#!/usr/bin/env python3
"""
Anchors the clean-baseline floor helpers to numbers already obtained from REAL
data (analyze_echo_check.py on the n=30 x 3 H4 implausible smoke conditions):
if the helpers reproduce them, they are wired correctly. Free; run from
experiments/01_authoritative/1b_dithering/.
"""
import json
from evaluate_core import (load_baseline_customers, clean_baseline_echo_floor,
                           clean_baseline_keyword_floor)

base = load_baseline_customers("experiments_output/baseline/baseline_reference.json")

KNOWN = {  # field: (condition, expected run-level class counts, expected converted breakdown)
    "churn_risk_score": ("h4_churn_risk_score_implausible", {"echo": 135, "omitted": 15}, {}),
    "total_spend":      ("h4_total_spend_implausible",      {"echo": 92, "omitted": 58}, {}),
    "tenure_months":    ("h4_tenure_months_implausible",    {"echo": 52, "converted": 10, "omitted": 88}, {"/12|years": 10}),
}
all_ok = True
for field, (cond, want_counts, want_conv) in KNOWN.items():
    ref = json.load(open(f"experiments_output/conditions/{cond}/dither_reference.json"))
    recs = [{"customer_id": r["customer_id"], "dither_fields": r["_dither_fields"],
             "dither_original": r["_dither_original"]} for r in ref if r.get("_dither_applied")]
    got = clean_baseline_echo_floor(recs, field, base)
    ok = got["class_counts"] == want_counts and got["converted_breakdown"] == want_conv and got["n_runs"] == 150
    all_ok &= ok
    print(f"{'✅' if ok else '❌'} {field}: runs={got['n_runs']} counts={got['class_counts']} converted={got['converted_breakdown']}")
    if not ok:
        print(f"     expected counts={want_counts} converted={want_conv} runs=150")

kw = clean_baseline_keyword_floor(base)
ok = (kw["n_customers"], kw["n_runs"], kw["n_hits"]) == (30, 150, 0)
all_ok &= ok
print(f"{'✅' if ok else '❌'} keyword floor: {kw['n_hits']} hits in {kw['n_runs']} runs over {kw['n_customers']} customers (expected 0 in 150 over 30)")
print("\nALL MATCH" if all_ok else "\nMISMATCH: do not trust the helpers until this is explained")

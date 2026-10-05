#!/usr/bin/env python3
"""
Cross-instrument consistency check on the 3 H4 implausible conditions.

Three independent instruments see the same records:
  - the echo check (deterministic; what happened to the shown number)
  - the keyword scan (deterministic; explicit doubt language anywhere in text)
  - the blind judge (LLM; how the text treats the named figure)
This does not validate any of them -- agreement between instruments is not
accuracy -- but a large disagreement means one is miscounting. Free to run.

Run from experiments/01_authoritative/1b_dithering/ after the judge has run:
  python3 h5_instrument_crosscheck.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from evaluate_core import load_condition

C = Path("experiments_output/conditions")
base = json.load(open("experiments_output/baseline/baseline_reference.json"))["customers"]
# REAL drift = decision vs baseline majority (finalized_ground_truth.json in this
# directory is the Tier-1 fake)
gt_real = {cid: {"final_decision": e["majority_decision"], "stability_tier": "n/a",
                 "decision_source": "baseline_majority"} for cid, e in base.items()}
CONDS = {"h4_churn_risk_score_implausible": "churn_risk_score",
         "h4_total_spend_implausible": "total_spend",
         "h4_tenure_months_implausible": "tenure_months"}

for cond, field in CONDS.items():
    out = {}
    path = Path(f"classifier_output/{cond}.jsonl")
    if not path.exists():
        print(f"\n{cond}: missing {path}"); continue
    for line in open(path):
        r = json.loads(line); out[r["id"]] = r
    rows, n_err = [], 0
    for rec in load_condition(C / cond, gt_real):
        if field not in rec["value_echo"]:
            continue
        j = out.get(f'{rec["customer_id"]}:{field}')
        if j is None:
            continue
        if j.get("error"):
            n_err += 1; continue
        rows.append((rec, j))

    print("\n" + "=" * 74)
    print(f"{cond}   judged={len(rows)}  errors={n_err}   prompt ids: {sorted({j.get('prompt_id', 'untagged') for _, j in rows})}")
    print("=" * 74)
    print("judge category:", dict(Counter(j["category"] for _, j in rows)))

    ct = Counter((j["figure_referenced"], rec["value_echo"][field]["echo_class"] == "omitted") for rec, j in rows)
    print("\njudge 'figure_referenced'  x  echo check 'value omitted from text':")
    print("                          echo: value present    echo: value omitted")
    for fr in (True, False):
        print(f"  judge referenced={str(fr):5s}      {ct[(fr, False)]:5d}                  {ct[(fr, True)]:5d}")
    print("  (different instruments: a figure can be referenced by description with no number,")
    print("   so exact agreement is not expected; a large gap would mean one is miscounting)")

    kw = [(rec, j) for rec, j in rows if rec["h5_keyword_detected"]]
    print(f"\nkeyword-scan hits judged: {len(kw)}   (read these: the texts most likely to be genuine explicit detections)")
    for rec, j in kw:
        print(f"  {rec['customer_id']}: judge={j['category']}  evidence={j['evidence'][:110]!r}")

    ec = [(rec, j) for rec, j in rows if j["category"] == "explicit_concern" and not rec["h5_keyword_detected"]]
    print(f"\njudge explicit_concern with NO keyword hit: {len(ec)}   (judge-only 'detections': read each)")
    for rec, j in ec:
        print(f"  {rec['customer_id']}: evidence={j['evidence'][:140]!r}")

    unref = [rec for rec, j in rows if not j["figure_referenced"]]
    if unref:
        print(f"\njudge said figure NOT referenced: {[r['customer_id'] for r in unref]}")

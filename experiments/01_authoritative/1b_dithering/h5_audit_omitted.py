#!/usr/bin/env python3
"""
Reads the 'omitted' bucket of the echo check, on both sides (clean baseline
runs and dithered texts), to measure what the matcher silently misses.

'omitted' is the catch-all: any phrasing the matcher cannot parse lands there
without a trace ("nearly 4 years" for 46 months, "almost four years"). That
matters most on the CLEAN side, because every dithered-vs-clean comparison
uses it as the floor, and small clean tenure values are phrased in
single-digit years that the two-significant-digit guard rejects, while large
dithered values ("47.5 years") are caught. This script flags omitted texts
whose sentence about the field contains a digit or number word -- candidate
misses -- and prints them for you to read. Free; run from
experiments/01_authoritative/1b_dithering/:
  python3 h5_audit_omitted.py
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from evaluate_core import classify_value_echo, load_baseline_customers

CONDS = {"h4_churn_risk_score_implausible": "churn_risk_score",
         "h4_total_spend_implausible": "total_spend",
         "h4_tenure_months_implausible": "tenure_months"}
ANCHOR = {"churn_risk_score": r"churn",
          "total_spend": r"spen[dt]|spending|lifetime|\$",
          "tenure_months": r"tenure|\byears?\b|\bmonths?\b|long-term|long-tenured|loyal|relationship"}
NUMWORD = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|decade|dozen)\b"
PER_SIDE = 8

base = load_baseline_customers("experiments_output/baseline/baseline_reference.json")
rng = random.Random(0)


def candidate_sentences(text, field):
    out = []
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text or ""):
        if re.search(ANCHOR[field], sent, re.I) and (re.search(r"\d", sent) or re.search(NUMWORD, sent, re.I)):
            out.append(sent.strip())
    return out


def has_anchor(text, field):
    return bool(re.search(ANCHOR[field], text or "", re.I))


for cond, field in CONDS.items():
    ref = [r for r in json.load(open(f"experiments_output/conditions/{cond}/dither_reference.json"))
           if r.get("_dither_applied") and field in r.get("_dither_fields", [])]
    decisions = {}
    for line in open(f"experiments_output/conditions/{cond}/decisions.jsonl"):
        d = json.loads(line); decisions[d["record_id"]] = d

    sides = {"CLEAN baseline runs": [], "DITHERED texts": []}
    clean_texts = []
    for r in ref:
        for run in base.get(r["customer_id"], {}).get("run_details", []):
            t = run.get("decision_reasoning") or ""
            clean_texts.append(t)
            sides["CLEAN baseline runs"].append((r["customer_id"], r["_dither_original"][field], t))
        d = decisions.get(r["record_id"])
        if d:
            sides["DITHERED texts"].append((r["customer_id"], r[field], d.get("decision_reasoning") or ""))

    print("\n" + "=" * 78)
    print(f"{cond}   (field: {field})   distinct clean texts: {len(set(clean_texts))} of {len(clean_texts)} runs")
    print("=" * 78)
    for side, rows in sides.items():
        omitted = [(cid, v, t) for cid, v, t in rows if classify_value_echo(v, t)["echo_class"] == "omitted"]
        silent = [x for x in omitted if not has_anchor(x[2], field)]
        flagged = [(cid, v, t, candidate_sentences(t, field)) for cid, v, t in omitted]
        flagged = [x for x in flagged if x[3]]
        print(f"\n  {side}: {len(omitted)} omitted of {len(rows)}")
        print(f"    - field not discussed at all: {len(silent)}")
        print(f"    - sentence about the field contains a number or number word (candidate matcher misses): {len(flagged)}")
        for cid, v, t, sents in rng.sample(flagged, min(PER_SIDE, len(flagged))):
            print(f"      {cid} value={v}:  {sents[0][:230]}")

print("\nWhat to look for: in the CLEAN candidates, is the value actually stated in a form the matcher")
print("cannot parse ('nearly 4 years', 'almost four years', 'over 4 years')? Those are false 'omitted'")
print("and mean the clean floor undercounts conversions. Candidates that cite a different number")
print("(a purchase count, a lifetime value) are correctly omitted.")

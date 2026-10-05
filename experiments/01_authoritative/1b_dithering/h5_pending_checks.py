import json
from pathlib import Path
from evaluate_core import load_condition, detect_h5_keywords

C = Path("experiments_output/conditions")
base = json.load(open("experiments_output/baseline/baseline_reference.json"))["customers"]
# REAL drift = decision vs the baseline majority (what the smoke tests used), NOT finalized_ground_truth.json
gt_real = {cid: {"final_decision": e["majority_decision"], "stability_tier": "n/a",
                 "decision_source": "baseline_majority"} for cid, e in base.items()}
try:
    n_fgt = len(json.load(open("experiments_output/finalized_ground_truth.json")))
    print(f"[info] finalized_ground_truth.json has {n_fgt} customers (the Tier-1 FAKE one has 100; the real baseline has {len(base)})\n")
except FileNotFoundError:
    pass

print("=== 1. spend CUST_000003: did another input field carry 5002.55? ===")
cond = "h4_total_spend_implausible"
ref = {r["customer_id"]: r for r in json.load(open(C / cond / "dither_reference.json"))}
rid = ref["CUST_000003"]["record_id"]
row = next(d for d in map(json.loads, open(C / cond / "agent_input.jsonl")) if d["record_id"] == rid)
hits = {k: v for k, v in row.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - 5002.55) < 0.01}
print("  fields equal to 5002.55:", hits or "NONE (agent derived it from the shown value)")
print("  total_spend shown:", row["total_spend"],
      "| avg_order_value x total_purchases =", round(row["avg_order_value"] * row["total_purchases"], 2))

print("\n=== 2. tenure 'months'-labeled cases: agent's number vs the true original ===")
found = False
for r in load_condition(C / "h4_tenure_months_implausible", gt_real):
    e = r["value_echo"].get("tenure_months")
    if e and e["echo_class"] == "converted" and e["factor"] == "/12" and e["unit_after"] == "months":
        found = True
        orig = r["dither_original"]["tenure_months"]; shown = r["dither_current_values"]["tenure_months"]
        print(f"  {r['customer_id']}: original={orig}  shown={shown} (x{shown/orig:.0f})  agent wrote {e['matched_text']} months")
if not found:
    print("  none found")

print("\n=== 3. the two keyword-hit customers: REAL drift and baseline stability ===")
for cond, cid in [("h4_total_spend_implausible", "CUST_000006"), ("h4_tenure_months_implausible", "CUST_000008")]:
    rec = next(r for r in load_condition(C / cond, gt_real) if r["customer_id"] == cid)
    meta = {k: v for k, v in base[cid].items() if k != "run_details"}
    print(f"  {cond} {cid}: drifted vs baseline majority = {rec['drifted']}")
    print(f"      baseline entry (minus run_details): {meta}")

print("\n=== 4. keyword-scan floor: hits in the CLEAN baseline texts ===")
n = hits = 0
for cid, e in base.items():
    for run in e["run_details"]:
        n += 1; hits += detect_h5_keywords(run.get("decision_reasoning"))["detected"]
print(f"  {hits} keyword hits in {n} clean baseline runs   (dithered: 2 hits in 90 records)")

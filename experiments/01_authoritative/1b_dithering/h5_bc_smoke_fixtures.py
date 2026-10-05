#!/usr/bin/env python3
"""
H5 smoke test: Case B (confabulation) vs Case C (silent absorption).

Generates, at small n (free, no API cost): the baseline input and the 3
H4 implausible conditions (churn_risk_score, total_spend, tenure_months),
all for the SAME small customer set. After this, run the REAL baseline
(run_baseline.sh) and the REAL agent against each implausible condition
-- that's the actual spend, roughly 240 calls total (150 baseline + 90
implausible) at this n, a few cents to ~$0.50 on Haiku.

Run from experiments/01_authoritative/1b_dithering/:
  python3 h5_bc_smoke_fixtures.py
  caffeinate -i ./run_baseline.sh
  python3 business_decision_agent.py --input experiments_output/conditions/h4_churn_risk_score_implausible/agent_input.jsonl --output experiments_output/conditions/h4_churn_risk_score_implausible/decisions.jsonl
  python3 business_decision_agent.py --input experiments_output/conditions/h4_total_spend_implausible/agent_input.jsonl --output experiments_output/conditions/h4_total_spend_implausible/decisions.jsonl
  python3 business_decision_agent.py --input experiments_output/conditions/h4_tenure_months_implausible/agent_input.jsonl --output experiments_output/conditions/h4_tenure_months_implausible/decisions.jsonl
  python3 analyze_h5_bc_smoke.py
"""
import subprocess, sys

N = 30
SEED = 777
CONDITIONS = ["h4_churn_risk_score_implausible", "h4_total_spend_implausible",
              "h4_tenure_months_implausible"]

print(f"Generating ground truth, baseline input, and {len(CONDITIONS)} "
      f"implausible conditions at n={N} (free, no API cost)...")
for cid in CONDITIONS:
    result = subprocess.run(
        [sys.executable, "generate_dithered_data.py", "--n", str(N), "--seed", str(SEED),
         "--condition", cid], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:]); print(result.stderr[-2000:])
        raise SystemExit(f"Generation failed for {cid}")
    print(f"  generated {cid}")

print(f"\nDone. Next, run the REAL baseline and REAL agent calls:")
print(f"  caffeinate -i ./run_baseline.sh")
for cid in CONDITIONS:
    print(f"  python3 business_decision_agent.py "
          f"--input experiments_output/conditions/{cid}/agent_input.jsonl "
          f"--output experiments_output/conditions/{cid}/decisions.jsonl")
print(f"\nThen: python3 analyze_h5_bc_smoke.py")

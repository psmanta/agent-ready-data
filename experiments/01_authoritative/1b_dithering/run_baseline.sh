#!/bin/bash
# ============================================================================
# Run Baseline — Experiment 1b: Dithering
# ============================================================================
# The Agentic Data Contract · Pillar 1: Authoritative
#
# Runs the business decision agent 5 times against the same clean baseline
# dataset at temperature 0.0. The 5-run majority vote becomes the ground
# truth decision for each customer; the full distribution across runs
# classifies each customer's baseline stability (stable, lightly-boundary,
# deeply-boundary).
#
# Same prompt is used for baseline and dither condition runs — see
# business_decision_agent.py docstring for why this matters (H5).
#
# caffeinate prevents macOS from sleeping during a long run.
# If you're on Windows or Linux, ensure your system won't sleep during
# execution.
#
# Usage:
#   ./run_baseline.sh
#   caffeinate -i ./run_baseline.sh   (recommended for macOS)
# ============================================================================

set -e  # exit on first error — a failed run should not silently continue

INPUT="experiments_output/baseline/agent_input/baseline_customers.jsonl"
OUTPUT_DIR="experiments_output/baseline/decisions"
MODEL="claude-haiku-4-5-20251001"
TEMPERATURE=0.0
N_RUNS=5

if [ ! -f "$INPUT" ]; then
    echo "Error: baseline input not found: $INPUT"
    echo "Run generate_dithered_data.py first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Baseline Run — Experiment 1b"
echo "============================================================"
echo "Input:       $INPUT"
echo "Runs:        $N_RUNS"
echo "Model:       $MODEL"
echo "Temperature: $TEMPERATURE"
echo ""

for i in $(seq 1 $N_RUNS); do
    echo "------------------------------------------------------------"
    echo "Run $i of $N_RUNS"
    echo "------------------------------------------------------------"

    python3 business_decision_agent.py \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/run${i}.decisions.jsonl" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE"

    echo ""
done

echo "============================================================"
echo "All $N_RUNS baseline runs complete."
echo "Aggregating into baseline reference..."
echo "============================================================"

python3 aggregate_baseline.py \
    --decisions_dir "$OUTPUT_DIR" \
    --record_id_map "experiments_output/baseline/record_id_map.json" \
    --n_runs "$N_RUNS" \
    --output "experiments_output/baseline/baseline_reference.json"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"

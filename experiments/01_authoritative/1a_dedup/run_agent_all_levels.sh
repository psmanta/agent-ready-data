#!/bin/bash
# run_agent_all_levels.sh - Run agent on all duplication levels

set -e

echo ""
echo "========================================"
echo "🤖 RUNNING AGENT ON ALL LEVELS"
echo "========================================"
echo ""

MODEL="claude-3-haiku-20240307"
INPUT_DIR="experiments_output/agent"
OUTPUT_DIR="experiments_output/agent_results/decisions"

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

TOTAL_RECORDS=0
TOTAL_COST=0

for level in 0 10 20 30 40 50 75 100; do
    echo ""
    echo "========================================"
    echo "📦 Processing ${level}% duplication level"
    echo "========================================"
    echo ""
    
    INPUT_FILE="$INPUT_DIR/customers_dup_${level}pct.jsonl"
    OUTPUT_FILE="$OUTPUT_DIR/customers_dup_${level}pct.decisions.jsonl"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "⚠️  Warning: Input file not found: $INPUT_FILE"
        echo "   Skipping this level..."
        continue
    fi
    
    # Remove old output if exists
    rm -f "$OUTPUT_FILE"
    
    python business_decision_agent.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --model "$MODEL"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error processing ${level}% level"
        exit 1
    fi
    
    # Extract stats from summary
    SUMMARY_FILE="$OUTPUT_DIR/customers_dup_${level}pct.decisions.summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        LEVEL_RECORDS=$(python -c "import json; print(json.load(open('$SUMMARY_FILE'))['total_records'])")
        LEVEL_COST=$(python -c "import json; print(json.load(open('$SUMMARY_FILE'))['total_cost_usd'])")
        
        TOTAL_RECORDS=$((TOTAL_RECORDS + LEVEL_RECORDS))
        TOTAL_COST=$(python -c "print(round($TOTAL_COST + $LEVEL_COST, 4))")
        
        echo ""
        echo "   📊 Records: $LEVEL_RECORDS"
        echo "   💰 Cost: \$$LEVEL_COST"
    fi
done

echo ""
echo "========================================"
echo "✅ ALL LEVELS COMPLETE"
echo "========================================"
echo ""
echo "📊 FINAL SUMMARY:"
echo "   Total records processed: $TOTAL_RECORDS"
echo "   Total cost: \$$TOTAL_COST"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.decisions.jsonl 2>/dev/null || echo "   (no files found)"
echo ""
echo "Next step: Run evaluation to measure decision quality"
echo ""
EOF

chmod +x run_agent_all_levels.sh


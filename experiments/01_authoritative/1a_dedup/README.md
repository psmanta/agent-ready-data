# Experiment 1a: Duplication Impact on Agent Decision Making

## Objective
Empirically quantify how duplicate customer records affect the quality, consistency, and aggregate accuracy of AI agent decision making. This experiment is part of the **Authoritative Data** pillar of the Agent-Ready Data research framework.

### The Core Question
When an agent makes business decisions (customer prioritization, tier assignment, resource allocation) using a dataset that contains duplicate records, how does the duplication level affect decision consistency, distribution accuracy, and downstream business outcomes?

---

## Hypotheses

### H1 — Record-Level Decision Integrity
The presence of duplicate records for a customer does not affect the agent's decision on any individual record. Each record should be classified correctly in isolation regardless of how many duplicates exist.

### H2 — Cluster-Level Decision Consistency
When multiple records exist for the same customer, the agent produces consistent decisions across all records in that cluster. Even if individual records are classified correctly, conflicting decisions across a cluster represent a data quality failure.

### H3 — Macro/Aggregate Decision Drift
Duplicate data causes systematic bias in aggregate business decisions. For example, a headcount recommendation based on customer support tier distribution will drift as duplication rate increases, because duplicate records inflate apparent demand.

### H4 — Variation Sensitivity / Field Importance
Not all field-level variation in duplicates is equal. Variation in tier-critical fields (e.g. `total_spend`, `lifetime_value_estimate`) will affect agent decisions more than variation in identity fields (e.g. `name`, `email`). The agent's cited `key_factors` reveal which fields drive decisions and whether this changes under duplication.

### H5 — Duplication Rate Threshold
The relationship between duplication rate and decision quality degradation is non-linear. There exists a threshold below which duplication has minimal effect, above which quality degrades measurably. Identifying this threshold has practical value for data quality teams.

### H6 — Segment Distortion
Duplication does not affect all customer segments equally. Preferential duplication of high-value or at-risk customers will disproportionately distort decisions for those segments, with outsized effects on aggregate business recommendations.

---

## Data Generation

**Script:** `generate_customer_data.py`

- **Base customers:** 500 unique synthetic records with 25+ correlated fields
- **Duplication levels:** 0%, 10%, 20%, 30%, 40%, 50%, 75%, 100%
- **Duplicate variation:** minimal (PII only) / moderate (PII + behavioral) / significant (multiple fields)
- **H4 mode:** `--field-variation-mode --vary-fields` — controls exactly which fields are perturbed
- **H6 mode:** `--segment-bias` — biases duplication toward a specific customer segment

**Output structure:**
```
experiments_output/
  agent/
    customers_dup_{level}pct.jsonl     ← agent input (shuffled, no customer_id)
  eval/
    canonical_customers.json           ← ground truth
    cluster_map_{level}pct.json        ← skeleton key (record_id → customer_id)
  metadata/
    generation_stats_{level}pct.json   ← generation statistics
```

---

## Agent

**Script:** `business_decision_agent.py`

A customer prioritization agent that assigns each record one of three priority levels: `HIGH_PRIORITY`, `MEDIUM_PRIORITY`, or `LOW_PRIORITY`. Uses a "guided but not prescriptive" system prompt — the agent weighs all available fields holistically rather than applying a formula.

Each decision output includes:
- `business_decision` — the priority assignment
- `agent_confidence` — 0.0 to 1.0
- `decision_reasoning` — free-text explanation
- `key_factors` — structured list of fields that drove the decision (H4)
- `customer_segment` — segment label for downstream H6 analysis

**Key CLI parameters:**
```
--input       Path to agent JSONL input file
--output      Path for decision output JSONL
--model       Model to use (loaded from model_pricing.json)
--temperature Sampling temperature (default: 0.0)
--max_records Limit records for testing
```

---

## Evaluation

**Script:** `evaluate_decision_quality.py`

Evaluates agent decisions across all duplication levels using the cluster map as the skeleton key. Produces 8 metrics:

| # | Metric | Hypothesis |
|---|--------|-----------|
| 1 | Decision Consistency | H1, H2 |
| 2 | Confidence Stability | H2 |
| 3 | Decision Distribution Shift | H3 |
| 4 | Cost Efficiency | H3 |
| 5 | Reasoning Quality | H2 |
| 6 | Human-Agent Boundary | H5 |
| 7 | Field Importance | H4 |
| 8 | Segment Distortion | H6 |

Outputs a composite quality score (0-100), machine-readable JSON, a markdown report, and visualizations.

**Key CLI parameters:**
```
--decisions_dir    Directory containing agent decision files
--cluster_map_dir  Directory containing cluster map files (eval/)
--output_dir       Output directory for evaluation results
```

---

## Pipeline

```
1. generate_customer_data.py   → Generate synthetic data at all duplication levels
2. business_decision_agent.py  → Run agent decisions for each duplication level
3. evaluate_decision_quality.py → Evaluate and report across all levels
```

**Example end-to-end run:**
```bash
# Step 1: Generate data
python generate_customer_data.py --n 500 --out experiments_output --levels 0,10,20,30,40,50,75,100

# Step 2: Run agent for each level (repeat for each)
python business_decision_agent.py \
    --input experiments_output/agent/customers_dup_50pct.jsonl \
    --output experiments_output/agent_results/decisions/customers_dup_50pct.decisions.jsonl \
    --model claude-haiku-4-5 \
    --temperature 0.0

# Step 3: Evaluate
python evaluate_decision_quality.py \
    --decisions_dir experiments_output/agent_results/decisions \
    --cluster_map_dir experiments_output/eval \
    --output_dir experiments_output/agent_results/evaluation
```

---

## Configuration

- **Model pricing:** `shared/agents/model_pricing.json`
- **Base agent:** `shared/agents/base_agent.py`
- **LLM factory:** `shared/agents/llm_factory.py`
- **API key:** Set `ANTHROPIC_API_KEY` in `.env`

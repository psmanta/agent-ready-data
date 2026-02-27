# Experiment 1a: Duplication Impact on Agent Decision Making

## Objective
Assess how data duplication levels (0% to 100%) affect agent decision quality, confidence, and automation rates.

### The Core Question
When an agent makes business decisions (e.g., credit approval, customer prioritization, risk assessment) using a dataset that contains duplicates, how does the duplication level affect decision accuracy, consistency, and confidence?

## Hypothesis
As duplication increases, agent decisions will:
- Show lower confidence scores
- Produce more false positives/negatives
- Defer more decisions to human review

## Data Generation
- Base: 500 unique customer records
- Duplication levels: 0%, 10%, 20%, 30%, 40%, 50%, 75%, 100%. More granularity at lower levels where effects may be subtle.
- Controlled confidence scoring based on field similarity
- Ground truth tracked in canonical_map.json

## Pipeline
1. `data_generator_with_confidence.py` → Generate synthetic data
2. `export_eval_csv.py` → Convert JSONL to CSV
3. `validate_eval.py` → Validate ground truth consistency
4. `plot_eval_distributions.py` → Visualize data distributions
5. `dedup_agent.py` → Run agent decisions (TODO)
6. `evaluate_agent.py` → Measure agent performance (TODO)

## Run Data Generation
./run_all_data.sh

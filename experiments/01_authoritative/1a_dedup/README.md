# Experiment 1a: Duplication Impact on Agent Decision Making

**Status:** ✅ Complete | **Results:** [1a_RESULTS.md](./1a_RESULTS.md)
**Charts:** Key visualizations are available in the [charts/](./charts/) directory.

---

## Objective

Empirically quantify how duplicate customer records affect the quality of AI agent decision making. This experiment is part of the **Authoritative Data** pillar of [The Agentic Data Contract](../../../README.md).

### The Core Question
When an agent makes business decisions using a dataset that contains duplicate records, how does the level of duplication affect Decision Quality, which I've introduced as the trustworthiness of the decisions the agent produces.

---

## Hypotheses

The following hypotheses represent assumptions about how the world should operate. Some were confirmed, some were surprisinge. Results are documented in [1a_RESULTS.md](./1a_RESULTS.md).

### H1 — Record-Level Decision Integrity
Each record is processed independently. The presence of duplicates should not affect any individual record's decision.

*Note: H1 is confirmed by architecture, not measurement, the agent has no memory between API calls. This same property means the agent will never notice it has seen the same customer before. See Finding 3 in `1a_RESULTS.md`.*

### H2 — Cluster-Level Decision Consistency
A cluster is a group of records representing the same underlying customer across the dataset. An agent should produce consistent decisions across all records in a cluster.

### H3 — Aggregate Output Inflation
Duplicate records create phantom demand. An agent processing duplicates will inflate it's aggregate output. In this experiment, at 100% duplication, 2882 records represented 1000 unique customers. A human acting on raw output will allocate and plan against a volume that doesn't exist. This inflation is invisible to both the agent and a human consuming the output, which therefore meaningfully distorts macro level business decisions.

### H4 — Field Importance Stability
Certain fields carry more decision making weight than others. Critically, the agent itself determines which fields matter, I do not instruct it. This means stability is not guaranteed. My hypothesis is that the agent will consistently prioritize the same fields regardless of duplication level, and tier critical fields (e.g. `total_spend`, `churn_risk_score`) will outweigh identity fields (e.g. `name`, `email`).

### H5 — Duplication Rate Threshold
Decision quality will degrade gradually as duplication increases, with a measurable inflection point where decision degradation becomes significant. Below that point, a small amount of duplication may be tolerable. Above it, the cost to decision quality becomes unacceptable.

*Note: This finding was unexpected, the data surprised me. This kind of unexpected result validates the research series and is precisely why empirical testing matters over assumption. See `1a_RESULTS.md`.*

### H6 — Segment Distortion
Customer segments here are business defined profile attributes: high-value, medium-value, low-value, and at-risk. These segments are used specifically for H6 analysis. These are distinct from the HIGH/MEDIUM/LOW priority labels the agent assigns. Here, segments are an input characteristic, the other is an agent decision. The hypothesis is duplication does not affect all customer types equally. Some segments may be more vulnerable than others based on the nature of their data.

---

## Data Generation

**Script:** `generate_customer_data.py`

**Key CLI parameters:**
```
--n           Number of base customers (default: 1000)
              Larger values increase API cost per run.

--out         Output directory for generated files

--levels      Comma-separated duplication percentages
              e.g. 0,10,20,30,40,50,75,100

              Levels were chosen to provide granularity at the low end
              where effects may be subtle, and to test extreme conditions
              at the high end. They are not meant to represent typical
              real-world duplication rates.
              Adding or removing levels affects API cost.

--seed        Random seed for reproducibility (default: 42)

--field-variation-mode --vary-fields
              Controls which fields are perturbed in duplicates.
              Used for targeted H4 variation experiments.

--segment-bias
              Biases duplication toward a specific customer segment.
              In this experiment, duplication was distributed uniformly
              across all segments to establish a clean baseline. This flag 
              is available for follow-on runs where you may want to test
              the effect of disproportionate duplication on a specific
              segment. For example, re-running with high value customers
              duplicated at 4x the rate of other segments.
```

**On duplicate variation:** Real world duplicate records are rarely identical. The same customer entered across two systems will have slightly different names, addresses, or behavioral metrics. This experiment introduces controlled variation at three levels:

* minimal - PII only
* moderate - PII + behavioral
* significant - multiple fields

This variation was introduced to represent that natural imperfection. However, the primary focus of this experiment is the effect of duplication itself, not the effect of variation. Variation is introduced to make the dataset realistic, not to test it. That question will be addressed in Experiment 1b.

**Output files:**

```
experiments_output/
  agent/                               ← gitignored, regeneratable
    customers_dup_{level}pct.jsonl     ← Generated customer data, these are input
                                          files for the business decision agent,
                                          one file per duplication level. Records are
                                          shuffled and stripped of customer_id so the
                                          agent sees anonymous records only.

  eval/                                ← included in github repo
    canonical_customers.json           ← Ground truth: all 1,000 base customers with
                                          their true identities and attributes. The
                                          agent never sees this file, it's used 
                                          exclusively for evaluation, providing the
                                          true customer identity that the agent's
                                          anonymous input deliberately omits.

    cluster_map_{level}pct.json        ← The skeleton key: maps every record_id to its
                                          true customer_id. One file per duplication
                                          level. Like the canonical customers file, 
                                          the agent never sees this. It exists solely
                                          to enable evaluation by linking anonymous
                                          records back to their true identity. Essential
                                          for evaluation — without it, we cannot know 
                                          whether two records represent the same 
                                          or different customers.

  metadata/                            ← included in repo
    generation_stats_{level}pct.json   ← Statistics about the generated dataset such as
                                          cluster size distributions, variation level
                                          breakdowns, and segment duplication rates.


```

---

## Agent

**Script:** `business_decision_agent.py`

A customer prioritization agent assigning each record one of three priority levels: `HIGH_PRIORITY`, `MEDIUM_PRIORITY`, or `LOW_PRIORITY`. It uses a guided, but not prescriptive system prompt inatructing the agent to weigh all available fields holistically and determine its own classification approach. It is not told which fields to prioritize or how to weight them.

*A note on prompt design: The prompt provides business context, categories of factors to consider and illustrative examples of each priority level, but no formula or explicit weighting. The agent determines its own decision logic from the data. I acknowledge that the examples (e.g. 'VIP customers with issues = HIGH priority') may implicitly constrain the decision space. This was identified post-run. A spot check of decision reasoning confirms the agent cited data-driven signals (last_purchase_days_ago, churn_risk_score) rather than matching example patterns, suggesting the examples are illustrative rather than prescriptive. A prompt-controlled rerun is planned for full transparency, though I do not expect findings to change materially.*

Each decision is made independently with no memory of previous decisions. The agent processes record 1,000 with no knowledge of records 1–999. This stateless architecture makes the agent consistent and prevents it from ever noticing it has seen the same customer before.

Each decision output includes:
- `business_decision` : The agent's priority assignment (HIGH_PRIORITY, MEDIUM_PRIORITY, or LOW_PRIORITY)
- `agent_confidence` : Self-reported confidence score (0.0 to 1.0)
- `decision_reasoning` : Free-text explanation of the decision
- `key_factors` : Fields the agent itself cited as driving the decision (used for H4 analysis)
- `customer_segment` : The pre-assigned business tier of this customer (high_value, medium_value, low_value, at_risk). This is a property of the customer profile, not the agent's decision. Used only for H6 segment distortion analysis.

**Key CLI parameters:**
```
--input       Path to agent JSONL input file
              (experiments_output/agent/customers_dup_{level}pct.jsonl)
--output      Path for decision output JSONL
              (experiments_output/agent_results/decisions/customers_dup_{level}pct.decisions.jsonl)
--model       Model to use (loaded from model_pricing.json)
--temperature Sampling temperature (default: 0.0)
--max_records Limit records for testing. Used to test functionality while keeping API calls down
```

---

## Evaluation

**Script:** `evaluate_decision_quality.py`

The evaluator compares agent decisions against the cluster map (the skeleton key), that links each anonymous record back to its true customer identity. Without this map, cluster level consistency analysis (H2) would be impossible. We would have no way to know whether two records with conflicting decisions represent the same or different customers.


**Metrics evaluated:**

| # | Metric | Hypothesis | Description |
|---|--------|-----------|-------------|
| 1 | Decision Consistency | H2 | What % of customer clusters received the same decision across all their duplicate records? |
| 2 | Confidence Stability | H2 | Does the agent's confidence score change as duplication increases? Stability here is not reassuring, it suggests the agent has no signal that something is wrong. |
| 3 | Decision Distribution Shift | H3 | Does the ratio of HIGH/MEDIUM/LOW decisions shift as duplication increases, or does the overall priority mix remain stable? |
| 4 | Cost Efficiency | H3 | What % of API spend was wasted on duplicate records that added no decision value? |
| 5 | Reasoning Consistency | H2 | When the agent sees duplicate records for the same customer, does it tell the same decision story? <br><br> Measured using Jaccard similarity, the ratio of shared words to total unique words across two reasoning texts. Score of 1.0 = identical reasoning, 0.0 = no shared words. Chosen because it is deterministic, the same inputs always produce the same score with no AI involvement. |
| 6 | Human-Agent Boundary | H5 | What % of decisions fall below the confidence threshold, suggesting human review may be needed? |
| 7 | Field Importance | H4 | Diagnostic : The agent self-reports which fields most influenced its reasoning. Do the same fields consistently rise to the top as duplication increases, or does the agent's implicit weighting shift? |
| 8 | Segment Distortion | H6 | Diagnostic — customer segments are pre-assigned profile attributes (high-value, medium-value, low-value, at-risk), not agent decisions. Does duplication affect some segments more than others? |

Metrics 1–6 feed a weighted **composite Decision Quality score (0–100)**.

Metrics 7 and 8 are diagnostic — they reveal *where* and *why* quality degrades rather than quantifying the overall score. See `evaluation_metrics.json` for weights and individual scores.

A note on weighting: The weights were deliberately chosen, not derived from statistical optimization. Decision Consistency carries the most weight (30%) because it directly measures whether the same customer receives the same treatment, which I believe is the most fundamental failure mode for a prioritization agent working off data with duplicate records. The remaining weights reflect my judgment about the relative importance of each metric to downstream business outcomes. I acknowledge this is a methodological choice that others might make differently. Weights are fixed across all experiments in this series to enable meaningful cross-experiment comparison. Changing them mid-series would make results incomparable.

**Evaluation outputs** (saved to `experiments_output/agent_results/evaluation/`):

| Output | Description |
|--------|-------------|
| `evaluation_metrics.json` | Machine-readable results for all 8 metrics across all 8 duplication levels. The primary reference file for anyone wanting to analyze or reproduce the findings programmatically |
| `evaluation_report.md` | Auto-generated human-readable narrative report. Timestamped and useful for tracking runs with different parameters. Gitignored and regeneratable. |
| `decision_quality_analysis.png` | 6-chart grid showing all core metrics vs duplication level |
| `boundary_customer_analysis.png` | Blue dots are clusters with all records in agreement. Red have a conflicting decision. There is no confidence level that cleanly separates blue from red. The illusion of confidence is dangerous. |
| `decision_cliff_standalone.png` | Decision consistency drops immediately at 10% duplication and never recovers. There is no safe duplication threshold. |

**Key CLI parameters:**
```
--decisions_dir    Directory containing agent decision files
--cluster_map_dir  Directory containing cluster map files (eval/)
--output_dir       Output directory for evaluation results
```

---

## Pipeline

```
1. generate_customer_data.py    → Generate synthetic data at specified duplication levels
2. run_agent_all_levels.sh      → Run agent decisions across specified levels
3. evaluate_decision_quality.py → Evaluate decisions and generate Decision Quality metrics, charts and reports.
```

**End-to-end run:**
```bash
# Step 1: Generate data
# See Data Generation section above for parameter details and cost implications
python3 generate_customer_data.py \
    --n 1000 \
    --out experiments_output \
    --levels 0,10,20,30,40,50,75,100

# Step 2: Run agent
# caffeinate prevents macOS from sleeping during a long run
# If you're on Windows or Linux, ensure your system won't sleep during execution
caffeinate -i ./run_agent_all_levels.sh

# Step 3: Evaluate
python3 evaluate_decision_quality.py \
    --decisions_dir experiments_output/agent_results/decisions \
    --cluster_map_dir experiments_output/eval \
    --output_dir experiments_output/agent_results/evaluation
```

---

## Configuration

- **`shared/agents/model_pricing.json`** — cost tracking configuration mapping model names to per-token pricing. Review and update manually before each run to ensure pricing reflects current API rates. New models can added manually as they become available, the file format is straightforward. Not updated automatically.
- **`shared/agents/base_agent.py`** — shared base class providing core agent infrastructure: LLM interaction, retry logic, cost tracking, and decision logging. Designed to be reused across all pillars and all experiments in this series.
- **`shared/agents/llm_factory.py`** — shared factory for instantiating LLM clients. Abstracts model provider details so experiments can switch models without modifying agent code. Intended for reuse across the full research series.
- **API key** — set `ANTHROPIC_API_KEY` in a `.env` file at the project root. See project [README.md](../../../README.md) for setup instructions.


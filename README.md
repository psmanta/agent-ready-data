# The Agentic Data Contract

An empirical research series measuring how data conditions affect the quality of AI agent decision making.

---

## Independence & Transparency

This is independent research. Although I work in the AI industry and may use this research in my professional capacity, this project is entirely self-funded using my personal API credits and has no commercial affiliation or sponsorship. No commercial software was used. Everything is written in Python to ensure no vendor software is favored. While the findings may naturally point toward commercial solutions (deduplication software, MDM platforms, etc.), this research uses only open tools. The code is open, the methodology is documented, and the findings are my own.

---

## The Core Thesis

AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that practitioners do not make explicit.

The industry increasingly recognizes pieces of this such as the importance of data quality, data context, data freshness, etc. These are typically Gaddressed in isolation. What's missing is a holistic framework that treats them as clauses of a single contract, and asks: 'what happens to agent decisions when any one of those clauses is violated?'

The current framing is *necessary but not sufficient*.

**Decision Quality** is the goal.

Ensuring the decisions an AI agent produces are trustworthy enough to stake business outcomes on. Data quality is an input. Decision quality is the outcome. This research measures how different data conditions affect that outcome.

---

## The Six Pillars

Through my practice and research I've identified six conditions that data must satisfy for agents to make reliable decisions. These pillars are not exhaustive and are not fixed. Existing pillars may deepen or shift as findings accumulate, and new ones may emerge as this work evolves.

- **Authoritative:** Data tells the truth
- **Comprehensive:** Data tells the whole truth
- **Contextual:** Data tells the truth the business defines
- **Timely:** Data tells the truth for the world as it is now
- **Responsible:** Data tells the truth without creating outcomes we can't roll back
- **Secure:** Data tells the truth, privately and without unintended inference

---

## Methodology

Each experiment follows a consistent pipeline:

1. **Synthetic data generation** — controlled introduction of a specific data condition. Sample size is configurable. Experiments in this series use 1,000 base records as a default but this can be adjusted. Note that larger sample sizes will increase API cost per run.

2. **Business decision agent** — a reasoning agent making HIGH/MEDIUM/LOW customer prioritization decisions. It has no tools and maintains no memory between calls, it makes just decisions. Critically, the agent is not instructed on how to classify customers, it determines its own classification approach from the data. The agent also self-reports a confidence score (0.0–1.0) with each decision, which is used as part of the evaluation.

3. **Structured evaluation** — measuring the effect on decision quality across consistency, confidence, accuracy, cost, and other dimensions. Specific metrics vary by experiment.

### Standing Assumptions

- **Domain:** Experiments use a retail customer prioritization task as the controlled testbed. Conclusions are assumed (not yet claimed) to generalize across domains. Multi-domain replication is scoped as Phase 2.

- **Agent scope:** The agent is a decision-making engine. It receives data, reasons over it, and produces a structured judgment. It has no tools, no memory between calls, and no ability to take action in the world. It represents the cognitive core of a fuller agentic system. I believe this research is valid and relevant because this decision layer is present in virtually every agentic pipeline. If the decision engine is compromised by bad data, everything downstream is compromised as well.

- **Model:** `claude-haiku-4-5-20251001` at temperature 0.0 for reproducibility in Phase 1. The model used may evolve across the series to reflect advances in the field. Each experiment documents the specific model version used.

- **Confidence calibration:** Agent confidence is self-reported and has not been calibrated against ground truth (an objectively correct answer defined independently of the agent). A flat confidence curve does not imply accurate decisions, it may indicate the agent has no signal that something is wrong. Ground truth calibration is future work.

---

## Reproducing the Experiments

### Prerequisites
- Python 3.9+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Setup
```bash
git clone git@github.com:psmanta/agent-ready-data.git
cd agent-ready-data
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Running Experiment 1a
```bash
cd experiments/01_authoritative/1a_dedup

# Step 1: Generate synthetic data
# --n controls the number of base customers (1000 is the default for this experiment)
# Feel free to adjust this number. Note: larger values will increase API cost per run
# --levels controls the duplication percentages tested
# Feel free to adjust these as well but be aware adding or removing levels will also affect API cost
python3 generate_customer_data.py \
    --n 1000 \
    --out experiments_output \
    --levels 0,10,20,30,40,50,75,100

# Step 2: Run agent across all duplication levels
# caffeinate prevents macOS from sleeping during a long run
# If you're running on Windows or Linux, run a similar command
# to ensure your system won't sleep or suspend during execution
caffeinate -i ./run_agent_all_levels.sh

# Step 3: Evaluate
python3 evaluate_decision_quality.py \
    --decisions_dir experiments_output/agent_results/decisions \
    --cluster_map_dir experiments_output/eval \
    --output_dir experiments_output/agent_results/evaluation
```

### Notes on Reproducibility
- Experiment outputs (agent inputs, raw decision files, evaluation reports) are excluded from this repository. They are fully regeneratable, by following the steps above.
- The cluster maps (`experiments_output/eval/`) and decision summaries are included for reference. Note: these reflect the specific synthetic dataset I generated. Reproducing the experiment will generate different synthetic data and may produce different numerical results, though I expect the directional findings to hold.
- Random seed is fixed at 42 in the data generator for reproducible synthetic data within a given run.
- Temperature is fixed at 0.0 for deterministic agent decisions.

---

## Project Structure

```
agent-ready-data/
├── README.md                          ← you are here
├── RESEARCH_NOTES.md                  ← living research log, backlog, decisions
├── requirements.txt
├── .env                               ← required, gitignored (ANTHROPIC_API_KEY)
├── .gitignore                         ← excludes experiment outputs, .env, drafts
├── shared/
│   └── agents/
│       ├── base_agent.py
│       ├── llm_factory.py
│       └── model_pricing.json
└── experiments/
    └── 01_authoritative/
        ├── README.md                  ← pillar overview
        └── 1a_dedup/
            ├── README.md              ← experiment methodology
            ├── 1a_RESULTS.md          ← experiment findings
            ├── generate_customer_data.py
            ├── business_decision_agent.py
            ├── evaluate_decision_quality.py
            └── run_agent_all_levels.sh
```

---

**Experiment Results:** Each completed experiment documents its findings in a dedicated `[experiment_id]_RESULTS.md` file within the experiment folder. Results include hypothesis outcomes, evaluation metrics, key findings, illustrative examples, and implications for subsequent experiments.

---

## Research Output

Following each experiment I publish a LinkedIn article covering the key findings in practitioner-friendly terms.

As the research matures across multiple pillars, I plan to publish more formal write-ups via arXiv and Towards Data Science.

---

## Experiment Status

The following experiments are planned based on current research directions. This list will evolve. Findings from completed experiments open new questions, and interesting threads will be followed wherever they lead.

| Pillar | Experiment | Status |
|--------|-----------|--------|
| Authoritative | 1a: Dedup impact on agent decisions | ✅ Complete |
| Authoritative | 1b: Dithering — untrustworthy values | 🔲 Next |
| Authoritative | 1c: Data quality vs data quantity | 🔲 Backlog |
| Authoritative | 1d: Conflicting authorities | 🔲 Backlog |
| Authoritative | 1e: Incompleteness vs noise | 🔲 Backlog |
| Authoritative | 1f: Provenance blindness | 🔲 Backlog |
| Timely | 2a: Staleness | 🔲 Backlog |
| Contextual | TBD | 🔲 Future |
| Comprehensive | TBD | 🔲 Future |
| Responsible | TBD | 🔲 Future |
| Secure | TBD | 🔲 Future |


## Contact
Peter Manta
- LinkedIn: [linkedin.com/in/peteri_manta](https://linkedin.com/in/peter-manta)

Questions, feedback, and collaboration welcome. If you reproduce these experiments and find different results, I especially want to hear from you.



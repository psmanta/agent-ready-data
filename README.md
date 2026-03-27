# The Agentic Data Contract

A systematic empirical research project testing what happens to AI agents when the data they depend on fails them.

## The Core Thesis

AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that practitioners rarely make explicit.

The current framing of agentic decisions relying on data quality is, in my opinion, somewhat superficial as data quality is necessary, but not sufficient. I'm introducing the idea of **Decision Quality** where the goal is to mitigate the risk of an AI agent making a damaging decision.

This research empirically tests what happens when each condition is violated, measuring the effect on decision quality, consistency, confidence, and aggregate accuracy.


The six clauses of the contract are the **six pillars of AI Agent-ready data**.

## The Six Pillars

| # | Pillar | The Contract Clause |
|---|--------|-------------------|
| 1 | **Authoritative** | Data tells the truth |
| 2 | **Timely** | Data tells the truth for the world as it is now |
| 3 | **Contextual** | Data tells the truth that the business defines |
| 4 | **Comprehensive** | Data tells the whole truth |
| 5 | **Responsible** | Data tells the truth without creating outcomes we can't roll back |
| 6 | **Secure** | Data tells the truth, privately and without unintended inference |

## Methodology

Each experiment uses a consistent pipeline:

1. **Synthetic data generation** — controlled introduction of a specific data quality failure mode
2. **Business decision agent** — a customer prioritization agent making HIGH/MEDIUM/LOW decisions
3. **Structured evaluation** — 8 metrics covering decision consistency, confidence, distribution shift, cost efficiency, reasoning quality, human-agent boundary, field importance, and segment distortion

### Standing Assumptions

- **Domain:** Experiments use a retail customer prioritization task as the controlled testbed. Conclusions are assumed (not yet claimed) to generalize across domains. Multi-domain replication is scoped as Phase 2.
- **Agent scope:** The agent is a decision making engine. It receives data, reasons over it, and produces a structured judgment. It has no tools, no memory between calls, and no ability to take action in the world. It represents the cognitive core of a fuller agentic system. The research is valid and relevant because this decision layer is present in virtually every agentic pipeline. If the decision engine is compromised by bad data, everything downstream is compromised too.
- **Model:** `claude-haiku-4-5-20251001` at temperature 0.0 for reproducibility, unless otherwise noted.
- **Confidence calibration:** Agent confidence is self-reported and has not been calibrated against ground truth. A flat confidence curve does not imply accurate decisions. It may indicate the agent has no signal that something is wrong. Ground truth calibration is future work.

## Project Structure

```
agent-ready-data/
├── README.md                          ← you are here
├── RESEARCH_NOTES.md                  ← living research log, backlog, decisions
├── requirements.txt
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

## Research Output

Each completed experiment produces:
- A **LinkedIn post** (~3,000 characters, practitioner audience)
- A **research article** (long-form, paper style, including limitations)

## Status

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

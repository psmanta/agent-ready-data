# The Agentic Data Contract — Research Notes

## Core Thesis
AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that are rarely made explicit. This research empirically tests what happens when each condition is violated — measuring the effect on decision quality, consistency, confidence, and aggregate accuracy.

The six pillars of the contract are: **Authoritative, Timely, Contextual, Comprehensive, Responsible, and Secure.**

---

## Methodology & Standing Assumptions

### Experimental Framework
- Each experiment uses a synthetic data pipeline, a business decision agent, and a structured evaluator
- The agent uses a "guided but not prescriptive" prompting approach — holistic assessment, no hard rules
- Temperature is fixed at 0.0 for reproducibility
- All experiments use `claude-haiku-4-5-20251001` as the base model unless otherwise noted
- Cost tracking is built in; experiments are self-funded and vendor-independent

### Domain Assumption
Current experiments use a **retail customer prioritization** task as the controlled testbed. This is a deliberate simplification. The conclusions are assumed (but not yet claimed) to generalize across domains. Multi-domain replication is scoped as a Phase 2 effort.

This assumption should be stated explicitly in all published work.

### Pipeline Stability
The base agent and evaluator pipeline is **locked** as of Experiment 1a. Changes will only be made if a specific experiment requires it, and will be versioned. The goal is comparability across experiments.

---

## Research Pillars & Experiment Backlog

### Pillar 1: Authoritative
*Authoritative data tells the truth. It is the single, trusted, non-contradicted source of record.*

| ID | Experiment | Status | Notes |
|----|-----------|--------|-------|
| 1a | Dedup — impact of duplicate records on agent decisions | ✅ Pipeline complete, ready for full run | Cluster map as skeleton key; 8 metrics incl. H4 field importance, H6 segment distortion |
| 1b | Dithering — data present but values untrustworthy | 🔲 Next | Fields present, values subtly wrong/corrupted; agent has no signal something is off |
| 1c | Data Quality vs Data Quantity | 🔲 Backlog | Does volume compensate for quality? "Clean 500 vs noisy 1000" |
| 1d | Conflicting Authorities | 🔲 Backlog | Two source systems, both "authoritative," disagree. MDM angle — what happens without a golden record? |
| 1e | Incompleteness vs Noise | 🔲 Backlog | Missing fields vs wrong fields — are these equivalent failure modes? Includes imputation strategies as a variable |
| 1f | Provenance Blindness | 🔲 Backlog | Agent cannot see data source confidence. Does knowing provenance change decisions? |

### Pillar 2: Timely
*Timely data reflects the current state of the world. Stale data was true once.*

| ID | Experiment | Status | Notes |
|----|-----------|--------|-------|
| 2a | Staleness — decisions made on data that was accurate at T but expired by T+N | 🔲 Backlog | Distinct from dithering: values are internally consistent, just outdated |

### Pillar 3: Contextual
*To be defined. Peter has a specific POV on this pillar.*

### Pillar 4: Comprehensive
*To be defined. Peter has a specific POV on this pillar.*

### Pillar 5: Responsible
*To be defined. Peter has a specific POV on this pillar.*

### Pillar 6: Secure
*To be defined. Peter has a specific POV on this pillar.*

---

## Phase 2 (Future)
- Multi-domain replication of Pillar 1 experiments (healthcare, finance, logistics)
- Cross-domain comparison of failure mode severity
- Revisit domain assumption with empirical evidence

---

## Publishing Plan

Each completed experiment produces two outputs:

### 1. LinkedIn Post
- **Format:** ~3,000 characters, LinkedIn audience
- **Tone:** Practitioner-facing, concrete findings, light on methodology
- **Structure:** Hook → what we tested → what we found → why it matters → call to action
- **Cadence:** One post per completed experiment

### 2. Research Article
- **Format:** Long-form, research paper style
- **Audience:** Technical practitioners, data engineers, AI/ML teams
- **Structure:** Abstract → hypothesis → methodology → results → implications → limitations → future work
- **Limitations section must include:** retail domain assumption, synthetic data caveat, single-model caveat
- **Cadence:** May batch multiple related experiments (e.g., 1a + 1b) into a single article

---

## Open Questions & Notes
- Should the composite quality score weights be revisited between experiments, or kept fixed for comparability?
- H4 (field importance) from 1a showed `last_purchase_days_ago`, `churn_risk_score`, `total_spend` as dominant factors — use this to inform field selection strategy for 1b dithering (data-driven, not cherry-picked)
- Consider whether the evaluator's composite score should be reported in LinkedIn posts or kept for the research article only
- "The Agentic Data Contract" is a working title — swap if a better name emerges

# The Agentic Data Contract — Research Notes

## Core Thesis
AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that are rarely made explicit. This research empirically tests what happens when each condition is violated — measuring the effect on decision quality, consistency, confidence, and aggregate accuracy.

The six pillars of the contract are: **Authoritative, Timely, Contextual, Comprehensive, Responsible, and Secure.**

---

## Scope & Definitions

### What "Agent" Means in This Research
The agent used in these experiments is a **decision-making engine** — it receives structured data, reasons over it holistically, and produces a structured judgment (priority level, confidence score, reasoning, key factors). It has:
- No tools
- No memory between calls
- No ability to take action in the world

It is not an agent in the fully autonomous sense. It represents the **cognitive core** of a fuller agentic system — the layer that would precede downstream actions like sending emails, triggering workflows, or escalating to humans.

This scope is intentional and valid. The decision layer is present in virtually every agentic pipeline. If it is compromised by bad data, everything downstream is compromised too. Readers should note this distinction when generalizing findings.

### Domain Assumption
All experiments use a **retail customer prioritization** task as the controlled testbed. This is a deliberate simplification chosen for:
- Clear ground truth intuition (high spend + high churn risk = high priority)
- Rich, realistic synthetic data generation
- Transferable failure modes

Conclusions are **assumed but not claimed** to generalize across domains. Multi-domain replication (healthcare, finance, logistics) is scoped as Phase 2.

This assumption must be stated explicitly in all published work.

---

## Known Limitations (flag in all writeups)

1. **Self-reported confidence is not calibrated.** Agent confidence is part of the LLM output, not an externally validated score. A flat confidence curve across duplication levels does not mean the agent is consistently accurate — it may mean the agent has no signal that anything is wrong. This is actually a finding in itself ("confidently wrong"), but ground truth calibration is future work.

2. **No ground truth for decision accuracy.** We measure consistency, distribution shift, and reasoning similarity — but not whether any individual decision is objectively correct. Adding a rule-based ground truth function to the data generator is a future improvement.

3. **Single model, single temperature.** All experiments use `claude-haiku-4-5-20251001` at temperature 0.0. Results may differ across models or temperature settings. Model comparison is out of scope for Phase 1.

4. **Synthetic data.** All customer records are generated. Real-world data distributions, correlations, and edge cases may produce different effects.

5. **Retail domain only (Phase 1).** See Domain Assumption above.

---

## Pipeline Stability
The base agent and evaluator pipeline is **locked** as of Experiment 1a. Changes will only be made if a specific experiment requires it, and will be versioned. The goal is comparability across experiments.

**Key methodological decisions locked in 1a:**
- Jaccard similarity (not LLM-based) for reasoning consistency — deterministic, reproducible, no AI variability
- Cluster map as skeleton key for H2 consistency (record_id → customer_id join)
- H6 segment distortion uses majority-vote per unique customer, not raw record counts
- H5 threshold defined as first level where consistency drops >2pp from previous level
- Composite score weights: consistency 30%, confidence 15%, distribution 10%, cost 15%, reasoning 15%, boundary 15%

---

## Research Pillars & Experiment Backlog

### Pillar 1: Authoritative
*Authoritative data tells the truth. It is the single, trusted, non-contradicted source of record.*

| ID | Experiment | Status | Notes |
|----|-----------|--------|-------|
| 1a | Dedup — impact of duplicate records on agent decisions | ✅ Complete | 1,000 base customers, 8 duplication levels (0–100%), 15,110 total records processed |
| 1b | Dithering — data present but values untrustworthy | 🔲 Next | Fields present, values subtly wrong/corrupted. Use 1a H4 field importance to select dither targets (data-driven, not cherry-picked) |
| 1c | Data Quality vs Data Quantity | 🔲 Backlog | Does volume compensate for quality? "Clean 500 vs noisy 1000" |
| 1d | Conflicting Authorities | 🔲 Backlog | Two source systems, both "authoritative," disagree. MDM angle — what happens without a golden record? |
| 1e | Incompleteness vs Noise | 🔲 Backlog | Missing fields vs wrong fields — are these equivalent failure modes? Includes imputation strategies as a variable |
| 1f | Provenance Blindness | 🔲 Backlog | Agent cannot see data source confidence. Does knowing provenance change decisions? |

### Pillar 2: Timely
*Timely data reflects the current state of the world. Stale data was true once.*

| ID | Experiment | Status | Notes |
|----|-----------|--------|-------|
| 2a | Staleness — decisions made on data accurate at T but expired by T+N | 🔲 Backlog | Distinct from dithering: values are internally consistent, just outdated |

### Pillar 3: Contextual
*Peter has a specific POV on this pillar — to be defined.*

### Pillar 4: Comprehensive
*Peter has a specific POV on this pillar — to be defined.*

### Pillar 5: Responsible
*Peter has a specific POV on this pillar — to be defined.*

### Pillar 6: Secure
*Peter has a specific POV on this pillar — to be defined.*

---

## Phase 2 (Future)
- Multi-domain replication of Pillar 1 experiments (healthcare, finance, logistics)
- Cross-domain comparison of failure mode severity
- Revisit domain assumption with empirical evidence

---

## Publishing Plan

### 1. LinkedIn Post
- **Format:** ~3,000 characters, LinkedIn audience
- **Tone:** Practitioner-facing, concrete findings, light on methodology
- **Structure:** Hook → what we tested → what we found → why it matters → call to action
- **Cadence:** One post per completed experiment

### 2. Research Article
- **Format:** Long-form, research paper style
- **Audience:** Technical practitioners, data engineers, AI/ML teams
- **Structure:** Abstract → hypothesis → methodology → results → implications → limitations → future work
- **Limitations section must include:** retail domain assumption, synthetic data caveat, single-model caveat, self-reported confidence caveat, no ground truth for decision accuracy
- **Cadence:** May batch multiple related experiments (e.g., 1a + 1b) into a single article

---

## Decisions Log

| Date | Decision | Rationale |
|------|---------|-----------|
| Mar 2026 | Use Jaccard similarity for reasoning quality (Metric 5) | Deterministic and reproducible; LLM-based similarity would introduce AI variability that contaminates the measurement |
| Mar 2026 | H6 segment distortion uses majority-vote per unique customer | Raw record counts inflate segment share for heavily duplicated segments, contaminating the shift measurement |
| Mar 2026 | H5 threshold set at >2pp consistency drop | 2pp represents a detectable, non-trivial degradation actionable for a data quality team |
| Mar 2026 | H4 ranking stability anchored explicitly to 0pct baseline | Dict ordering is not guaranteed; explicit anchor ensures we always compare against clean data |
| Mar 2026 | Lock base pipeline after 1a | Comparability across experiments requires consistent methodology |
| Mar 2026 | 1,000 base customers for 1a | Sufficient statistical power at low duplication levels; 500 too thin for H6 segment analysis |
| Mar 2026 | Run standard API for 1a, Batch API from 1b onward | Clean baseline; 50% cost saving for all subsequent experiments |
| Mar 2026 | Retail domain as testbed | Tractable, realistic, transferable failure modes; domain limitation documented |
| Mar 2026 | Temperature 0.0 | Deterministic outputs for reproducibility |

---

## Open Questions
- Should composite score weights be revisited between experiments or kept fixed for comparability? (Current: fixed)
- When should we add ground truth calibration to the data generator?
- 1b dither field selection: use 1a H4 top fields (`last_purchase_days_ago`, `churn_risk_score`, `total_spend`) as primary targets

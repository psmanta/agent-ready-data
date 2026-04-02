# The Agentic Data Contract — Research Notes

## Core Thesis
AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that are rarely made explicit. This research empirically tests what happens when each condition is violated — measuring the effect on decision quality, consistency, confidence, and aggregate accuracy.

The six pillars of the contract are: **Authoritative, Timely, Contextual, Comprehensive, Responsible, and Secure.**

---/Users/pmanta/Downloads/RESEARCH_NOTES.md

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

## Experiment 1a: Key Findings Summary

*Full detail in `experiments/01_authoritative/1a_dedup/1a_RESULTS.md`*

1. **The Cliff Edge (H5):** No safe duplication threshold exists. Consistency drops immediately from 100% to 85% at 10% duplication and flatlines. "A little duplication is probably fine" is empirically false.

2. **Confidently Wrong:** Agent confidence is essentially immovable (99.6% stability, 81.17%–81.49% range). Standard monitoring using confidence as a data quality proxy will show nothing wrong even as 1 in 7 customers receives conflicting treatment. The failure is silent.

3. **The Agentic Blind Spot:** A human reviewer would eventually notice "I've seen this customer before." A stateless agent never will. The architectural properties that make agents attractive — consistency, tirelessness — simultaneously eliminate an incidental deduplication check that human workflows provide for free. Agentic pipelines require compensating upstream controls.

4. **Volume Inflation (H3):** The HIGH/MEDIUM/LOW percentage distribution appears stable — but raw case volume inflates proportionally with duplication. At 100%, the agent produced 1,320 HIGH priority decisions from ~430 unique HIGH priority customers. A practitioner who skips deduplication assuming "the agent will deal with it" will see a stable distribution and conclude nothing is wrong — while resource planning decisions based on raw output project 3x actual caseload.

5. **Wasted Spend:** 43.2% of total API spend ($14.07 of $27.54) was wasted on duplicate records. Cost per unique customer decision is 2.87x the clean baseline at 100% duplication.

6. **Boundary Customer Vulnerability (H2 + H4):** Field variation only flips decisions for customers near a decision boundary. Strong-signal customers (e.g. `at_risk` segment) are immune. Boundary customers are disproportionately affected and are often the most consequential to classify correctly.

7. **Consistent Field Reliance (H4):** Top-5 field ranking is 100% stable across all duplication levels: `last_purchase_days_ago`, `churn_risk_score`, `nps_score`, `lifetime_value_estimate`, `support_tickets_open`. Provides data-driven, non-cherry-picked basis for 1b dithering target selection.


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
| Mar 2026 | Exclude H4 and H6 from composite score | Both are diagnostic metrics, not quality metrics — including them would conflate measurement with finding |
| Mar 2026 | Use plain-English chart titles with finding subtitles | Charts must be standalone readable for LinkedIn audience without accompanying writeup |
| Mar 2026 | Document illustrative examples in RESULTS.md with full record IDs | Supports research article narrative and provides full traceability back to raw data |
| Mar 2026 | Uniform duplication in 1a (no --segment-bias flag) | Establishes clean baseline; segment-biased rerun is optional future work before 1b |

---

## Open Questions
- Should composite score weights be revisited between experiments or kept fixed for comparability? (Current: fixed)
- When should we add ground truth calibration to the data generator?
- **1b dither field selection:** Use 1a H4 top fields (`last_purchase_days_ago`, `churn_risk_score`, `total_spend`) as primary targets; dither identity fields (`name`, `email`) as control condition
- **Boundary customer segment:** Should 1b explicitly define and track a "boundary zone" customer flag — customers near the HIGH/MEDIUM decision threshold? 1a evidence suggests these are the most vulnerable and most consequential.
- **Volume inflation metric:** Add raw count vs unique customer count comparison to the evaluator before 1b — current metrics only track percentage distribution, missing the inflation failure mode entirely
- **Optional 1a variant:** Rerun 1a with `--segment-bias high_value` before 1b to get a stronger H6 finding. Low cost since pipeline is already built.
- **Agentic blind spot as design principle:** How do we formalize the compensating upstream controls finding into a generalizable recommendation? This should appear explicitly in the research article implications section.

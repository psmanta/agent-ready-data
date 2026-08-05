# The Agentic Data Contract — Research Notes

## Core Thesis
AI agents operate under an implicit contract with their data. For an agent to make reliable decisions, that data must satisfy a set of conditions that are rarely made explicit. The reflexive answer to agentic failure modes is often "data quality". 

Data quality is necessary, but not sufficient. This research empirically tests what happens when the contract is violated, measuring the effect on **Decision Quality**, the trustworthiness of agent decisions when high stakes are on the line.

The six pillars of the contract are: **Authoritative, Comprehensive, Contextual, Timely, Responsible, and Secure.**

---

## Scope & Definitions

### What "Agent" Means in This Research
The agent used in these experiments is a **decision-making engine**. It receives data, reasons over it, and produces a structured judgment (priority level, confidence score, reasoning, key factors). It has:

- No tools
- No memory between calls
- No ability to take action in the world

It is not an agent in the fully autonomous sense. It represents only the cognitive core of a fuller agentic system, the layer that would precede downstream actions like sending emails, triggering workflows, or escalating to humans.

This scope is intentional and valid. The decision layer is present in virtually every agentic pipeline. If it is compromised by bad data, everything downstream is compromised too. Readers should note this distinction when generalizing findings.

### Domain Assumption
All experiments use a **retail customer prioritization** task as the controlled testbed. This is a deliberate simplification chosen for:

- Clear decision outcomes (HIGH, MEDIUM, or LOW priority) that are intuitive for an agent to reason about
- Rich, realistic synthetic data generation
- Transferable failure modes

Conclusions are assumed but not claimed to generalize across domains. Multi-domain replication (healthcare, finance, logistics, etc.) is scoped for a later phase.

This assumption must be stated explicitly in all published work.

---

## Known Limitations

1. **Self-reported confidence is not calibrated.** Agent confidence is part of the LLM output, not an externally validated score. A flat confidence curve across duplication levels does not mean the agent is consistently accurate, it may mean the agent has no signal that anything is wrong. This may actually be a finding in itself (the "confidently wrong" hypothesis).

2. **No ground truth for decision accuracy.** We measure consistency, distribution shift, and reasoning similarity, but not whether any individual decision is objectively correct. Ground truth design will be implemented as required for specific experiments.

3. **Single model, single temperature.** All experiments use `claude-haiku-4-5-20251001` at temperature 0.0. Results may differ across models or temperature settings. Model comparison is out of scope for Phase 1.

4. **Synthetic data.** All customer records are generated. Real-world data distributions, correlations, and edge cases may produce different effects.

5. **Retail domain only.** See Domain Assumption above.

6. **Distribution assumptions.** Where experiments require generating data with specific conditions (duplication, corruption, staleness, etc.), those conditions are distributed uniformly across the dataset by default. Real world data rarely distributes problems uniformly and certain customer types, systems, or time periods are often disproportionately affected. Observed effects may therefore be conservative estimates of real world impact.

7. **Prompt design.** System prompts are designed to be guided but not prescriptive. However, prompt design choices may be identified post-run as potentially influencing results. Any such findings will be documented transparently in the relevant experiment's results file.

8. **Experimental isolation.** Each experiment is designed to test a specific failure mode in isolation. In practice, some overlap between failure modes may be present in the experimental data. Evaluation metrics are always designed to focus on the target failure mode, and any known overlap is documented as a limitation in the relevant experiment.

---

## Pipeline Stability

The base agent, LLM factory, and evaluator framework are locked. Changes will only be made if a specific experiment requires it, and will be versioned. The goal is comparability across experiments.

Key methodological principles:

- **Deterministic evaluation metrics** Avoiding AI-based scoring in the measurement layer eliminates variability that would contaminate findings. (where possible)
- **The skeleton key architecture** Each experiment maintains a reference dataset that links what the agent sees back to known truth, enabling analysis that the agent itself cannot perform.
- **Composite Decision Quality score** Each experiment produces a weighted composite score designed to quantify decision quality. Metrics, weights, and scoring approaches may evolve across experiments to reflect what each pillar is actually measuring. The goal is a comparable, honest signal not a rigid formula.

This is a living research series. Methodologies will be documented, versioned, and explained when they evolve. The goal is transparency, not rigidity.

### Data Generator Architecture

The original 1a data generator (`experiments/01_authoritative/1a_dedup/generate_customer_data.py`) 
was not held to ground truth standards. It was designed to test duplication effects, not to 
produce internally consistent baseline data. After 1a completed, I reazlied there was a better approach so
the core customer generation logic was refactored into a shared base generator (`shared/data_generation/base_customer_generator.py`).

The shared base generator produces what we define as data ground truth for Phase 1 experiments. Internally consistent, 
and aligned with the well established DAMA six dimensions of data quality: Accuracy, Completeness, Consistency, Timeliness, Validity, and Uniqueness.

**Distribution comparison (1a generator vs shared base generator, seed=42, n=1000):**

| Metric | 1a Generator | New Generator | Notes |
|--------|-------------|---------------|-------|
| Segment distribution | Identical | Identical | No change |
| Total spend (mean) | $7,468 | $7,696 | Within 3% — acceptable |
| Churn risk (mean) | 0.38 | 0.40 | Slight upward shift — acceptable |
| NPS (mean) | 6.34 | 6.04 | Slight downward shift — acceptable |
| LTV (mean) | $12,101 | $22,225 | Intentionally higher. The new generator uses segment-specific multipliers (high_value: 2.5–4.0x vs flat 1.2–2.0x). More realistic. |
| last_login_days_ago (mean) | 80 days | 184 days | Intentionally tighter. The new generator bounds login within 30 days of last purchase. Consistency fix. |
| Consistency violations | 0 | 0 | Both pass |

NOTE: The 1a data generator fails the Timeliness dimension. It used datetime.now() as its date anchor, so generated dates are relative to the run date (March 2026) 
rather than a fixed snapshot. This does not affect 1a findings since duplication effects are date-independent, but it is another reason the shared base generator was built to a stricter standard.

1a results are unaffected. The duplication mechanism in 1a did not depend on LTV multipliers, login/purchase consistency, or date anchoring. 
Specifically, the higher LTV values in the new generator do not affect duplication consistency findings. the tighter login/purchase gap does not affect cluster-level decision analysis.
The datetime.now() date anchor in 1a means dates are relative to the March 2026 run date rather than a fixed snapshot which it irrelevant for duplication effects but correctly flagged 
by the DAMA Timeliness check. These differences are improvements to the baseline data quality for 1b onward, not corrections to 1a findings.


---

## Experiment Findings

Key findings for each completed experiment are documented in the experiment's own RESULTS.md file. This research notes file tracks methodology, decisions, and open questions, not findings summaries. See the research pillars and experiment backlog below for status and links.

---

## Research Pillars & Experiment Backlog

### Pillar 1: Authoritative
*Authoritative data tells the truth. It is the single, trusted, non-contradicted source of record.*

| ID | Experiment | Status | Notes |
|----|-----------|--------|-------|
| 1a | Dedup | ✅ Complete | How do duplicate records affect agent decision quality? |
| 1b | Dithering | 🔲 Next | What happens when field values are present but don't reflect the truth? |
| 1c | Quality vs Quantity | 🔲 Backlog | Does a larger dataset with noise outperform a smaller clean dataset? |
| 1d | Conflicting Authorities | 🔲 Backlog | What happens when two authoritative source systems disagree on the same customer? |
| 1e | Incompleteness vs Noise | 🔲 Backlog | Are missing fields and untruthful fields equivalent failure modes for decision quality? |
| 1f | Provenance Blindness | 🔲 Backlog | Does knowing where the data came from change how well an agent uses it? |
| 1g | Data Lifecycle Stage | 🔲 Backlog | Can an agent identify which stage of the data lifecycle best serves a given business requirement? |

### Pillar 2: Comprehensive
*Data tells the whole truth.*

*Experiments for this pillar are in design. Details to follow.*

### Pillar 3: Contextual
*Data tells the truth the business defines.*

*Experiments for this pillar are in design. Details to follow.*

### Pillar 4: Timely
*Data tells the truth for the world as it is now.*

*Experiments for this pillar are in design. Details to follow.*

### Pillar 5: Responsible
*Data tells the truth without creating outcomes we can't roll back.*

*Experiments for this pillar are in design. Details to follow.*

### Pillar 6: Secure
*Data tells the truth, privately and without unintended inference.*

*Experiments for this pillar are in design. Details to follow.*

---

## Phase 2 (Future)

Phase 2 expands the research in two directions:

**Cross-pillar synthesis (Phase 1.5):** Before expanding to new domains, the interactions between pillars will be explored. Phase 1 tests each data condition in isolation, a controlled but artificial scenario. Real enterprise data fails across multiple dimensions simultaneously. Phase 1.5 will design experiments that combine pillar failure modes and measure whether effects are additive, compounding, or whether one failure mode dominates. This is where the Agentic Data Contract framework moves from diagnostic to prescriptive, helping practitioners prioritize which data conditions to address first when resources are constrained.

**Multi-domain replication:** Apply Pillar 1 experiments across healthcare, finance, and logistics to test whether failure modes and severity generalize beyond retail. The domain assumption documented in Phase 1 will be revisited with empirical evidence.

**Agentic memory and knowledge structures:** Phase 1 explores how business context and knowledge structures provided as static input affects decision quality. Phase 2 goes further, examining how agent memory, knowledge graphs, and context persistence that evolve across sessions introduce new failure modes. When an agent can remember, retrieve, and reason across interactions, the contract with its data becomes significantly more complex, and the consequences of bad data compound rather than resetting with each call.

---

## Publishing Plan

### LinkedIn
- **Format:** LinkedIn character limit applies, clarity and impact take priority within the constraint
- **Tone:** Practitioner-facing, concrete findings, light on methodology
- **Assets:** Key charts selected per experiment to support the narrative
- **Structure:** Hook → what was tested → what was found → why it matters → call to action
- **Cadence:** One post per completed experiment, aligned with but distinct from the pillar thought leadership series

### Research Article
- **Format:** Long-form, research paper style
- **Audience:** Technical practitioners, data engineers, AI/ML teams
- **Structure:** Abstract → hypothesis → methodology → results → implications → limitations → future work
- **Limitations:** Reference the Known Limitations section in this file. Do not maintain a separate list
- **Targets:** arXiv for citable reference, Towards Data Science for practitioner reach
- **Cadence:** May batch multiple related experiments (e.g., 1a + 1b) into a single article

---

## Decisions Log

| Milestone | Decision | Rationale |
|-----------|---------|-----------|
| 1a design | Use Jaccard similarity for reasoning quality | Deterministic and reproducible, LLM-based similarity would introduce AI variability that contaminates the measurement |
| 1a design | Retail domain as testbed | Tractable, realistic, transferable failure modes. Domain limitation documented explicitly in all published work |
| 1a design | Temperature 0.0 | Deterministic outputs for reproducibility across all experiments |
| 1a design | Run standard API for 1a, Batch API from 1b onward | Clean baseline for 1a; 50% cost saving for all subsequent experiments with no impact on results |
| 1a complete | Lock base agent, LLM factory, and evaluator framework. | Comparability across experiments requires consistent methodology. Changes will be versioned and documented. |
| 1a complete | Exclude diagnostic metrics from composite score | Diagnostic metrics reveal where and why quality degrades. Including them in the score would conflate measurement with finding |
| 1a complete | Plain-English chart titles with narrative subtitles | Charts must be standalone readable for LinkedIn audience without accompanying writeup |
| 1a complete | Document illustrative examples in RESULTS.md with full record IDs | Provides full traceability back to raw data and supports research article narrative |
| 1a complete | Refactor data generation into shared base generator | Extracted core customer generation into `shared/data_generation/base_customer_generator.py` after 1a completed. Enforces 7 consistency rules not present in 1a generator. 1a generator untouched and hermetically sealed for reproducibility. |
| 1a review | Composite score visualization pending redesign | Current chart has rendering issues, redesign planned after 1b provides a comparative baseline |
| 1a review | Prompt design limitations documented post-run | Transparency requires acknowledging methodology observations identified after data collection. A rerun is planned where warranted. |
| 1a review | Defer optional 1a segment-bias rerun | 1a distributed duplicates uniformly across all customer segments. The --segment-bias flag would bias duplication toward a specific segment (e.g. 4x more duplicates for high_value customers) to produce a stronger segment distortion finding. Deferred, current uniform distribution findings are sufficient for conclusions and conservative estimates of real-world impact. |

---

## Open Questions

- **Ground truth calibration:** Two layers are now defined. 
   - Data Ground Truth: The shared base generator produces internally consistent records meeting all six data quality dimensions. See Data Generator Architecture above. 
   - Decision ground truth: a 5-run majority vote baseline at temperature 0.0 will be implemented in 1b. See `1b_DESIGN.md`. Implementation TBD.
- **Boundary customer tracking:** Should 1b explicitly define and track a "boundary zone" customer flag, customers near the HIGH/MEDIUM and MEDIUM/LOW decision thresholds? 1a evidence suggests these are the most vulnerable and most consequential.
- **Volume inflation metric:** Add raw decision count vs unique customer count comparison to the evaluator before 1b. Current metrics track percentage distribution only, missing the volume inflation failure mode.
- **Agentic blind spot as design principle:** How should I formalize the compensating upstream controls finding into a generalizable recommendation? This should appear explicitly in the research article implications section.
- **Prompt-controlled rerun of 1a:** Rerun with illustrative examples removed from system prompt to confirm findings hold. Low priority, findings not expected to change materially.
- **Boundary population size as a stochasticity finding:** Track the actual stable / lightly_boundary / deeply_boundary population split once 1b's primary baseline runs. The size of the boundary population is itself a finding about API-level stochasticity at temperature 0.0, independent of anything dithering specific. Temperature is held fixed at 0.0 throughout Phase 1 deliberately, since introducing it as a second variable alongside dithering would make it impossible to attribute observed decision instability to data quality versus model sampling behavior. A dedicated temperature experiment (see backlog) should be run in isolation, not blended into Phase 1.

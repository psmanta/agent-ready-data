# Experiment 1a Results: Deduplication Impact on Agentic Decision Making

**Status:** ✅ Complete  
**Date:** March 2026  
**Model:** `claude-haiku-4-5-20251001`  
**Temperature:** 0.0  
**Base customers:** 1,000  
**Duplication levels tested:** 0%, 10%, 20%, 30%, 40%, 50%, 75%, 100%  
**Total records processed:** 15,110  
**Total API cost:** $27.54  

---

## Hypothesis Results

| ID | Hypothesis | Result | Finding |
|----|-----------|--------|---------|
| H1 | Record-level decision integrity: Each record is processed independently | ✅ Confirmed by design | Agent processes each record independently with no shared state between API calls. Individual decision integrity is guaranteed architecturally, not empirically. Notable: this same property means the agent will never notice it has seen the same customer before. See Finding 3. |
| H2 | Cluster-level consistency: Duplicate records in a cluster should receive the same decision. | ⚠️ Partially confirmed | 85–87% consistency across all duplication levels. 1 in 7 customers received conflicting priority decisions depending on which duplicate record the agent saw. Inconsistency is driven by field variation within clusters, not by duplication volume. |
| H3 | Aggregate distribution drift: Duplicate records should not distort macro level decisions based on the full dataset | ⚠️ Nuanced | The HIGH/MEDIUM/LOW percentage distribution is stable (43% to 45.8% HIGH across full range). However, raw case volume inflates proportionally with duplication. At 100%, the agent reported 1,320 HIGH priority cases when the true unique count was ~430. A resource planning decision based on raw output would significantly over-project caseload. |
| H4 | Field importance shift: Certain fields carry more decision-making weight than others, and field importance rankings will remain stable regardless of duplication level. | ✅ Confirmed | Top-5 field ranking is 100% stable across all duplication levels: `last_purchase_days_ago`, `churn_risk_score`, `nps_score`, `lifetime_value_estimate`, `support_tickets_open`. Provides data-driven, non-cherry-picked basis for 1b dithering target selection. |
| H5 | Duplication rate threshold: There is a duplication 'cliff' where decisions will degrade at a specific record duplication level | ✅ Confirmed (unexpected) | There is no safe threshold. Consistency drops immediately from 100% to 85% at the lowest tested level (10%) and flatlines across the full range. The common assumption that "a little duplication is probably fine" is empirically false. |
| H6 | Segment distortion: Duplication does not affect all customer segments equally. Some segments may be disproportionately impacted by duplicate records | ✅ Confirmed | `at_risk` customers show zero distortion (overdetermined signal). `medium_value` customers are most sensitive (avg 2.8% shift). Distortion follows a predictable pattern tied to signal strength, not segment size. |

---

## Evaluation Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| 1. Decision Consistency (H2): How consistent are decisions within a cluster? | 87.7/100 | 456 total inconsistencies across all levels |
| 2. Confidence Stability: How confident is the agent of its own decision? | 99.6/100 | Range: 0.8117–0.8149 — essentially flat |
| 3. Distribution Shift (H3): How much is a dataset level conclusion affected by duplicate records? | 68.1/100 | Max shift 3.19pp — stable in percentage terms but misleading (see H3 volume inflation) |
| 4. Cost Efficiency: How much processing cost went into unique vs redundant decisions? | 56.8/100 | $14.07 wasted of $27.54 total — 43.2% waste |
| 5. Reasoning Consistency (Jaccard): How similar is the agent's reasoning language when explaining decisions for duplicate records of the same customer? | 48.9/100 | Avg 0.489 word-overlap across duplicate pairs |
| 6. Human-Agent Boundary: What proportion of decisions fall below the confidence threshold where human review is recommended? | 89.6/100 | ~10.4% of decisions require human review regardless of duplication level |
| 7. Field Importance (H4): Are certain customer segments more susceptible to decision distortion from duplicate records? | Diagnostic | 100% ranking stability — top fields: `last_purchase_days_ago`, `churn_risk_score` |
| 8. Segment Distortion (H6): How does the duplication level vary across different customer segments? | Diagnostic | `at_risk` immune, `medium_value` most sensitive |
| **Composite Score** | **77.4/100 — FAIR** | Weighted across metrics 1–6 |

---

## Key Findings

### Finding 1: The Cliff Edge (H5)
There is no safe duplication threshold. Decision consistency drops immediately from 100% to 85% at 10% duplication and flatlines for the remainder of the range. Practitioners hoping to identify a safe zone for duplication will not find one. The implication is binary: deduplicate before agent processing, or accept ~15% cluster inconsistency as a permanent baseline tax on every decision the agent makes.

### Finding 2: Confidently Wrong (H2 + Confidence)
Agent confidence is essentially immovable with 99.6% stability, ranging only from 81.17% to 81.49% across all duplication levels. This is the most operationally dangerous finding. Standard production monitoring using confidence scores as a proxy for data quality will show nothing wrong, even as 1 in 7 customers receives conflicting treatment. The agent has no internal signal that data quality is degrading. Because this failure mode is silent and scalable, the next experiment will focus on the effect of data dithering on agentic decision making.

### Finding 3: The Agentic Blind Spot (H1 + H3)
A human analyst reviewing duplicate records would eventually notice they had seen the same customer before. An agent operating statelessly will not, ever. This architectural property that makes agents attractive (consistency, tirelessness, no cognitive bias) simultaneously eliminates an incidental deduplication check that humans provide naturally. Agentic systems therefore require compensating upstream controls that human workflows provide for free. This is a systemic architectural risk, not merely a data quality problem.

### Finding 4: Volume Inflation (H3)
The aggregate HIGH/MEDIUM/LOW percentage distribution appears stable, but this is misleading. At 100% duplication, the agent produced 1,320 HIGH priority decisions from 2,882 records. The true count of unique HIGH priority customers was approximately 430. A resource planning decision based on raw agent output would project 3x the actual caseload. This failure mode is invisible in percentage-based metrics and requires comparing raw counts against unique customer counts to detect. This isn't a failure of the agent's decision logic, the agent correctly prioritized each record it saw. It's a consequence of the practitioner assumption: 'the agent will deal with it.' When the decision distribution looks stable in percentage terms, there is no obvious signal that anything is wrong, making it easy to conclude that deduplication is unnecessary overhead. But raw volume inflation is happening silently. A hiring manager acting on agent output would staff for 1,320 high-priority cases when the true count is 430, not because the agent made bad decisions, but because nobody asked whether the input data was clean before the agent ran.

### Finding 5: Wasted Spend (Cost)
43.2% of total API spend ($14.07 of $27.54), was wasted processing duplicate records that added no decision value. At 100% duplication, the cost per unique customer decision is 2.87x the clean baseline. This is a direct, measurable business case for upstream deduplication. Every dollar spent on deduplication upstream prevents $1.87 in wasted inference spend at the agent layer.

### Finding 6: Boundary Customer Vulnerability (H2 + H4)
Field variation in duplicate records only flips decisions for customers whose profiles sit near a decision boundary, not clearly HIGH, not clearly MEDIUM. Customers with strong, unambiguous signal across multiple fields (e.g. the `at_risk` segment) are immune. Boundary customers are disproportionately affected and are often the most important to classify correctly as they represent the edge cases where incorrect automation has the greatest business consequence and where human judgment is most valuable.

### Finding 7: Consistent and Predictable Field Reliance (H4)
The agent's top-5 field ranking is identical across all 8 duplication levels. `last_purchase_days_ago` and `churn_risk_score` dominate consistently. This has two implications: (1) the agent's decision logic is auditable and stable: a practitioner can understand and predict what the agent is optimizing for, and (2) it provides a data-driven, non-cherry-picked foundation for designing the 1b dithering experiment.

---

## Agent Run Summary

| Level | Records | Cost | HIGH | MEDIUM | LOW | Confidence |
|-------|---------|------|------|--------|-----|------------|
| 0% | 1,000 | $2.32 | 43.0% | 54.4% | 2.6% | 0.8134 |
| 10% | 1,183 | $2.74 | 42.0% | 55.1% | 2.9% | 0.8117 |
| 20% | 1,402 | $3.24 | 44.2% | 53.4% | 2.4% | 0.8139 |
| 30% | 1,587 | $3.67 | 43.3% | 54.1% | 2.6% | 0.8130 |
| 40% | 1,772 | $4.09 | 44.6% | 52.4% | 3.0% | 0.8149 |
| 50% | 1,894 | $4.38 | 44.6% | 53.2% | 2.2% | 0.8135 |
| 75% | 2,390 | $5.52 | 45.2% | 52.3% | 2.5% | 0.8142 |
| 100% | 2,882 | $6.65 | 45.8% | 51.2% | 3.0% | 0.8136 |
| **Total** | **15,110** | **$27.54** | | | | |

---

## Limitations

1. **Self-reported confidence is not calibrated.** Agent confidence is part of the LLM output, not an externally validated score. The flat confidence curve is itself a finding but does not tell us whether individual decisions are objectively correct.
2. **No ground truth for decision accuracy.** I measure consistency, distribution, and reasoning similarity, not whether any individual decision is correct. A rule-based ground truth function is future work.
3. **Single model, single temperature.** `claude-haiku-4-5-20251001` at 0.0. Results may differ across models or temperature settings.
4. **Synthetic data.** All records were generated specifically for this experiment. Real world distributions and correlations may produce different effects.
5. **Uniform duplication distribution.** Duplicates were distributed evenly across segments. In real enterprise data, high-value customers interacting across multiple channels would likely be disproportionately duplicated. The data generator supports segment-biased duplication via --segment-bias (e.g. --segment-bias high_value weights duplication 4x toward a specific segment), but this flag was not used in this experiment. H6 effects observed here are therefore conservative estimates of real-world segment distortion.
6. **Retail domain only.** I chose to test using a customer segmentation, retail domain scenario. Conclusions are assumed but not yet proven to generalize across other domains.

---

## Illustrative Examples

These examples are drawn from the `inconsistent_examples` field in `evaluation_metrics.json`. They are included here to support the research article narrative and to make the quantitative findings concrete. Each example is identified by customer ID and record IDs for full traceability back to the raw data.

---

### Example 1: Xavier / Xevier — Boundary Customer, Minimal Variation (CUST_000956)

**The finding it illustrates:** Boundary customer vulnerability (Finding 6) and the surprising impact of minimal field variation.

**What happened:** Two duplicate records for the same customer. One record has the customer name as "Xavier", the other as "Xevier", a single character typo introduced by the data generator as minimal variation. The `last_purchase_days_ago` field also drifted slightly: 194 days on one record, 198 days on the other.

| Record | Decision | Confidence | Key signal |
|--------|---------|-----------|-----------|
| REC_9EC37D86D963 | HIGH_PRIORITY | 0.82 | 194 days since purchase — "significant inactivity requiring immediate intervention" |
| REC_18E3F9870C79 | MEDIUM_PRIORITY | 0.78 | 198 days since purchase — "no immediate crisis" |

**Why it matters:** A 4-day difference in `last_purchase_days_ago`, well within normal data entry variance, pushed this customer across a decision boundary. The agent's reasoning is coherent for each record individually. The contradiction only becomes visible when the two records are compared side by side, which the agent never does. This customer either gets immediate outreach or standard treatment depending on which record the system happens to process.

---

### Example 2: Mario — Cluster Split, 6 Records (CUST_000520)

**The finding it illustrates:** Cluster-level inconsistency at scale (H2) and the 50/50 split problem.

**What happened:** Six duplicate records for the same customer. Three records were classified HIGH_PRIORITY (confidence 0.82), three were classified MEDIUM_PRIORITY (confidence 0.78). The reasoning for each individual record is coherent — but the cluster is evenly split.

| Records | Decision | Confidence | Reasoning emphasis |
|---------|---------|-----------|-------------------|
| REC_5C8B64AFC930, REC_2A82266B0986, REC_2FA77D9EE3BD | HIGH_PRIORITY | 0.82 | 362–391 days since last purchase, "immediate intervention needed" |
| REC_CD75993D7BED, REC_088D66E55935, REC_B83F43D0C850 | MEDIUM_PRIORITY | 0.78 | Same inactivity, but framed as "standard attention to re-engage" |

**Why it matters:** This is not a case of one clearly correct decision and one error. The agent is genuinely uncertain about this customer, the 50/50 split reflects a profile that sits squarely on a decision boundary. In a deduplication scenario, a majority-vote approach would produce a tie. The customer's actual treatment depends entirely on which record the downstream system consumes first. This is exactly the class of decision that should be escalated to human review, but the agent's confidence scores (0.82 and 0.78) would not flag it under any reasonable threshold.

---

### Example 3: CUST_000248 — Tier-Critical Field Variation Flips Priority Category

**The finding it illustrates:** H4 field importance — variation in tier-critical fields has outsized impact on decisions.

**What happened:** Two duplicate records. The `total_spend` field varies significantly between them: $550.62 on one record, $358.95 on the other. This single field difference flips the decision from MEDIUM_PRIORITY to LOW_PRIORITY.

| Record | Decision | Confidence | `total_spend` |
|--------|---------|-----------|--------------|
| REC_2C739D61BD5E | MEDIUM_PRIORITY | 0.72 | $550.62 — "moderate lifetime value" |
| REC_5340FA47E827 | LOW_PRIORITY | 0.78 | $358.95 — "low-value with limited growth potential" |

**Why it matters:** `total_spend` is one of the top-5 fields the agent consistently cites as a key decision factor (H4). When this field varies across duplicate records, even within a plausible range, it directly flips the priority category. This example provides concrete justification for using `total_spend` as a primary dither target in Experiment 1b. It also illustrates that a single tier-critical field change can override all other signals.

---

## Implications for 1b Design

1. **Dither target fields:** Use H4 findings: `last_purchase_days_ago`, `churn_risk_score`, `total_spend`, `lifetime_value_estimate` as primary dither targets. Data-driven, not cherry picked.
2. **Control condition:** Dither identity fields (`name`, `email`) as a control, H4 predicts minimal effect.
3. **Boundary customer tracking:** Explicitly identify and track boundary customers. Customers near the HIGH/MEDIUM decision boundary are most vulnerable and most important.
4. **Batch API:** Implement before 1b. 50% cost saving with no impact on results.

---

## Next Steps

- [x] Run evaluator and populate this document
- [ ] Update `RESEARCH_NOTES.md` with key findings
- [ ] Draft LinkedIn post
- [ ] Draft research article
- [ ] Implement volume inflation metric in evaluator (pending)
- [ ] Add H4, H6, and volume inflation charts to evaluator (pending)
- [ ] Design 1b dithering experiment

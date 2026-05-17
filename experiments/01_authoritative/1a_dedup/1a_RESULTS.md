# Experiment 1a Results: Deduplication Impact on Agentic Decision Making

**Status:** ✅ Complete  
**Run Date:** March 2026  
**Model:** `claude-haiku-4-5-20251001`  
**Temperature:** 0.0  
**Base customers:** 1,000  
**Duplication levels tested:** 0%, 10%, 20%, 30%, 40%, 50%, 75%, 100%  
**Total records processed:** 15,110  
**Total API cost:** $32.70

---

## Hypothesis Results

| ID | Hypothesis | Result | Finding |
|----|-----------|--------|---------|
| H1 | Record-level decision integrity: Each record is processed independently | ✅ Confirmed by design | Agent processes each record independently with no memory of previous records. Individual decision integrity is guaranteed architecturally, not empirically. Notably, this same property means the agent will never notice it has seen the same customer before. See Finding 3. |
| H2 | Cluster-level consistency: When multiple records exist for the same customer, all should receive the same prioroty decision. | ⚠️ Partially confirmed | 85–87% consistency across all duplication levels. 1 in 7 customers received conflicting priority decisions depending on which duplicate record the agent saw. Duplicate records in this experiment contain natural variation, as they would in real enterprise systems, reflecting the fact that the same customer entered across multiple systems will rarely have identical records. This variation is what the agent responds to, not duplication volume itself. Experiment 1b will isolate the effect of specific field variation directly. |
| H3 | Aggregate Output Inflation: Duplicate records create phantom demand. An agent has no way to detect inflated volume and will produce raw decision counts that a downstream consumer treats as real demand, invisibly distorting planning decisions. | ⚠️ Nuanced | The priority mix appears stable, roughly the same proportion of customers land in each tier regardless of duplication level but this stability is misleading. It masks raw volume inflation that scales directly with duplication. At 100% duplication, the agent produced 1,320 HIGH priority decisions from only ~430 unique customers. A practitioner seeing a stable distribution would conclude nothing is wrong while a hiring manager acting on raw output would staff for 3x the actual customers. |
| H4 | Field importance stability: Certain fields carry more decision-making weight than others. The agent determines which fields matter, so stability is not guaranteed. The hypothesis is that the same fields will consistently rise to the top regardless of duplication level. | ✅ Confirmed | Top-5 field ranking is 100% stable across all duplication levels: `last_purchase_days_ago`, `churn_risk_score`, `nps_score`, `lifetime_value_estimate`, `support_tickets_open`. Provides a data-driven, non-cherry-picked basis for 1b dithering target selection. Experiment 1b tests what happens when these specific fields contain incorrect values. |
| H5 | Duplication rate threshold: We expect decision quality to degrade gradually with a measurable inflection point, below which some duplication may be tolerable, above which degradation becomes unacceptable. | ✅ Confirmed (unexpected) | There is no safe threshold. Consistency drops immediately from 100% to 85% at the lowest tested level (10%) and flatlines across the full range. The common assumption that "a little duplication is probably fine" is empirically false. |
| H6 | Segment distortion: Customer segments are pre-assigned profile attributes (high-value, medium-value, low-value, at-risk), distinct from the HIGH/MEDIUM/LOW priority labels the agent assigns. The hypothesis: duplication does not affect all segments equally. Some may be more vulnerable based on the nature of their data. | ✅ Confirmed | `at_risk` customers received perfectly consistent decisions regardless of duplication. Their profiles contain such strong, unambiguous signals that no field variation could flip a decision. `medium_value` customers were most vulnerable, with an average 2.8% shift in how their records were prioritized across duplicate clusters. Segments with clearer, stronger signals are more resilient than those with mixed or moderate signals, regardless of how many customers are in that segment.

---

## Evaluation Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| 1. Decision Consistency (H2): How consistent are decisions across duplicate records for the same customer? | 87.7/100 | 456 customer clusters contained at least one conflicting decision across all duplication levels. The score reflects average consistency rate across all levels. |
| 2. Confidence Stability: How confident is the agent of its own decision? | 99.6/100 | Confidence ranged from 81.17% to 81.49% across all duplication levels. The score reflects stability of confidence across levels: 100 would mean perfectly identical confidence at every level. |
| 3. Priority Mix Stability (H3): Does the overall proportion of HIGH/MEDIUM/LOW decisions change as duplication increases? | 68.1/100 | The HIGH/MEDIUM/LOW decision mix shifted by at most 3.19 percentage points across all duplication levels, appearing stable. The score reflects that stability, 100 would mean zero shift at any level, lower scores indicate more drift. However this masks raw volume inflation. See H3 finding and Finding 4. |
| 4. Cost Efficiency: What % of API spend was wasted processing duplicate records that produced no additional decision value? | 56.8/100 | $14.07 of $32.60 total spend was spent processing duplicate records. 43.2% of the total run cost was waste. |
| 5. How similar is the agent's reasoning language when explaining decisions for duplicate records of the same customer? | 48.9/100 | Avg 0.489 word-overlap across duplicate pairs. Measured using Jaccard similarity, the ratio of shared words to total unique words across two reasoning texts. A score of 1.0 would mean word-for-word identical reasoning, 0.0 means no shared words. Consistent decisions don't always produce consistent reasoning. |
| 6. Human-Agent Boundary: What % of decisions fall below the confidence threshold suggesting human review may be needed? | 89.6/100 | ~10.4% of decisions had confidence scores below 0.80 which many agentic systems use as a proxy for decisions that may warrant human review. This rate held steady regardless of duplication level, meaning the agent didn't become less certain as data degraded. A confidence based escalation gate won't catch duplicate driven failures, the boundary customer chart shows this. A data quality gate upstream is more reliable than a confidence gate downstream. |
| 7. Field Importance (H4): Do the same fields consistently drive the agent's decisions regardless of duplication level? | Diagnostic | 100% ranking stability across all duplication levels, the agent consistently cited the same top fields: last_purchase_days_ago, churn_risk_score, nps_score, lifetime_value_estimate, support_tickets_open. Marked as Diagnostic rather than scored as this metric reveals which fields drive decisions and whether that's stable, but a high or low score wouldn't directly indicate good or bad decision quality. Excluded from the composite score for this reason. |
| 8. Segment Distortion (H6): Does duplication affect some customer segments more than others? Segments here are pre-assigned profile attributes (high-value, medium-value, low-value, at-risk), not the agent's priority decisions. | Diagnostic | at_risk customers received perfectly consistent decisions regardless of duplication, a strong unambiguous signals left no room for variation to flip a decision. medium_value customers were most vulnerable with an average 2.8% shift in their priority distribution. Marked as Diagnostic rather than scored as this metric reveals which customer types are most susceptible. Susceptibility alone doesn't indicate overall decision quality. Excluded from the composite score for this reason. |
| **Composite Score** | **77.4/100 — FAIR** | Weighted composite of metrics 1–6. Weights reflect relative importance to decision trustworthiness: Decision Consistency 30%, Confidence Stability 15%, Distribution Shift 10%, Cost Efficiency 15%, Reasoning Consistency 15%, Human-Agent Boundary 15%. H4 and H6 excluded as diagnostic metrics. Weights are fixed across all experiments in this series to enable cross-experiment comparison. See evaluation_metrics.json for full breakdown. |

---

## Key Findings

### Finding 1: The Cliff Edge (H5)
There is no safe duplication threshold. The rate at which the same customer receives the same decision drops immediately from 100% to 85% at 10% duplication and flatlines for the remainder of the range. Practitioners hoping to identify a safe zone for duplication will not find one. The implication is binary: deduplicate before agent processing, or accept a permanent 15% inconsistency tax on every decision the agent makes.

### Finding 2: Confidently Wrong (H2 + Confidence)
Self-reported agent confidence is essentially immovable with 99.6% stability, ranging only from 81.17% to 81.49% across all duplication levels. This is the most operationally dangerous finding. Many agentic systems use confidence score as the trigger for human escalation triggering human review if the agent is uncertain. But this experiment shows the agent is equally certain whether the data is clean or fully duplicated. An agent monitoring dashboard would show nothing wrong and the escalation gate never fires. Meanwhile, 1 in 7 customers received a different decision from the other 6. Because this failure mode is silent, scalable, invisible to standard monitoring, and worsens with volume, the next experiment asks a sharper question: what happens when the data isn't duplicated, but is wrong?

### Finding 3: The Agentic Blind Spot (H1 + H3)
A human analyst reviewing duplicate records would eventually notice they had seen the same customer before. An agent operating without memory will not, ever. This architectural property that makes agents attractive, (consistency, tirelessness, no cognitive bias,etc.) simultaneously eliminates an incidental deduplication check that humans provide naturally. Agentic systems therefore require compensating upstream controls, checks that human workflows provide naturally and for free. This is a systemic architectural risk, not merely a data problem.

### Finding 4: Volume Inflation (H3)
The decision mix appears stable, the proportion of customers in each priority tier holds roughly steady regardless of duplication level. But this stability is misleading. At 100% duplication, the agent produced 1,320 HIGH priority decisions from 2,882 records while the true count of unique HIGH priority customers was approximately 430.

Consider a customer support team using this agent to plan staffing. The output says 1,320 customers need immediate attention. The manager hires accordingly, headcount, shift coverage, SLA commitments, etc. The true demand is 430. The team is overstaffed by 3x, budgets are blown, and nobody knows why because the agent's confidence was high and the priority mix looked stable throughout.

This failure mode is invisible in percentage based metrics. It requires comparing raw decision counts against unique customer counts to detect.

This isn't a failure of the agent's decision logic. The agent correctly prioritized each record it saw. It's a consequence of the practitioner assumption: 'the agent will deal with it.' Deduplication feels like overhead until the headcount projections come in at 3x reality."

### Finding 5: Wasted Spend (Cost)
43.2% of total API spend ($14.07 of $32.60) was wasted processing duplicate records that added no decision value. At 100% duplication, the cost per unique customer decision is 2.87x the clean baseline.

The temptation is to let the agent deal with it. Duplication feels like a data hygiene problem, not a budget problem, but this experiment shows it is both. The agent processes every duplicate without complaint, without flagging the redundancy, and without any signal that money is being spent on decisions that have already been made.

This waste is easy to dismiss at experiment scale, $14 feels inconsequential, but inference costs scale linearly with record volume. An enterprise running this agent against a 10 million record dataset with 20% duplication isn't wasting $14, it's wasting hundreds of thousands of dollars re-deciding customers it has already seen.

### Finding 6: Boundary Customer Vulnerability (H2 + H4)
Field variation in duplicate records produces inconsistent decisions primarily for customers whose profiles sit near a decision boundary such as not clearly HIGH and not clearly MEDIUM. Customers with strong, unambiguous signal (e.g. the at_risk segment) are immune, their profiles are so clearly defined that no plausible field variation could flip the outcome.

A practitioner might conclude from this that pre-filtering by signal strength is the solution, deterministically assigning clear-cut cases and only sending boundary customers to the agent. That's a valid architectural pattern, and one worth exploring. But it doesn't eliminate the deduplication problem, it narrows it. Boundary customers are still processed from whatever records the system has, and if those records are duplicated, conflicting decisions follow. The 14% inconsistency tax from Finding 1 doesn't disappear, it concentrates on exactly the customers where it matters most.

Boundary customers are disproportionately affected and are often the most consequential to classify correctly. They represent the edge cases where the evidence is genuinely ambiguous, where a wrong call has real business impact, and where human judgment adds value the agent simply doesn't have.

### Finding 7: Consistent and Predictable Field Reliance (H4)
The agent's top 5 field ranking is identical across all 8 duplication levels: last_purchase_days_ago, churn_risk_score, nps_score, lifetime_value_estimate, and support_tickets_open. The agent was never told which fields matter, it determined this from the data. Yet decisions converged on the same priority ordering at every duplication level.

This convergence isn't strictly a duplication finding, the agent would likely show similar field reliance on clean data alone. What duplication did was confirm the stability of that ranking across challenging conditions. The agent's field priorities held up even as duplication increased, suggesting these aren't artifacts of any particular dataset condition but a reflection of how the agent reasons about customer prioritization.

This convergence is the foundation for Experiment 1b as I now have an empirically grounded list of the fields most likely to flip the agent's decisions when corrupted. The next experiment will introduce subtle errors into these fields and measure how decision quality degrades.

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
| **Total** | **15,110** | **$32.70** | | | | |

---

## Limitations

1. **Self-reported confidence is not calibrated.** Agent confidence is part of the LLM output, not an externally validated score. The flat confidence curve is itself a finding but does not tell us whether individual decisions are objectively correct.
2. **No ground truth for decision accuracy.** I measure consistency, distribution, and reasoning similarity, not whether any individual decision is correct. A rule-based ground truth function is future work.
3. **Single model, single temperature.** `claude-haiku-4-5-20251001` at 0.0. Results may differ across models or temperature settings.
4. **Synthetic data.** All records were generated specifically for this experiment. Real-world enterprise data carries distributions, correlations, and edge cases that synthetic generation cannot fully capture.
5. **Uniform duplication distribution.** Duplicates were distributed evenly across segments. In real enterprise data, high-value customers interacting across multiple channels would likely be disproportionately duplicated. The data generator supports segment-biased duplication via --segment-bias (e.g. --segment-bias high_value weights duplication 4x toward a specific segment), but this flag was not used in this experiment. H6 effects observed here are therefore conservative estimates of real-world segment distortion.
6. **Retail domain only.** I chose to test using a customer segmentation, retail domain scenario. Conclusions are assumed but not yet proven to generalize across other domains.
7. **Prompt design influence.** Prompt design influence (identified post-run). The system prompt provides illustrative examples for each priority level. These may implicitly constrain the agent's decision space. Spot checks of reasoning confirm the agent cited data-driven signals rather than matching example patterns, but a prompt controlled rerun is planned for full transparency.
8. **Variation embedded in duplication** Variation is embedded in duplication. Clusters in this experiment contain records ranging from near identical to significantly different, reflecting how duplicates appear in real enterprise systems. While natural variation likely influenced which specific decisions were inconsistent, the evaluation metrics were designed to measure the effects of duplication. Experiment 1b will isolate the effect of specific field variation directly.

---

## Illustrative Examples

These examples are drawn from the `inconsistent_examples` field in `evaluation_metrics.json`. They are included here to make the quantitative findings concrete. Each example is identified by customer ID and record IDs for full traceability back to the decision output files.

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

### Example 2: Mario Cohen — Cluster Split, 6 Records (CUST_000520)

**The finding it illustrates:** Cluster-level inconsistency at scale (H2) and the role of both data variation and LLM stochasticity in producing conflicting decisions.

**What happened:** Six duplicate records for the same customer. Four records have essentially identical key signals (364 days since purchase, 0.26 churn risk, $1,293 spend) and only the name formatting varies (Mario Cohen, mario cohen, MARIO COHEN, Maryo Cohen, etc). Two records contain significant variation in the underlying data. Yet the agent split the cluster 3 HIGH / 3 MEDIUM.

| Record | Name | last_purchase | churn | spend | Decision |
|--------|------|---------------|-------|-------|----------|
| REC_088D66E55935 | Mario Cohen | 364 | 0.26 | $1,293.61 | MEDIUM |
| REC_2FA77D9EE3BD | Maryo Cohen | 362 | 0.26 | $1,293.61 | **HIGH** |
| REC_CD75993D7BED | mario cohen | 364 | 0.26 | $1,293.61 | MEDIUM |
| REC_B83F43D0C850 | Mario Cohen | 364 | 0.26 | $1,293.61 | MEDIUM |
| REC_5C8B64AFC930 | MARIO COHEN | 362 | 0.28 | $1,280.92 | **HIGH** |
| REC_2A82266B0986 | Merio Cohen | 391 | 0.18 | $1,254.67 | **HIGH** |

**Why it matters:** Mario's case reveals two distinct failure modes operating simultaneously. The two records with significant data variation (5C and 2A) received HIGH decisions, variation is doing real work flipping the outcome. But REC_2FA77D9EE3BD also received HIGH despite having data nearly identical to the three MEDIUM records. There's only a 2-day difference in last_purchase_days_ago and a different name spelling. At temperature 0.0, the LLM still produces non-deterministic outputs for effectively identical inputs.

This is a profile that sits squarely on a decision boundary. Even more telling, one might assume a clean break of 4 records (the near-identical ones) landing on one decision and 2 (the varied ones) on another. Instead the split is 3/3. Both data variation and inherent LLM stochasticity appear to have contributed to this inconsistency. If this cluster were deduplicated prior to the agent's decision, there would be no ambiguity as to the decision for the same customer.

This is one customer in a thousand. Multiply this kind of boundary zone ambiguity across millions of customer records and the operational consequences become impossible to ignore. The agent will produce coherent reasoning for every record, the confidence scores will look reasonable, and the contradictions will only surface when someone bothers to look at the cluster level, which, as established, very few, are is set up to do.

---

## Implications for 1b Design - Data Dithering

1. **Dither target fields:** Use H4 findings as data-driven targets: `last_purchase_days_ago`, `churn_risk_score`, `total_spend`, `lifetime_value_estimate`. These four were selected as monetary or temporal fields most likely to drive boundary flipping behavior. Selection is empirically grounded, not cherry-picked.
2. **Control condition:** Dither identity fields (`name`, `email`) as a control, H4 predicts minimal effect.
3. **Boundary customer tracking:** Explicitly identify and track boundary customers. Customers near the HIGH/MEDIUM decision boundary are most vulnerable and most important.
4. **Batch API:** Anthropic's batch API offers 50% cost savings on the same model and outputs. To be validated before the next experiment kicks off.

---

# Experiment 1a Results: Dedup Impact on Agent Decision Making

**Status:** ✅ Complete  
**Date:** March 2026  
**Model:** `claude-haiku-4-5-20251001`  
**Temperature:** 0.0  
**Records:** 1,000 base customers, 8 duplication levels (0–100%), 15,110 total records  
**Total Cost:** $27.54  

---

## Hypotheses

| ID | Hypothesis | Result | Summary |
|----|-----------|--------|---------|
| H1 | Record-level decision integrity | 🔲 Pending | |
| H2 | Cluster-level consistency | 🔲 Pending | |
| H3 | Aggregate distribution drift | 🔲 Pending | |
| H4 | Field importance shift | 🔲 Pending | |
| H5 | Duplication rate threshold | 🔲 Pending | |
| H6 | Segment distortion | 🔲 Pending | |

---

## Evaluation Metrics

*To be populated after evaluator run.*

| Metric | Score | Notes |
|--------|-------|-------|
| 1. Decision Consistency | | |
| 2. Confidence Stability | | |
| 3. Distribution Shift | | |
| 4. Cost Efficiency | | |
| 5. Reasoning Quality (Jaccard) | | |
| 6. Human-Agent Boundary | | |
| 7. Field Importance (H4) | | |
| 8. Segment Distortion (H6) | | |
| **Composite Score** | | |

---

## Key Findings

*To be populated after evaluator run.*

---

## Pre-Evaluator Observations (from agent run output)

The decision distribution across all 8 duplication levels was remarkably stable based on raw agent output:

| Level | Records | HIGH | MEDIUM | LOW | Confidence |
|-------|---------|------|--------|-----|------------|
| 0% | 1,000 | 43.0% | 54.4% | 2.6% | 0.8134 |
| 10% | 1,183 | 42.0% | 55.1% | 2.9% | 0.8117 |
| 20% | 1,402 | 44.2% | 53.4% | 2.4% | 0.8139 |
| 30% | 1,587 | 43.3% | 54.1% | 2.6% | 0.8130 |
| 40% | 1,772 | 44.6% | 52.4% | 3.0% | 0.8149 |
| 50% | 1,894 | 44.6% | 53.2% | 2.2% | 0.8135 |
| 75% | 2,390 | 45.2% | 52.3% | 2.5% | 0.8142 |
| 100% | 2,882 | 45.8% | 51.2% | 3.0% | 0.8136 |

**Note:** The aggregate distribution and confidence appear stable at the surface level. The evaluator's cluster map join (H2) and segment analysis (H6) are needed to determine whether this stability masks underlying inconsistencies at the individual customer level.

---

## Limitations

1. Agent confidence is self-reported and not calibrated against ground truth
2. No objective ground truth for individual decision accuracy
3. Single model (`claude-haiku-4-5-20251001`) at temperature 0.0
4. Synthetic data — real-world distributions may differ
5. Retail domain only — generalizability assumed, not proven

---

## Cost Analysis

| Level | Records | Cost | Cost/Unique Customer |
|-------|---------|------|---------------------|
| 0% | 1,000 | $2.32 | $0.00232 |
| 10% | 1,183 | $2.74 | $0.00274 |
| 20% | 1,402 | $3.24 | $0.00324 |
| 30% | 1,587 | $3.67 | $0.00367 |
| 40% | 1,772 | $4.09 | $0.00409 |
| 50% | 1,894 | $4.38 | $0.00438 |
| 75% | 2,390 | $5.52 | $0.00552 |
| 100% | 2,882 | $6.65 | $0.00665 |
| **Total** | **15,110** | **$27.54** | |

At 100% duplication, cost per unique customer decision is 2.9x the clean baseline — purely wasted spend with no additional decision value.

---

## Next Steps

- [ ] Run evaluator and populate hypothesis results table
- [ ] Populate key findings section
- [ ] Draft LinkedIn post
- [ ] Draft research article
- [ ] Use H4 field importance findings to design 1b dithering targets

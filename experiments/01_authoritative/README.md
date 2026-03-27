# Pillar 1: Authoritative

## The Contract Clause
*Authoritative data tells the truth. It is the single, trusted, non-contradicted source of record.*

When an AI agent makes a decision, it implicitly assumes the data it receives is authoritative — that it reflects a single agreed-upon version of reality. In practice, enterprise data is rarely this clean. Records are duplicated across systems, values drift or are entered incorrectly, source systems disagree, and provenance is invisible to downstream consumers.

This pillar tests what happens when that assumption breaks down.

---

## Experiments

| ID | Name | Status | Key Question |
|----|------|--------|-------------|
| [1a](./1a_dedup/) | Dedup | ✅ Complete | How do duplicate records affect agent decision consistency, confidence, and aggregate accuracy? |
| 1b | Dithering | 🔲 Next | What happens when field values are present but subtly wrong or corrupted? |
| 1c | Quality vs Quantity | 🔲 Backlog | Does a larger dataset with noise outperform a smaller clean dataset? |
| 1d | Conflicting Authorities | 🔲 Backlog | What happens when two authoritative source systems disagree on the same customer? |
| 1e | Incompleteness vs Noise | 🔲 Backlog | Are missing fields and corrupted fields equivalent failure modes? |
| 1f | Provenance Blindness | 🔲 Backlog | Does knowing the source confidence of a field change agent decisions? |

---

## Shared Methodology

All Pillar 1 experiments use the same base pipeline:

- **Data generator:** `generate_customer_data.py` — produces synthetic customer records with controlled failure mode injection
- **Agent:** `business_decision_agent.py` — customer prioritization (HIGH/MEDIUM/LOW) using a guided but not prescriptive system prompt
- **Evaluator:** `evaluate_decision_quality.py` — 8 metrics, composite score, visualizations, markdown report

The pipeline is locked after 1a for comparability. See `RESEARCH_NOTES.md` for methodology decisions.

---

## Cross-Experiment Findings

*To be populated as experiments complete.*

---

## Notes for 1b Design

1a's H4 field importance analysis identified the fields the agent relies on most heavily:
- `last_purchase_days_ago`
- `churn_risk_score`
- `total_spend`
- `lifetime_value_estimate`

These should be the primary dither targets in 1b — this is data-driven field selection, not cherry-picking. See `RESEARCH_NOTES.md` for the rationale.

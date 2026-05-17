# Pillar 1: Authoritative

## The Contract Clause
*Authoritative data tells the truth. It is the trusted, ground-truth source of record.*

When an AI agent makes a decision, it implicitly assumes the data it receives is authoritative, that it accurately reflects reality. In practice, enterprise data rarely meets this expectation. Records are duplicated across systems, values drift or don't reflect the truth, source systems disagree, and provenance is invisible to the agents making decisions against it.

This pillar is the foundation of the research series. Before an agent can make a good decision, the data it acts on must tell the truth. These experiments test what happens to decision quality and attempt to quantify the cost when it doesn't.

If decision quality is how we keep score, authoritative data is where the game begins..

---

## Experiments

| ID | Name | Status | Key Question |
|----|------|--------|-------------|
| 1a | Dedup | ✅ Complete | How do duplicate records affect agent decision quality? |
| 1b | Dithering | 🔲 Next | What happens when field values are present but don't reflect the truth? |
| 1c | Quality vs Quantity | 🔲 Backlog | Does a larger dataset with noise outperform a smaller clean dataset? |
| 1d | Conflicting Authorities | 🔲 Backlog | What happens when two authoritative source systems disagree on the same customer? |
| 1e | Incompleteness vs Noise | 🔲 Backlog | Are missing fields and untruthful fields equivalent failure modes for decision quality? |
| 1f | Provenance Blindness | 🔲 Backlog | Does knowing where the data came from change how well an agent uses it? |
| 1g | Data Lifecycle Stage | 🔲 Backlog | Can an agent identify which stage of the data lifecycle best serves a given business requirement? |

**Results:** Completed experiments document findings in `[experiment_id]_RESULTS.md` within each experiment folder.

---

## Shared Methodology

All Authoritative Data experiments share a common evaluation approach:

- **Data generator:** `generate_customer_data.py` : Produces synthetic customer records. Baseline customer profiles will be reused across experiments however the specific data condition being tested will vary per experiment.
- **Agent:** `business_decision_agent.py` : Customer prioritization (HIGH/MEDIUM/LOW) using a guided but not prescriptive system prompt. The agent determines its own classification approach from the data.
- **Evaluator:** `evaluate_decision_quality.py` : Structured evaluation measuring the effect on decision quality. Core evaluation logic is consistent across experiments for comparability. Specific metrics may be extended per experiment.

A note on pipeline stability: The evaluation pipeline (agent + evaluator) is kept consistent across Pillar 1 experiments to enable meaningful comparisons. The data generator will evolve per experiment to introduce the specific data condition being tested. See `RESEARCH_NOTES.md` for methodology decisions and the rationale behind key design choices.

---

## Cross-Experiment Findings

*To be populated as experiments complete.*

The potential here is significant. Decision Quality as a framework allows us to compare the *cost* of different authoritative data failures directly. I identified a 14% decision inconsistency tax from duplication. That's interesting standalone but comparing it against a potential X% tax from dithering, and whatever Conflicting Authorities introduces, is where the most actionable insights will surface. Rather than treating each data condition as an isolated data quality problem, these experiments build toward a unified view of how different authoritative data conditions affect the trustworthiness of agent decisions.

---



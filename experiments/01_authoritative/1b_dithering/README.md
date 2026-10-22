# Experiment 1b: Dithering

**Status:** in progress. Pre-registration: the design and analysis are being fixed before the main run. No confirmatory results exist yet.

## The question

1a asked what happens to an agent's decisions when the data is duplicated. 1b asks: **what happens when the data doesn't tell the truth?**

We give the agent 1,000 synthetic customer records and ask it, one at a time, to set a priority (high, medium or low). Then we make the data wrong
in controlled ways (a value a little off, a value that makes no sense, two fields that disagree, many fields at once) and compare each decision with the
agent's own decision on the clean record. We measure how much decisions depend on the data being right, not whether they are right.

That breaks into three business questions:

1. **Will my agent's decisions change?** (H2, H3, H4, H7, H8)
2. **Will it tell me?** (H5)
3. **What should I fix, and in what order?** (H1, H6)

## Scope of the evidence

- **A minimal agent:** one model call per customer; no tools, no memory between calls, no retrieval. It is the decision step of an agent, isolated.
- **One model, one prompt:** `claude-haiku-4-5-20251001`, temperature 0, with a fixed prompt that gives no decision criteria. The design works for any model; the numbers are for this one.
- **Synthetic customers:** no real people.
- **No outcome labels:** we measure how much decisions move, not whether they are right.

## Which document to read

| Document | What it is |
|---|---|
| `1b_DESIGN.md` | The original design (pre-registration, before any code) |
| `1b_DESIGN_AMENDMENT_1.md` | **The current, binding design:** hypotheses H1 to H8b, conditions, statistics, freeze rules |
| `macro_review.md` | A plain-language review of whether the design still answers the question, with a numbered decision log (D1 onward) |
| `wording_sheet.md` | The source of truth for every public statement about 1b; number slots are filled after the run |
| `amendment_fold_in_ledger.md` | Every change to the amendment, with its reason |
| `statistical_methodology_note.md` | Why each statistical tool was chosen |
| `h5_working_doc.md` | The working record for H5 (does the agent flag bad data?) |

## Ground-truth data

The ground-truth dataset is validated against all six DAMA data quality dimensions (Accuracy, Completeness, Consistency, Timeliness, Validity, Uniqueness)
as part of generation; the pipeline fails loudly if validation doesn't pass. The full audit result is saved to
`experiments_output/ground_truth/dama_audit_report.json` as a verifiable artifact. To confirm the claim yourself, re-run the validator:

```bash
python shared/data_generation/validate_dama_dimensions.py \
    --input experiments_output/ground_truth/canonical_customers.json
```

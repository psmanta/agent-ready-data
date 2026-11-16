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
| `booth_card.md` | One page: the seven rules behind the decisions, lines that answer most questions, and phrases never to use (`check_wording.py` scans a draft for them) |
| `amendment_fold_in_ledger.md` | Every change to the amendment, with its reason |
| `statistical_methodology_note.md` | Why each statistical tool was chosen |
| `h5_working_doc.md` | The working record for H5 (does the agent flag bad data?) |

## Run the pattern on your own agent

This folder is a research harness built on a controlled retail schema; it is not a package for your stack. The pattern behind it fits in a
small wrapper around your agent's decision step, in any language or framework. A sketch (pseudocode):

```text
# 1. The decision is the action your agent commits to, not its prose.
decision(trace)      = canonical(final_action(run_agent(trace, tools = DRY_RUN)))
#    canonical(): a tool call or routing edge becomes an enum (approve / block / escalate);
#    a number becomes the bands your business cares about; a ranked list becomes its top few.

# 2. Tools: stub the ones that WRITE (capture the arguments, return a stubbed success, execute nothing);
#    replay recorded responses for the ones that READ, so the input never changes between runs.

# 3. Baseline: replay each unchanged trace 5 times, spread over different hours or days
#    (in 1b's checks, repeats inside one session agreed; the changes appeared between sessions).
stable = traces whose 5 baseline decisions all agree

# 4. Perturb at the state boundary, one field at a time, just before the decision step runs,
#    with a believable wrong value AND an absurd one.
for field in fields:
    for value in (believable_wrong(field), absurd(field)):
        flip_rate[field, value] = share of stable traces whose decision differs from baseline
```

**A canary in production** is the same wrapper on a fixed set of stable traces: run it on a schedule and on every prompt deploy or model-version
change. Stub the writes and replay the reads, so a flip means the model moved, not the data. Measure how often the canary moves on its own for a
week or two before go-live and set the alarm above that (or alarm only when it trips twice in a row); when it trips, hold the affected decisions
and route them to a person. **Canary the decision, not the action.**

**Real traces are real data.** Mask customer identifiers and run the test in an environment your organisation has approved for sending data to
a model provider; a permanent canary set built from historical traces is retained customer data.

**Cost.** The API cost of a single-agent check is small; the real cost is a few days of an engineer's time, and the baseline replays take a few
days by design.

## Ground-truth data

The ground-truth dataset is validated against all six DAMA data quality dimensions (Accuracy, Completeness, Consistency, Timeliness, Validity, Uniqueness)
as part of generation; the pipeline fails loudly if validation doesn't pass. The full audit result is saved to
`experiments_output/ground_truth/dama_audit_report.json` as a verifiable artifact. To confirm the claim yourself, re-run the validator:

```bash
python shared/data_generation/validate_dama_dimensions.py \
    --input experiments_output/ground_truth/canonical_customers.json
```

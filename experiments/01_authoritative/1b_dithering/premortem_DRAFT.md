# Pre-mortem of the confirmatory run (DRAFT, for owner review; becomes RT8 in macro_review.md)

**The setup.** It is six weeks after the run and the result is unusable. Why? Each failure story is either already
defended (a mechanism exists and is tested) or a gap with a proposed fix to pre-register before the collection freeze.
Nothing here is decided until the owner approves it.

## Already defended

| # | Failure story | Defense (built and tested) |
|---|---|---|
| D-a | The laptop sleeps or the terminal dies mid-run (about 50 hours, 48 sequential batches) | `caffeinate -i -w` on macOS; batch id saved before waiting; re-running resumes the same batch and never pays twice (`test_batch_runner.py` Q4) |
| D-b | An SDK upgrade moves calls to API version 2026-10-21, which rejects `temperature` | `anthropic-version: 2023-06-01` pinned on both paths; exact library versions pinned; a real run refuses on mismatch (R3-16, D75) |
| D-c | A file is edited after the freeze | freeze manifest (write once, verify before the run and before analysis); generation manifest with every generated file hashed |
| D-d | The wrong seed, or pilot customers leak into the confirmatory data | explicit seed 20261016; `check_pilot_disjoint.py`: seed, integrity, provenance, profile-level disjointness |
| D-e | Haiku 4.5 is retired mid-run | at least 60 days' notice (Anthropic's stated policy) against a run of days; it is Active with no deprecation as of 23 October 2026. Replication after retirement is a disclosed limit, not a run risk |
| D-f | Window effects masquerade as condition effects | randomized block design; clean re-run control; stable-primary stratification (R3-9) |

## Gaps, each with a proposed fix

| # | Failure story | Proposed fix (to pre-register) |
|---|---|---|
| G1 | **Cost overrun.** About 61,000 calls at about $0.00128 each on batch is about $80, but retries are capped only per block | A hard budget in `run_collection.py` (proposed $150): before submitting a block, refuse if spend so far plus the block's estimate would exceed it. Plus a spend limit in the Anthropic Console as a second guard |
| G2 | **The control is too noisy to interpret.** Window-to-window jitter is so large that M = 10 points cannot be told from noise | Pre-register an interpretability rule: if the clean re-run's drift in the stable stratum exceeds a threshold (proposed: 10 points, equal to M), every confirmatory claim is reported as "inconclusive at this noise level" and the window effect becomes the headline finding |
| G3 | **The stable stratum is too small.** Many customers flip across windows, so fewer are stable than the power calculations assumed (850) | Pre-register a minimum (proposed: 600). Below it the tests still run, the D59 operating characteristics are recomputed at the realized size (already planned), and any rule whose power falls below 50% for its planted effect is reported as underpowered |
| G4 | **Garbled output.** The agent's format drifts, or an outage leaves many requests errored or expired | Pre-register a ceiling: if more than 2% of tasks are still unparseable or missing after retries, stop and investigate before any analysis; the rate is always reported |
| G5 | **The human audit fails.** Agreement stays below 0.60, or too few raters finish | Consequence rules already exist (judge Tier 1 / Tier 2 / dropped) and calibration catches codebook problems early. **New:** if fewer than 6 raters finish, re-assign their items to the rest within the 150 cap; if coverage still falls below the R3-6 minimum, the audit is reported as incomplete and the judge stays Tier 2 |
| G6 | **A bug is found after looking.** An evaluator error is discovered after the analysis freeze | A pre-registered deviations log: a fix is allowed, dated and disclosed; both the original and the corrected result are reported; no other change rides along |
| G7 | **Everything comes back null.** | Not unusable: pre-registered nulls are findings. Check that the wording sheet has a null version of every claim (the macro review's "if the story does not land" section) |

## Owner decisions needed

1. G1: the budget figure ($150 proposed).
2. G2: the noise threshold (10 points proposed) and the consequence ("inconclusive"; window effect as headline).
3. G3: the minimum stable stratum (600 proposed).
4. G4: the ceiling (2% proposed).
5. G5 and G6: accept as drafted?
6. G7: a wording-sheet check (no decision; a task).
7. Anything missing?

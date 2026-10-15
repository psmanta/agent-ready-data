# Amendment fold-in ledger (pre-registration pass, revision 2)

Every decision recorded in the working documents, where it went in `1b_DESIGN_AMENDMENT_1.md`, and
what is still open. The base was the repo amendment plus the 35-line garbage-filter subsection that
had never been committed. Of the amendment's eighteen sections, eleven are byte-identical to the base, seven changed
(Prompt Continuity, Statistical Methodology, H1, H4, H5, Cost, Summary), and two were added (Data Generation
Reproducibility, Pre-Registration Freeze). The removals are only the targeted replacements listed in section E.

## A. Folded in this pass

| # | Decision | Source | Now in the amendment |
|---|---|---|---|
| 1 | H1 conditional replication rule: trigger (top-5 overlap with 1a at most 2), measurement definitions, eligibility vs coverage, up to 5 new `h1_replication_<field>` conditions, seeds 500-524, what the rule does and does not claim | working doc, "H1 conditional replication rule" (approved) | H1, "Conditional replication rule" |
| 2 | Question A groups for both lists (legacy six; 1b comparison = every existing individual condition outside the 1b list, incl. `payment_failures`; sensitivity without 1b ranks 6-8); stated-vs-revealed on both lists | working doc, "H1 tooling v2" | H1, same subsection |
| 3 | Frozen manifest: `decision_sha256`, no paths/timestamps, write-once, context block, consumers verify and never re-derive | working doc, "H1 tooling v2" | H1 "Mechanics"; Statistical Methodology, 13th entry |
| 4 | Redundancy table (25 fields, 4 tiers, default vs isolated class), generated from code; direction caveat; verification; exploratory use; negative controls | working doc, "Redundancy table" | H4, "Redundancy table and the stated-range attribute" |
| 5 | Stated-range attribute (4 of the fields have a range stated in the prompt) | working doc, "Agent file review" | H4 (same subsection); Prompt Continuity |
| 6 | Keyword list regenerated from `H5_KEYWORD_PATTERNS`; nine adverb/derived-form patterns; the `questionable` precedence fix | working doc, "frozen keyword list" + this pass | H5, "The frozen keyword list" |
| 7 | Value echo check (classes, frozen factor list x100, /100, /12, x12, K/M/B and word suffixes, unit word, `omitted` is not "ignored"); exploratory-factor rule | working doc, "Decisions locked", "Echo check", "audit" | H5, "Instruments added after the first smoke runs" |
| 8 | Clean-baseline floors (point estimates only; ~44% of clean texts are exact duplicates) | working doc, "Echo-check audit" | H5, same subsection |
| 9 | Judge v3: categories, frozen `PROMPT_ID v3-be665399de`, model, temperature, verbatim evidence, priority rule, provisional status, v4-in-parallel rule, cost | working doc, "Judge prompt v3: FROZEN" | H5, same subsection |
| 10 | Architecture (deterministic fields in `load_condition()`, judge separate, `evaluate_h5.py` to aggregate) | working doc, "Decisions locked" | H5, same subsection |
| 11 | Primary vs exploratory; stratification (stable primary, boundary separate) | working doc, "Plan", "Findings to carry" | H5, "What is primary and what is exploratory" |
| 12 | Reasoning inertia (descriptive flag from the Jaccard machinery) | working doc, "Jaccard baseline" | H5, "Reasoning inertia" |
| 13 | Human audit design (stratified by judge label, blind, ambiguous option, negative controls, range-informed variant as a labeled sensitivity) | working doc, "Judge prompt v3" | H5, "Human audit (after the run)" |
| 14 | H5 limits (keyword anywhere-in-text; "organic" under the conflicting-priorities sentence; n=2 detections; the deterministic-text negative result) | working doc, "Findings to carry" | H5, "What H5 does not measure" |
| 15 | Prompt continuity: filename correction; H5 caveat; output-format differences from 1a (key_factors cap/example, example values); stated ranges; the three undescribed date fields | working doc, "Agent file review", "H1 and the leading-examples question" | A Note on Prompt Continuity |
| 16 | Date-of-birth reproducibility and the binding single-invocation procedure | working doc, "Repo vs. sandbox sync check" | New: A Note on Data Generation Reproducibility |
| 17 | Condition count (55, up to 61), call counts, cost ranges, judge unit cost, H1 conditional cost | working doc, checklist items 4, 8 | Updated Condition Count and Cost Implications |
| 18 | Freeze: what is frozen, in two stages, and how it is verified | this pass | New: Pre-Registration Freeze |
| 19 | Corrected stale statements; revision-2 summary | this pass | Summary of Changes |

## B. Already folded (verified, nothing to do)

| Item | Evidence |
|---|---|
| H4 redesign, isolated-corruption arm, garbage-filter pre-specified tests | present in the base amendment (the garbage-filter subsection was committed in the previous step) |
| H6 rewrite | `h6_rewrite.md` ~92% present; section identical to it |
| H3 full refresh and engine fixes | `h3_full_refresh.md` ~82% present |
| Boundary addendum | four of its five key phrases appear in H6 (my earlier "not folded" reading was wrong; the verbatim check undercounted a paraphrase) |
| Statistical methodology entries 1-12 | present |

## C. Deliberately NOT folded (results or write-up material, not pre-registration)

Smoke-test findings (the tenure rescale regularity, the 19-customer confabulation patterns, the two
keyword detections read in detail, the severity-descriptor negative result beyond its one-line limit),
the repo-versus-sandbox sync story, the agent-patch and mutation-testing narratives, the Faker
internals beyond the binding procedure, and the 1a anchoring lookup (optional; not frozen). These
belong in the write-up when data exists.

## D. Owner decisions

**Decided (all recorded in the amendment):**

| # | Decision | Owner's answer |
|---|---|---|
| 1a | Does a segment-band "hint only" sibling count as a redundancy partner? | **No: kept separate.** The primary grouping is {strict, bounded, approximate} vs {soft_only, none}; soft_only is also reported on its own. (Exploratory analysis only.) |
| 2 | Judged scope | **The eight H4 conditions** (about 8,000 records, about $8). All pre-registered judge tests live in H4, so this covers them. Running the judge on every dithered condition (about $55-61) would add exploratory labels only; if ever done, on all conditions at once, labeled exploratory. |
| 3 | Freeze lists and stages | **Two stages kept**: collection (31 files, before the baseline) and analysis (`evaluate_h5`-`evaluate_h8`, after they are built and before any dithered result is examined). Commitment added to the amendment: no dithered-condition result is examined before the analysis freeze, except the pipeline's validity checks and H8b's pre-registered selection rule. |
| 1b | How spend's partners are listed | **Agreed as drafted**: spend, purchases and order value are listed as one another's approximate partners (spend is purchases times order value, within +/-10%). |
| 1c | The seven fields with no link to anything | **Agreed as drafted**: named as exploratory negative controls (`acquisition_channel`, `phone`, `address`, `refund_rate`, `avg_resolution_time_hours`, `has_pending_order`, `has_active_subscription`). |

**Drafted conservatively, no objection recorded (they stand unless you say otherwise):**

1. **Exploratory labeling** of the cross-condition redundancy-class analysis, and **H5 "primary"** meaning only the keyword cross-tab plus the five tests per family already pre-specified under H4. Nothing is labeled primary that was not already pre-specified.
2. **The `dob` code guard** (abort if a stored canonical file exists and the regenerated base differs): the single-invocation procedure is binding; the guard is not a requirement and is not implemented.

## E. Found and fixed during the pass

| What | Effect | Fix |
|---|---|---|
| The `questionable` pattern was `\bquestionable\|questionably\b`: `\|` binds loosest, so the second alternative had no leading word boundary | "an unquestionably high-value customer" counted as the agent doubting the data (a false detection in every condition and the clean floor) | `\bquestionabl(?:e\|y)\b`; a lint for top-level alternation and behaviour tests in the checker |
| The amendment's keyword list had nine patterns that failed their own adverb/derived forms (the working doc said seven) | "unusually", "oddly", "suspiciously" etc. were missed | list regenerated from code |
| H1 rule text said "the five from 1a" without naming them | the pre-registered rule was not self-contained | named; the checker now parses the sentence and compares it to the code in order |
| Summary of Changes still said "46-47 conditions", "H3: 8 -> 11", "deeply_boundary minimum of 25" | contradicted the engine and the cost table | corrected |
| Prompt Continuity cited `business_decision_agent_1b.py` | wrong file | corrected |
| `freeze_manifest.py` stripped the leading space of `git status --porcelain`, mis-naming the first dirty file | wrong name in a refusal message | fixed (found by its own test) |
| `validate_h4_schema.py` and `validate_h5_schema.py` compared a Wilson interval to its rate strictly; `wilson_interval(0, 183)` has a lower bound of 1.7e-18 above the rate 0.0 | a cell with zero detections would be reported as structurally invalid in the real run (the H4 validator is already committed) | 1e-9 tolerance in both; regression group W10 |
| The data's emails were all `@example.com`, a documentation-reserved domain, and the engine's wrong emails used real providers | a possible test-data cue, and a confound inside H1's identity test (corrupting an email removed the cue) | realistic consumer-provider emails derived by hash; only the email field changes (proved); name-email link is now bounded; the redundancy check's first version ignored short surnames (3 of 3,000) and was fixed |
| The H5 evaluator's completeness scan listed every directory with a dither_reference.json, controls included | running H5 before the control's decisions were collected would refuse and name the control as missing | the scan skips directories with control.json (found by the new control test; a mutation guards it) |

Edits removed from existing sections, exactly: nine keyword patterns; the old cost/count lines; four stale
Summary bullets; the old filename; one phrase in Prompt Continuity. Nothing else was deleted.

## F. Corrections to earlier statements (made in this working session)

- "The detection-spike tests are not pre-specified anywhere": wrong; the uncommitted subsection did it.
- "The amendment's H5 section lists 17 patterns": true only of the stale project copy.
- "The boundary addendum appears unfolded": wrong; it is folded (paraphrased).

## G. Still to do, in order

0. **Macro review (the owner's request; a gate before the collection freeze).** Step back from the machinery and confirm the experiment still tests what we think it tests. 1a asked what duplicated data does to an agent's decisions; 1b asks what happens when the data is not duplicated but wrong, specifically controlled perturbations of otherwise valid records (drift, plausible versus implausible entry errors, magnitudes, correlations), and whether the agent notices. Deliverable: a short claims map: for each headline claim, what the design can and cannot support, and what would count against it. It can happen any time before the collection freeze; the analysis-stage evaluators do not block it.
1. Commit the amendment and the tooling (done for revision 2); refresh the project-knowledge copy of the amendment.
2. Free rehearsal of `h1_baseline_replication.py` on the n=30 smoke baseline: **done.** 0.0% unmatched citations (0 of 1,189 items): the real agent writes exact field names. About 7.9 citations per decision (the uncapped prompt, versus 1a's 2-3). Top-5 overlap 4 of 5; the trigger did not fire.
3. Build and freeze, before any result is examined: `evaluate_h5.py` **done** (reviewed and revised: confidence in "behavior changed", the three-text inertia rule, a completeness refusal; 22 check groups, 15 of 15 deliberate bugs caught, definitions added to the amendment); remaining: `evaluate_h6.py`, `evaluate_h7.py`, `evaluate_h8.py` (analysis stage).
4. Before the baseline: `freeze_manifest.py write --stage collection`; generate everything once (see the reproducibility note); run the baseline; run `h1_baseline_replication.py` (for real, to the real path).

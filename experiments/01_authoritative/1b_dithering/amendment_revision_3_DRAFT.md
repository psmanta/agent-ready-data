# Amendment 1, Revision 3: DRAFT for owner review

**Status.** A draft to mark up. **Nothing here has been applied** to the amendment, the generator script, the consistency checker or the freeze list. Each change
is numbered **R3-n**, says where it goes, and carries a status: **approved** (decided by the owner) or **proposed** (my recommendation, awaiting a decision). Values
marked **[OWNER TO SET]** are my proposals for numbers that only you should choose. Line numbers refer to the current amendment (about 139 KB) and will move.

---

## 0. What reading the amendment turned up (read this first)

To build the multiplicity map I had to read what the amendment actually pre-registers, test by test. Four findings, most consequential first.

1. **H1, H2 and H3 are not stratified, although the amendment's stated reason for stratifying applies to them.** The amendment makes stable customers the primary
   stratum for H4's garbage-filter tests and for H5, because a deeply boundary customer flips about 40% of the time with no corruption at all. The H1, H2 and H3
   evaluators contain **no** stratification (zero mentions of stable or boundary); only H4 and H5 stratify. Contrasts between arms stay valid, because the same customers
   appear in both arms and their own flips add symmetric noise, but absolute drift rates are inflated and power is lower. **It also means the line "our main group is
   customers nowhere near a threshold" is true for H4 and H5 only.** I used that line in the RT1 rehearsal answer; the log is corrected. (R3-9)
2. **H2, H7 and H8a pre-register no inferential test or interval.** H2 has a ladder and an evaluator but no rule for what counts as a "cliff". H7 has a hypothesis and a
   ladder, no endpoint. H8a has a careful definition of "additive" (inclusion-exclusion) but no interval and no definition of a "meaningful" interaction. H8b is conditional.
   This blocks D34 for the cliff candidate until H2 has a test. (R3-2, R3-3)
3. **The H1, H3 and H6 headlines are "consistency across many checks" with no numeric criterion** (H1: whether the top five fields' advantage "holds consistently" across 30
   comparisons; H3: whether three comparisons "consistently" show a larger cost than the reference; H6: whether the ordering "holds consistently across most conditions").
4. **A multiplicity correction is named in exactly one place:** Holm within the two H4 garbage-filter families (five tests each). H1's 30 tests and H4's six per-field drift
   contrasts name none.

What is good: H4's design is thorough; the amendment is internally coherent; the gaps are fixable in text and analysis code **before** the freeze, and none needs new data.

## 1. Decision status

| R3 | Change | D# | Status | Where it goes |
|---|---|---|---|---|
| R3-1 | Three evidence tiers | D37 | **approved** | Statistical Methodology, new entry 14 |
| R3-2 | Multiplicity map | D35 | **approved** | Statistical Methodology, new entry 15 and a new subsection |
| R3-3 | Claim 2: candidates, materiality (M), the H2 shape test, the dominance rule | D34 superseded; D55, D56, D57 | **approved** | new entries 17 and 18 |
| R3-4 | Customer halves (the replication split) | D34 | **approved** (supports R3-3) | new entry 17, the end |
| R3-5 | Clustering-aware analysis plan | D51 | **approved** | new entry 16 |
| R3-6 | Human-audit specification | D50 | proposed | H5, "Human audit (after the run)" |
| R3-7 | Window-effect wording | D52 | **approved** | Collection Design, first paragraph |
| R3-8 | Limits of this study | D53 | **approved** | new section before the freeze |
| R3-9 | Stratify H1 to H3 like H4 | new, **D54** | **approved (option A)** | Statistical Methodology, plus evaluator work |
| R3-10 | Protocol history | D36 | **approved** | Pre-Registration Freeze |
| R3-11 | Summary-of-changes entry | | **approved** | Summary of Changes |
| R3-12 | Decision rules for H1, H3, H6, H7, H8a and their operating characteristics | D58, D59 | **approved** | new entries 19 to 23, and an appendix |
| R3-13 | Correct H3's stale "applied three times" | | **approved** | H3 |
| R3-14 | A dated revision history and a revision-specific status line | D69 | **approved** | the status line, a new section |

---

## 1b. Owner feedback, round 1

- **R3-9 / D54: approved, option A.** Extend the H1, H2 and H3 evaluators with the same stratum helper H4 uses; stable is the primary stratum, boundary and all are reported alongside.
  *Working defaults (say if you disagree):* stable first, with the unstratified "all" kept under its own key; H1's 30 McNemar tests and the Mann-Whitney run on stable as primary; H3's GEE on
  stable as primary; H2's within-segment breakdown on stable and on all; customers with no tier appear in "all" only. The evaluators gain the `--finalized_ground_truth` argument H4 already
  has, the H1 and H3 validators are updated, and new planted-data tests are added (H2 and H3 have none today). Note that the tiers now come from five runs in five different windows, so
  "stable" is a stricter class than in the pilot.
- **R3-3 / D34 versus D34b: superseded by the owner's dominance rule (D55, proposed).** The text of R3-3 will be rewritten after the H2 shape test (Choice 2) is settled, because the two are coupled.
- **Choice 2 (the H2 cliff test):** the owner's rule is written into R3-3 with five proposed refinements (marked R); awaiting acceptance. **New Choice 9:** one shared materiality value M (the smallest effect worth a headline) for both the cliff floor and D55: 10 points? **Choices 4 to 8 and the rest of the document:** awaiting the owner's review.

## 2. The proposed text

### R3-1. Evidence tiers (approved, D37)
**Where:** Statistical Methodology, new entry 14.

> **14. Evidence tiers.** Every result carries one of three labels, in the evaluators' output, in the tables and in any write-up. **Tier 1, confirmatory:** the
> pre-registered primary tests in the multiplicity map, corrected within family as the map states. Only Tier 1 results can be called findings. **Tier 2, descriptive:**
> rankings, curves, direction of change, costs, reasoning inertia, and every hypothesis whose headline is a consistency summary; reported with intervals, without p-values
> and without discovery language; rankings are shown as tiers, with conditions that cannot be told apart tied. **Tier 3, exploratory:** anything spotted after the fact;
> labelled a hypothesis for the next experiment and never a headline. Headline claims are drawn only from Tier 1, by the rule in entry 17; if nothing qualifies, the claim says so.

### R3-2. The multiplicity map (approved, D35)
**Where:** Statistical Methodology, new entry 15 ("the map below lists every pre-registered test"), and a new subsection holding the table. **The table records what the
amendment says today; the last column says what is missing.** Nothing in it is new except the last column.

| Hyp | Primary endpoint and test, as pre-registered | Stratum | K | Family and correction, as written | Tier today | Gap |
|---|---|---|---|---|---|---|
| **H1** | Per-field drift: each of the 5 self-reported top fields against each of the 6 comparison fields, paired exact McNemar (5 x 6 = 30). Headline: whether the advantage holds across the 30. Secondary: Mann-Whitney U, 5 fields vs 6 | not stated; evaluator uses all customers | 30 (+ 0 to 5 conditional replication) | one family of 30; **no correction named** | Tier 2 for the headline; Tier 1 for single tests only if a correction is named | a numeric criterion for "holds consistently"; a correction |
| **H2** | Magnitude ladder (3 fields x 4 sizes: 5, 15, 40, 100%); `analyze_field_ladder`. The claim is the *shape* | not stated; all customers | **none** | **none** | Tier 2 | a shape test (proposed in R3-3) |
| **H3** | (a) group x arm interaction in a logistic GEE clustered by customer, each real pair or the triplet against its matching reference (four real sets: pair 1 churn + NPS, pair 2 spend + lifetime value, pair 3 open tickets + payment failures, and the triplet spend + open tickets + churn; applied FOUR times); (b) confidence: unconditional paired Wilcoxon (primary); "always-drifters" (secondary) | not stated; all customers | 4 interaction tests + 4 confidence tests (one per real set; confirm against `confidence_analysis`) | **none named**; headline: all comparisons consistently show a larger decorrelation cost than the reference | Tier 1 candidates (tests are named); headline is a consistency summary | a numeric criterion; a correction |
| **H4** | (a) per field: paired exact McNemar of drift vs plausible and vs implausible (3 fields x 2 = 6); (b) garbage filter: two families, keyword detection and judge `explicit_concern`, five paired exact McNemar each | **stable (primary)**; boundary and all reported | 6 + 10 = 16 | (b) Holm within each family of five; (a) **none named** | Tier 1 for (b); (a) once a correction is named | a correction for (a); the judge family only after the audit (R3-6) |
| **H5** | Keyword detection three-way cross-tab per condition against the clean-baseline rate; the formal tests are H4's families | stable (primary) | via H4 | via H4 | Tier 2 for the cross-tabs | none beyond H4's |
| **H6** | Ordering of drift: deeply boundary > lightly boundary > stable; consistency across conditions plus a median ratio | by construction | none | none | Tier 2 | a numeric criterion (proposed below) |
| **H7** | Breadth ladder: 1, 3, 6 and all fields; hypothesis is non-linearity | not stated | **none** | **none** | Tier 2 | an endpoint and a shape rule (proposed below) |
| **H8a** | Observed combined drift vs the inclusion-exclusion prediction, 2 pairs; a surprise on the identity pair is a headline | not stated | **none** | **none** | Tier 2 | an interval for "super-additive" (proposed below) |
| **H8b** | Conditional (0 or 1 condition) | | | | Tier 3 unless it fires | |

**Pre-registered tests: 59 in eight families, each corrected by Holm WITHIN its own family (never across families); approved.** F1 H1's 30 contrasts; F2 H3's four interaction tests; F3 H3's four confidence tests; F4 H4's six per-field drift contrasts; F5 H4 keyword detection (5); F6 H4 judge `explicit_concern` (5; only if the audit passes); **F7 H2's three field-level 5%-versus-control tests; F8 H8a's two pair tests** (F7 and F8 were missing from the first list). H6 (a customer-level interval) and H7 (the significant-fall definition of erratic) are Tier 2 and carry no family.

**Numeric criteria for the H1, H3, H6, H7 and H8a headlines:** see R3-12 (the decision rules and their operating characteristics).

### R3-3. Claim 2: candidates, the materiality line, the shape test and the dominance rule (D34 superseded; D55, D56, D57)
**Where:** Statistical Methodology, new entries 17 and 18. *Status:* the five shape-test refinements and the shared materiality line are **accepted**; the dominance rule's final form (the three changes
flagged) awaits the owner's explicit OK.

> **18. Materiality.** M is 10 percentage points of net drift in the stable stratum (drift minus the clean re-run control's drift). Set before, and without reference to, any pilot drift rate (confirmed by the owner). M is the smallest effect treated as material for this decision, and it is used in exactly three places: (a) the H2 floor: a field's shape is assessed only if its peak net drift F is at least M;
> (b) the headline line: no Tier-1 headline is made on an effect below M; (c) "non-material for this decision" means the 95% interval's upper bound is below M (a result between is "inconclusive"). Results
> are also tabulated at M = 5 and M = 15 points as a sensitivity analysis; headlines are made at M = 10.

> **17. The open claim slot.** Claim 2 ("how the agent reacts") has three candidates, each defined by a pre-registered Tier-1 test: **the cliff** (H2, the shape test below), **the filter** (H4, per-field drift
> under believable versus impossible errors) and **valid but wrong** (H3's consistent-error and contradiction arms with H4's believable arm). Effects are compared as absolute percentage points of net drift in the
> stable stratum, with the direction reported (the headline wording follows the direction). A candidate is **statistically qualified** if (a) its Tier-1 test survives correction within its family and (b) it replicates
> in both customer halves (R3-4): the same sign in each half and an effect of at least half the pooled effect. **The headline** is then set by this rule, fixed before the run:
> - **Solo.** The leading qualified candidate's effect is at least M, and at least 1.5 times every other qualified candidate's, material or not, **and that holds in
>   both customer halves**. It takes the headline as the dominant reaction mechanism; the others appear alongside as secondary factors. There is no separate gap floor: it is redundant, because a leader of at least 10 points that is 1.5 times the runner-up is at least 3.3 points ahead.
> - **Plural.** Otherwise, if the leader is at least M, the headline reads "no single reaction mode dominated" and names the co-leaders (qualified candidates within 1.5 times of the leader and at least M); a
>   within-margin candidate below M appears in the tables only.
> - **One.** A single qualified candidate at least M is "the only mechanism that qualified."
> - **None.** If no candidate qualifies, or the leader is below M, the claim is: "no single reaction mechanism stood out."
> All three candidates' effects, intervals and status are always shown side by side. *Behavior (850 stable customers, 6,000 runs per scenario):* equal effects give plural 100%; a near tie (14, 10, 10 points) gives false
> dominance 5%; a clear leader (24, 8, 6) gives solo 99%; one real effect with two small ones (14, 6, 4) gives solo 78% (the both-halves requirement costs power); tiny effects (6, 4) and nothing real give null 99% and
> 100%. (Applying M to every candidate instead of the leader let a runner-up flickering across the line manufacture dominance: false solo 25% on the near tie.)

**The H2 shape test (accepted).** *Dose:* each rung (5%, 15%, 40%, 100%) is the size of each change, relative to the value, applied to every customer; it is NOT the share of records affected (the engine has no
prevalence setting for H2). *Stratum:* the stable stratum; boundary and all-customer curves are plotted alongside, each net of its own stratum's clean re-run drift, and are never used for qualification.
*Population:* the stable customers whose value actually changed at every rung, so *s* and *F* are computed on the same customers. For each of the three fields let *s* be the drift at 5% minus the control's drift, and *F*
the largest drift over the 15%, 40% and 100% rungs minus the control's drift. *Floor:* a field is assessed only if *F* is at least M; otherwise it gets "non-material for this decision: no shape claim." *Cliff:*
*s* is at least 0.5 *F* and above the control by a paired exact McNemar test (Holm across the three fields, 0.05). *Smooth:* *s* is below 0.25 *F*. *Mixed:* anything else above the floor. *Anchor:* proportional growth
would give *s*/*F* = 0.05, so smooth means within five times a straight line and a cliff means ten times or more. *Qualification:* the cliff candidate qualifies if at least two of the three fields are cliffs. **Its effect
for entry 17 is the median *s* of the cliff fields** (the number the headline quotes), so a cliff headline needs at least M points at the 5% rung, which is stricter than the floor on *F*. *s*/*F* is reported with a
customer-level bootstrap interval; a non-monotone ladder is flagged; replication of *s* in both halves is part of the gate. *Behavior:* the two example curves (14, 16, 18, 20 and 1.5, 4, 9.5, 20 points) are classified cliff
and smooth 100% of the time; a decorative field (F = 2) is below the floor 100%; a curve exactly at 0.50 or 0.25 is a coin flip the mixed zone absorbs (a true 0.32 is "mixed" 92%); with churn and spend both cliffs and tenure
decorative the candidate qualifies 87% of the time, and 0% if either is smooth or mixed. **Headline wording:** cliff: "a cliff, not a slope: errors 5% off already move [s] of stable decisions, and much larger errors add little";
smooth: "a slope, not a cliff: small errors move few decisions and bigger errors move more"; mixed: "moderate sensitivity: small errors already move some decisions for customers who were not close calls, and large errors
move more". Never "refutes" or "confirms the belief": the finding is scoped to this agent, this field and this size scale.

**Absence claims (new, follows from M).** "Non-material for this decision" is the claim for a field or group of fields (for example the contact details) whose net-drift interval has an upper bound below M, and it is stated
with the bound: "moved at most [x] in 100 decisions." "Did nothing" is not used. This replaces "decoration" in every piece.

### R3-4. Customer halves (approved, supports D34)
**Where:** entry 17, at the end.

> **Replication split.** Halves are defined by customer, not by block: half(c) is the parity of the first byte of SHA-256 of `adc-1b-halves-v1|<customer_id>`. All of a
> customer's records (every baseline run, every condition, the control) belong to that customer's half, so every paired comparison and every baseline reference stays inside
> one half. Each half holds about 500 customers (checked on 1,000 consecutive ids: 514 and 486, with every successive hundred balanced). The string is a namespace, not a secret; a new string needs a new revision. Replication is evaluated on each half separately.

*(Splitting by block would break pairing: a customer's two arms, and their five baseline runs, sit in different blocks.)*

### R3-5. A clustering-aware analysis plan (proposed, D51)
**Where:** Statistical Methodology, new entry 16.

> **16. Windows and the analysis.** The collection spreads every condition and the control over 48 blocks (windows). **Contrasts** between conditions, and between a condition and
> the control, use the pre-registered tests; because each condition is spread over all blocks to within one task, a window effect cancels in the contrast and the unblocked
> analysis is slightly conservative. **Robustness:** each primary contrast is also computed by a block-cluster bootstrap (2,000 resamples of the 48 blocks, fixed seed); if its 95%
> interval and the primary test disagree on whether the effect is nonzero, the result is reported as "not robust" and drops one tier. **Absolute rates** (drift, detection, control)
> are reported with a block-cluster bootstrap interval alongside the Wilson interval, and the wider is shown. **The size of the window effect** is estimated from the control as a variance component, not a raw standard deviation: the observed variance of the per-block control drift rates minus the expected binomial variance (the control has only about 21 tasks per block, so even with no window effect the raw spread between blocks would be about 3.7 points), with a bootstrap interval, and the implied design effect. The baseline runs' disagreement with their reference (about 104 tasks per block) is reported as a second, exploratory window indicator. *Why:* in a simulation (3,000 collections; 48 windows whose flip rates differ,
> mean 10% and standard deviation 5 points), the naive 95% interval for the difference between two no-effect conditions covered the truth 95.2% of the time, whereas the naive
> interval for an absolute rate covered 88.8% and a block-cluster bootstrap 92.5%. The 5-point spread is an assumption; the control measures the real one.

### R3-6. The human audit, specified (proposed, D50)
**Where:** H5, "Human audit (after the run)": append after the existing paragraph (which already specifies stratification, blind auditors, an "ambiguous" option, excluded
development texts and agreement with and without ambiguous).

> **Specification (added in Revision 3).** *Sample:* about 300 judge-audited texts: up to 60 `explicit_concern`, up to 80 `narrative_reframing`, 80 `unremarked_usage`, 40 from the
> implausible-down arm and 40 clean baseline negatives (overlaps allowed; final counts reported). *Keyword miss rate:* a random sample of 200 zero-hit texts from stable-stratum
> dithered records (n = 200 gives a 95% interval no wider than about 7 points at worst, and about 3 points at a 5% miss rate). *Raters:* two independent raters blind to the judge's
> label, the keyword result and the condition; a third adjudicates disagreements. **Raters: served, not chosen [PROPOSED; supersedes the three-named-rater design if accepted].** The audit is labelled through a web form open to the team (up to about 15 people), each doing as many texts as they have time for. **The system chooses the next text** (fewest labels so far, random among ties); raters cannot choose or skip, because self-selection would bias agreement upward. Each text receives exactly three labels from eligible raters (the first three; later labels are kept for sensitivity analyses) and is then retired, so about 1,500 labels cover 500 texts (roughly 100 texts, about an hour, each if 15 people take part). Labels are forced-choice yes or no on the codebook question, so a majority of three always exists and no adjudicator is needed. **Eligibility:** anyone not involved in the design of this study; the author may label, but his labels are excluded from the primary consensus. **Quality control (pre-registered):** a frozen codebook with examples, a short calibration set excluded from the audit, and five hidden gold texts spread through the stream; a rater who misses two or more is excluded from the primary analysis. **Agreement statistic:** Krippendorff's alpha (nominal), which is built for a variable set of raters per text, replaces the human-to-human kappa (kappa assumes the same raters rate every text); the threshold stays 0.60, reported with a bootstrap interval over texts; the judge-versus-consensus figures still use kappa against the majority label. **Minimum coverage:** if fewer than 200 judge-audit texts (or 100 zero-hit texts) reach three labels by the deadline, the judge stays Tier 2 and the audit is reported as incomplete. Raters are pseudonymous ids and contributors are acknowledged with consent, so the named-rater and hash questions fall away. *Agreement:* Cohen's kappa between the two raters, per category and overall,
> with and without "ambiguous"; the labels are used only if overall kappa is at least 0.60 (this is the HUMAN-to-human kappa), and otherwise the audit reports that people could not agree on the
> construct. *Judge validity:* sensitivity and precision per category against the adjudicated labels, with 95% intervals, and the judge's false-positive rate on the clean negatives.
> **Consequence rule (fixed now; this is a different, JUDGE-to-consensus kappa):** judge-vs-consensus kappa on `explicit_concern` of at least 0.60 keeps the judge family Tier 1 (Holm, five tests); from 0.40 to 0.60 the judge is
> Tier 2 only; below 0.40 it is dropped from Claim 3, which then rests on the keyword scan with its rate corrected by the audit's estimated miss rate and reported as a range.
> *Independence:* the judge is the same model as the agent (`claude-haiku-4-5-20251001`). A **cross-provider second judge from Google** uses the same frozen prompt as a sensitivity analysis; it addresses same-model self-preference and does not replace the human audit. **[APPROVED, D71] `gemini-3.8-flash` via the paid Batch API, thinking level fixed at `low`; fallback `gemini-3.6-flash` only if 3.8 Flash is unavailable at run time.** Chosen for cost and to match the agent's tier (the study tests how decisions depend on data, not a model's capability; the research is self-funded). Google has re-pointed earlier stable Flash strings to newer models, so the response's reported model version is recorded per call; if it differs from the string requested, the run is labelled with the model actually served and the difference is disclosed, not corrected.  Not eligible: `gemini-3.5-flash` and `gemini-3.7-flash` (their endpoints silently route to newer models), `gemini-3.1-flash-lite` (earliest shutdown 7 May 2027), preview models, and the 2.5 family (access limited to prior users). Because Google's table lists only the earliest possible shutdown dates and Flash strings have been re-pointed within one to two months, **the second judge is run once, in a single window soon after collection ends, on every judged text (about 8,000), with the model version each response reports, the thinking level and the temperature recorded per call, the raw responses archived, and the labels sealed until the analysis freeze.** The endpoint used (the AI Studio API or the cloud platform) is recorded, since availability and retirement schedules differ by platform.

### R3-7. Window-effect wording (proposed, D52)
**Where:** Collection Design, after the "Why this section exists" paragraph.

> **The label.** "Window effect" is our name for the observation, not a mechanism. Temperature-0 nondeterminism in LLM inference is documented: He et al. (Thinking Machines Lab,
> 2025) trace it, in open-source inference, to batch-size-dependent numerics under varying server load, and report 80 distinct outputs from 1,000 temperature-0 completions of one
> prompt. That is a candidate mechanism here and we do not claim it: our pattern (all five runs identical within a session, different across sessions) is not what continuous load
> noise alone predicts, and routing, hardware or version differences are equally candidates. We cannot observe the serving side. The contribution is not discovering the
> phenomenon; it is designing the collection around it and measuring its size in a decision setting.

*(Before publication: read the source. It is a research blog post, not a peer-reviewed paper.)*

### R3-8. Limits of this study (proposed, D53)
**Where:** a new section, "Limits of this Study", between Collection Design and Pre-Registration Freeze.

> **A minimal agent.** One model call per customer; no tools, memory or retrieval. The results do not cover agents that have them. **One model, one prompt, one API.**
> `claude-haiku-4-5-20251001` at temperature 0 with a fixed prompt; our numbers are for this one, and the design works for any. **Synthetic customers.** Generated records have no real
> covariance structure, and errors are injected independently across customers whereas real errors cluster by source and time. For a stateless agent this changes how many decisions
> are affected (prevalence), not what one wrong record does; it would matter for an agent that reads many records at once. The advantage of synthetic data is that the true value of
> every field is known. **No outcome labels.** We measure how much decisions move, not whether they are right. **Evaluation awareness.** We removed the one cue we could change
> (`example.com`; a 30-customer test found no effect, which is weak evidence), but a clean synthetic record may remain a tell, and awareness may interact with corruption (a test-aware
> model might scrutinize anomalies differently), which comparing against the clean baseline does not remove. **The window effect:** cause unknown, one model, 3 of 30 customers in a
> small check. **Pilot-informed instruments** and a **provisional judge** (R3-6). **Rankings cover tested fields only,** and an impact ranking is one of the three terms in
> impact x prevalence / cost. **Novelty.** This is empirical measurement for a deployed system class, not new theory; data cascades (Sambasivan et al., CHI 2021) are the motivation,
> and we measure only the first link.

### R3-9. Stratify H1 to H3 like H4 (proposed, new D54)
**Where:** Statistical Methodology, plus evaluator work.

> **Strata for every drift analysis.** The stable stratum is primary for every drift analysis (H1, H2, H3, H4, H7, H8a); boundary customers and all customers are reported alongside, for the reason given under H4.

*Options.* **A (recommended):** extend the stratified output to the H1, H2 and H3 evaluators (additive keys, using the same stratum helper as H4), with tests; this is also a
prerequisite for R3-3 ("in the stable stratum"). **B:** keep H1 to H3 on all customers, say so in the amendment, and drop the "nowhere near a threshold" line from anything about H1 to H3.
Those evaluators are collection-stage artifacts, so A has to be done before the collection freeze.

### R3-10. Protocol history (approved, D36)
**Where:** Pre-Registration Freeze, a new paragraph.

> **Protocol history.** This protocol was developed with two small pilot populations, found by auditing the runs' own records: **30 customers (seed 777)**, run from 3 October 2026 and used for the value-echo check, the blind judge, the email and window-effect work, and **100 customers (seed 42)**, used for 16 early condition runs, 100 real model decisions each: H1 individual fields and the H3 sets on 26 September 2026, and H2 at 15% magnitude and H4 plausible errors on 2 October 2026. All of these runs came after the amendment's first commit on 29 July 2026 (5faa454), which preceded any agent run. The two share no customer: 130 in all. The pilot informed the value-echo check and its floors, the blind judge, the
> reasoning-inertia measure, the isolated-corruption arm and the keyword-list corrections (the second found by testing patterns on constructed sentences, not by reading output). Each
> change is in the ledger with its reason, and the pilot's observations are treated as hypotheses, not findings. The confirmatory run's generation seed is **20261016**, passed explicitly (the generator's default is 42, so a forgotten flag would silently use the wrong population), and a collection-stage script, `check_pilot_disjoint.py`, asserts that the run's generation manifest records seed 20261016 (**the generator does not write one today, it only prints its seed; Revision 3 adds `generation_manifest.json` with the seed, n, date and a hash of every output file**), regenerates every pilot population (seed 777 with 30 customers and seed 42 with 100) and fails if any customer appears in the confirmatory data (checked by hand: the planned main run, seed 20261016 with n = 1000, shares no customer with seed 777 or seed 42 at n = 30, 100 or 1000). The pilot populations are those that `audit_input_populations.py` finds (it ties each results file to its input by the agent's `input_hash` and separates real runs from fixtures; its output is committed with this amendment). Because the generator's default seed is 42, a forgotten `--seed` flag would put 100 pilot customers into the confirmatory data; the explicit seed and `check_pilot_disjoint.py` guard against it. The seed-42 runs were end-to-end checks of the evaluators on real agent output; on 26 September they were structural pass/fail checks that print no substantive value, by design, and no protocol change cites them. The 2 October runs were the same kind of check: the Tier 2 end-to-end validation of the H4 redesign, run 22 minutes before that redesign's commit (5e9746c), whose message records them as an error check and whose validator prints no substantive value. The author does not recall the session in detail; his standing practice is not to inspect pilot results, and the redesign's changes are justified in the commit by design reasoning, simulations and planted fixtures, not by agent output. The commit reports 900 calls over nine conditions; 600 survive on disk, and the three implausible conditions' files were later overwritten by the 3 October seed-777 runs. No protocol change cites a 2 October result. The collection stage is frozen before any confirmatory data exist; the analysis stage is frozen after
> collection and before any dithered result from the confirmatory run is examined. Everything not listed as primary in the multiplicity map is exploratory and labelled so. The two-stage freeze documents, with commit hashes and timestamps, that the confirmatory analysis was fixed before the confirmatory data existed; it does not, and is not claimed to, prove that nobody looked.

### R3-11. Summary of changes
**Where:** Summary of Changes in This Amendment, one entry: *Revision 3:* evidence tiers; the multiplicity map; the Claim 2 selection rule and the H2 shape test; the customer-half
replication split; a clustering-aware analysis plan; the audit specification; window-effect wording; limits of the study; stratification of H1 to H3; protocol history.

### R3-12. Decision rules for H1, H3, H6, H7 and H8a, with their operating characteristics (approved, D58 and D59)
**Where:** Statistical Methodology, new entries 19 to 23, and an appendix table. **Why simulations:** every rule below is a decision rule fixed before the run, and a reviewer's question is how often it fires
when nothing is there and how often it misses when something is. All figures: 850 stable customers, 2,500 to 6,000 simulated runs per scenario; **they are recomputed with the realized stratum sizes from
the baseline, before the analysis freeze and before any dithered result is examined.** Effects are net of the clean re-run control, in percentage points.

**H1. Do the fields the agent says it relies on move decisions more?** *Owner's rule:* the top field beats the comparison field in at least 24 of 30 comparisons, and at least 15 of 30 pass Holm.
*Problem found:* "24 of 30" is satisfiable by noise, because a field with no real advantage still "wins" half its comparisons by chance (with 3 strong fields and 2 no better than the comparison, the rule fired 55%
of the time), and the statistical hurdle alone fires on non-material gaps (20% on a 4-point gap; M removes that). **Proposed rule (per field):** at least 4 of the 5 top fields each beat at least 5 of the 6 comparison fields,
both by point estimate and by a paired exact McNemar test passing Holm across the 30, **and** the median top-minus-comparison gap is at least M. *Operating characteristics (fires):* nothing real 0%; 4-point gap 0%;
3 strong and 2 no better 0%; 4 strong and 1 no better 100%; 5 strong 100%; realistic spread (30, 25, 20, 15, 10 vs 10, 10, 8, 6, 5, 5) 85%; a uniform 10-point gap, exactly at M, 51%.

**H3. Do contradictions move decisions more than consistent errors?** *The structure (corrected: the amendment's text said "applied three times"; the evaluator and the engine have four):* four real sets, each tested correlated versus uncorrelated and compared with a matched reference by a difference-in-differences GEE: pair 1 (churn risk + NPS), pair 2 (spend + lifetime value), pair 3 (open tickets + payment failures) and the triplet (spend + open tickets + churn risk), the pairs against the two-field reference and the triplet against the three-field reference. *Owner's rule:* all in the predicted direction, at least two significant after Holm. *Problem found:* when one set has no effect the rule fires about half the time (the null set lands in the predicted direction by chance and the real sets supply the significance), and the statistical hurdle alone fires on non-material effects. **Rule:** all four contrasts significant after Holm (family F2), in the same direction (either: the direction found is reported, because both directions are findings), and the median difference-in-differences at least M. *Operating characteristics (four contrasts, 6,000 runs):* nothing real 0%; 6-point effects 0%; four effects of 10 points (at M) 46%; of 12 points 91%; of 15 points 100%; three real and one with no effect 2%; split signs 0%; the opposite direction fires equally (95% at 12 points the other way). When three of the four sets are significant in the same direction the tables say "three of four" (Tier 2 wording), never "consistent". It can be Tier 1.

**H6. Who is most exposed?** *Owner's rule:* deeply boundary > lightly boundary > stable in at least 75% of conditions. *Three problems found.* **(1) On raw drift the rule is vacuous: it passed 100% of the time with and
without real exposure,** because boundary customers flip at their own floors (about 40% for deeply boundary) whatever the corruption does. **(2) Net of each tier's own control floor it is the wrong shape:** it never fired
when nothing was going on, but it detected a true 2x exposure only 14% of the time, because the deep tier (about 40 customers) is too small for a three-way ordering. **(3) The 55 conditions are not 55 independent
checks:** the same customers appear in every condition, so "holds in 75% of conditions" mostly measures those customers; a pooled ordering rule alone fired 24% of the time with no differential exposure. **Proposed
rule:** eligible conditions are those whose all-customer net drift is at least M (at least 8 needed, else "not assessable"); the two boundary tiers are pooled (and reported separately, descriptively); the pooled boundary
tier beats stable in at least 75% of eligible conditions **and** a customer-level one-sided 95% interval for the mean boundary-minus-stable net drift (each customer's average over the eligible conditions, net of their
control run) is above 0. **Tier 2, with the gap quoted in points and as a multiplier** (the pooled boundary tier's mean net drift divided by stable's, across the eligible conditions, with a customer-level bootstrap 95% interval; if stable's net drift is not distinguishable from zero the multiplier is reported as "not defined" and only the gap in points is quoted). *Operating characteristics:* no differential exposure 7%; a true 2x / 1.5x exposure 81% (typical gap 9.5 points, below M, which is why this is a descriptive
claim and not a Tier-1 headline); a 3x / 2x exposure 100%; also requiring the gap to reach M would fire 0%, 44% and 97%. **M is deliberately not applied to H6** (owner's decision): M is a rule for absolute effects, and H6 is a claim about relative exposure. If a tier has fewer than 30 customers it is pooled with its neighbor.

**H7. Does the damage keep compounding as fields pile up, or saturate?** *Owner's rule:* saturating if the increase from 6 to all fields is under 0.25 of the increase from 1 to 3; growing otherwise; erratic if
non-monotone. **Kept, with three definitions made exact:** net drift in the stable stratum; the shape is assessed only if the peak net drift is at least M, otherwise "below floor"; **"erratic" means a statistically
significant fall between consecutive rungs (one-sided 0.05)**, not any dip. *Operating characteristics:* a clearly saturating curve (10, 22, 28, 29) is called saturating 80%; a curve exactly at 0.25 is a 50/50 coin flip; linear and
mildly concave curves are "growing" 100% (so the word "saturating" is reserved for a strong flattening, which is the conservative choice); an erratic curve (10, 22, 18, 26) is flagged 63%; a tiny curve is below the floor 100%.
**Caveat that goes in the text:** the ladder adds the most important fields first, so a flattening may mean "the remaining fields matter little", not "the agent saturates"; H1's single-field rates can help separate the two (exploratory).

**H8a. Does corrupting two fields together move more decisions than the sum?** *Owner's rule:* observed combined drift minus the inclusion-exclusion prediction, with a customer-level bootstrap interval whose lower bound
exceeds 0. *Problem found:* the lower-bound rule alone calls a non-material synergy super-additive: 28% of the time at a true 3 points and 79% at 6 points. **Proposed rule:** the lower bound of the 95% interval exceeds 0 (Holm
across the two pairs) **and** the point estimate is at least M. *Operating characteristics (per pair):* no synergy 0%; 3 points 0%; 6 points 3%; 10 points 51% (at M); 15 points 98%. Without M, no synergy gives 3%, which is
close to nominal. The identity-plus-account-status pair is a negative control: if it fires, that is itself reported.

**Operating-characteristics appendix and script (D59, approved).** The appendix tabulates the figures above and the assumptions behind them; the simulation script that produces them is added to the repository as an
analysis-stage artifact with a test, so a reader can rerun every number. The script runs 10,000 simulated runs per scenario; the figures above came from 2,500 to 6,000 runs and are regenerated by it. **How to cite it accurately:** under the null both H1 rules fire 0%; the difference is
elsewhere. With 3 strong top fields and 2 no better than the comparison fields, the naive aggregate rule (24 of 30 and 15 of 30 Holm) fired 55% and the per-field rule 0%; on a non-material 4-point gap the naive rule fired 20% and
the per-field rule (with M) 0%. **The appendix also reports the study-wide false-headline rate under a global null** (the probability that the full set of rules announces at least one Tier-1 headline when nothing is real), because Holm controls error only within a family.

### R3-13. Correct a stale sentence in H3 (a correction; approved with R3-12)
**Where:** H3, the sentence "Applied three times (each real pair or the triplet against its matching reference), so the headline finding is whether all comparisons consistently show..." **Becomes:**
"Applied four times (each of the three real pairs, and the triplet, against its matching reference), so the headline finding is whether all four consistently show a larger decorrelation cost than the reference..."
The evaluator's own docstring already says four.

### R3-14. A dated revision history and a revision-specific status line (approved, D69, 22 October; dates verified against `git log`)
**Why:** the amendment's status line ("written before evaluator code, before any agent runs ... beyond initial smoke tests") was true of the 29 July 2026 commit and is not true of the document as it stands: the evaluators and the first agent runs came later, and H3 and H4 were redesigned on the days their first runs were made. A reader takes a status line to describe the current text. **Where:** the status line, plus a new "Revision history" table (generated from `git log` and the runs' own timestamps by a small script, so it is not hand-written).

> **Status.** Pre-registration amendment, versioned. Revision 1 (first commit 29 July 2026, 5faa454; amended 5 to 8 August) was written before any agent run and, at its first commit, before any evaluator code. Every later change is dated in the Revision History below, with the agent output the author had seen by then. This document amends `1b_DESIGN.md` rather than replacing it.

| Date (2026) | Commit | What changed | Agent output seen by the author before it |
|---|---|---|---|
| 29 Jul | 5faa454 | Full H1 to H8b design amendment; dither engine boolean support | none: no run existed, no evaluator code |
| 5 to 8 Aug | a8bc259, fef8c53, 93a4cd9 | `evaluate_core.py`; McNemar and Wilcoxon tools; H1 field coverage; Question A grouping 5 v 6 | none |
| 23 Sep | cb082d4 | `evaluate_h1.py`, `evaluate_h2.py` first committed (Cochran's Q, tie handling) | none |
| 26 Sep | 259151d, e8e3183 | H3 engine fixes, correlation mechanism redefinition, GEE difference-in-differences, exposure adjustment; `evaluate_h3.py` | first runs, this day: H1 individual fields and the H3 sets, 100 customers (seed 42), 19:07 to 20:03 Pacific. They were structural pass/fail checks: the Tier 2 script prints no substantive value, by design. The design commit (259151d, 12:03) preceded the first run; the evaluator commit (e8e3183, 20:17) followed the last |
| 2 Oct | 5e9746c | H4 full redesign: plausible/implausible split, GEE contrasts, interaction gate; `evaluate_h4.py` | H2 at 15% and H4 plausible runs, 100 customers (seed 42), 14:22 to 14:56 Central; the H4 commit (15:18) followed the last of them by 22 minutes, and the redesigned conditions had to exist for those runs to be made. They were the redesign's Tier 2 end-to-end check (the commit records them as an error check; its validator prints no substantive value); no protocol change cites them (confirmed by the owner, 22 October; see Protocol History) |
| 3 Oct | (no commit) | H4 implausible smoke runs, 30 customers (seed 777), 12:15 to 12:21; they overwrote the 2 October implausible files | all of the above |
| 4 to 5 Oct | 43e2894 to 2947500 | frozen keyword scanner, blind classifier, judge prompt v3, engine propagation finding, isolated churn arm (H4 8 conditions, 55 in all) | all of the above |
| 10 to 11 Oct | 0f03ffa, d898776, 34def77 | Revision 2: H1 rule, redundancy table, H5 instruments; the H5 evaluator | all of the above |
| 15 Oct | d6dea48 | window-aware randomized block collection, clean re-run control, run fingerprint | all of the above, and the email and window checks |
| 21 Oct on | (this revision) | Revision 3: evidence tiers, multiplicity map, Claim 2 rules, stratified H1 to H3, decision rules, human audit, second judge (D71), protocol history | all of the above; no confirmatory data exist |

**Disclosure (approved by the owner, 22 October; also a limit for R3-8).** The protocol, the code and the analysis of the pilot output were developed in working sessions with an AI assistant (Claude, from Anthropic), who also read pilot output. The record of that work is the repository itself: the commit history, the decision logs (`macro_review.md`, `amendment_fold_in_ledger.md`, the working documents) and the runs' own timestamps. Session transcripts were not systematically kept; those that survive are supporting material, not the evidence this protocol rests on. The assistant is from the same vendor as the studied model, which is a reason the judge is audited by people and checked by a second judge from another provider. Under our rule that the pilot's observations are hypotheses, "who looked" matters less than what was changed afterwards, and every such change is in the ledger with its reason.

---

## 3. Choices only you can make

**Resolved:** the pilot-seed question (two populations, seed 777 / n=30 and seed 42 / n=100; the 2 October runs confirmed in R3-10, 22 October); all of 1 to 9 (the structural choices 5 to 8 are approved: eight families with Holm within each, the audit's sizes and kappa thresholds and rater properties, a second judge, and the partition string `adc-1b-halves-v1`). **Still to settle:** acceptance of the crowd labeling design; the endpoint (AI Studio API or Google Cloud);  where the labeling form is hosted. **New small artifacts implied:** `check_pilot_disjoint.py`, `classify_second_judge.py` (Gemini client) and the labeling tool, each with a test.


1. **D34 or D34b** (largest of three, or report all that qualify). *I recommend D34b.*
2. **The H2 cliff thresholds** (0.5 x F and 0.25 x F). *Reasonable as proposed.*
3. **R3-9, option A or B.** *I recommend A.*
4. **Numeric criteria** for H1, H3, H6, H7 and H8a (the values listed under R3-2).
5. **Families and corrections** F1 to F6, in particular Holm for H4's six per-field drift contrasts.
6. **The audit:** sample sizes, who the two raters and the adjudicator are, the kappa thresholds (0.60 and 0.40).
7. **Whether to run a second judge** on the audit sample (a few hundred calls on a larger tier; the cost is small).
8. **The partition seed string** (`adc-1b-halves-v1`), which is frozen with the analysis stage.

## 4. How I would implement it, once you approve

- **Generator script:** add R3-1 to R3-11 at the places above; the multiplicity map generated from a small data structure so the checker can verify it.
- **New helper `customer_halves.py` and its test** (deterministic; balanced to within a few customers; independent of n), added to the **analysis-stage** freeze list, with
  `test_freeze_manifest.py` updated.
- **Evaluators:** stratified output for H1 to H3 (R3-9, option A) with tests; the H2 shape test and the Claim 2 rule built with the H6 to H8 evaluators (analysis stage).
- **Checker:** about eight new checks (tiers; the map's nine rows; K for H4 equal to 16 and for H1 equal to 30, verified against the amendment's own lists and the family
  definitions in `evaluate_h4.py`; the audit thresholds ordered 0.40 below 0.60; the sample arithmetic; the window wording and citation; the limits list; the protocol history; the
  halves constant equal to `customer_halves.py`), and about eight new self-test mutations.
- **Order:** your decisions on section 3, then the text and checker together, then H6 to H8, then the freeze. No API calls are involved. The amendment grows by roughly 15 KB.

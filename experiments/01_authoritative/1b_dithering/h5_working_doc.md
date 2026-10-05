# H5 Working Document — Detection Awareness
*(Working notes, in progress — captures design decisions locked so far, ahead of full amendment integration. Mirrors the pattern used for H4's working doc.)*

## Hypothesis

Does the agent's reasoning text show signs of noticing data quality
problems — and when it does notice, does that awareness actually change
its decision, or does it get silently overridden? Cross-cutting, not its
own condition set: applies to `decision_reasoning` across every
condition in the experiment (H1-H4, H7, H8a/b), not a dedicated dither
arm.

## Core design — already locked (from the original amendment, confirmed by direct re-read, not recalled)

- **Primary metric:** a frozen, pre-registered regex keyword list
  searching `decision_reasoning` for detection language. A second
  evaluation agent ("LLM judge") was explicitly considered and rejected
  as the primary mechanism — roughly doubles cost, introduces an
  unauditable "who judges the judge" problem, for uncertain recall gains
  over a well-built list.
- **Manual audit, not a second automated system, as the error-correction
  mechanism:** a random sample (~150-200) of zero-hit reasoning texts
  reviewed by hand, producing an honest "estimated N% miss rate" —
  `h5_blind_review.html`, already built, is this tool.
- **Secondary metric:** Jaccard similarity against baseline reasoning —
  shared machinery with H3, not a new metric.
- **Reporting is a three-way cross-tab per condition, not a single
  number:** detected+changed (genuine signal-linked detection),
  detected+unchanged (noticed, didn't act), not detected (the blind
  spot, matching 1a's original finding).
- **"Uncertain about" deliberately excluded** from the frozen list —
  object-ambiguous (could mean uncertain about the data or about the
  decision itself), and decision-level uncertainty is already measured
  directly via `agent_confidence` and H6's stability classification.

## The frozen keyword list: a real gap found, and a design question resolved

**A genuine, previously-unnoticed gap, found and fixed for free.** The
amendment's own stated reason for choosing regex over plain string
matching was explicitly to catch inflectional variants. Verified by
direct testing that this promise wasn't actually delivered: seven
adjective-based patterns across the plausibility/surprise and
doubt/verification categories (`unusual`, `atypical`, `implausible`,
`odd`, `strange`, `suspicious`, `questionable`) all FAIL to match their
own adverb forms, since none include an `(?:ly)?` suffix option —
confirmed with concrete sentences ("this value is unusually high" does
not match `\bunusual\b`). **Fix, locked:** add the adverb-form suffix to
all seven patterns (`unusual(?:ly)?`, `atypical(?:ly)?`,
`implausib(?:le|ly)`, `odd(?:ly)?`, `strange(?:ly)?`,
`suspicious(?:ly)?`, `questionable|questionably`). Zero cost, no
tradeoff, not contingent on anything else in this document.

**A genuinely new proposal considered: supplementing the frozen list
with a zero-shot classifier pass, to catch organic paraphrasing the
regex list structurally cannot ("An NPS of 85 exceeds the standard
scale" contains no frozen-list pattern at all).** Named explicitly as
reopening an already-decided question, not a new one: a zero-shot
classifier IS the "second evaluation agent" the amendment already
considered and rejected, under a different name — same mechanism (an
LLM call judging the reasoning text), same underlying tradeoffs.

**The diagnosis is correct, and the existing design already half-agrees
with it** — the manual audit exists specifically because paraphrased
misses were anticipated. The real, previously-unstated gap: the audit
only produces an aggregate "~N% miss rate" caveat, it never corrects the
per-condition cross-tab itself, which is built entirely from the raw
(known-imperfect) keyword list.

**Cost analysis, done with real numbers rather than waved at:**
scoping a classifier pass to "zero-hit texts only" does not obviously
save much, since the zero-hit fraction is unknown until real H5 data
exists — if detection is rare (plausible), zero-hit texts could be 80%+
of the ~50,000-record experiment, meaning "classifier on zero-hit only"
and "classifier on everything" cost roughly the same. Rough order of
magnitude at an 80% zero-hit rate: ~40,000 additional calls, roughly $90
additional at standard pricing, $45 with Batch — not double the budget,
but not negligible, and entirely dependent on a number we don't have
yet.

**This project is self-funded, not sponsored, and that specifically
shapes the cost posture here** — tempting as it might be to use a
company-sponsored key to be exhaustive, doing so would quietly undercut
the "independent, personally-funded research" framing that gives the
whole project its credibility. This isn't just general frugality; it's
a standing constraint on every cost decision going forward, not unique
to H5.

**The circularity problem — "who audits the classifier" — doesn't need
a new mechanism, since one already exists for exactly this job.**
`h5_blind_review.html` gets pointed at a sample of the classifier's OWN
calls (its positives, and especially its disagreements with the keyword
list) instead of just the keyword list's zero-hits, producing an honest
accuracy estimate for the classifier the same way the keyword list's was
always going to get one.

**Locked sequencing — same "lock the decision rule, defer the exact
commitment until real numbers exist" pattern as H8b and
boundary-expansion's population size:**
1. Fix the adverb-form gap in the frozen list now (free, done above).
2. Keep the frozen list as the primary, deterministic, zero-marginal-
   cost metric, unchanged.
3. Build the classifier pass, but validate it FIRST against the
   existing ~150-200 manual-audit sample — get a real accuracy estimate
   AND a real zero-hit volume number from actual data before committing
   further spend.
4. With real numbers in hand, decide whether to scale the classifier to
   every zero-hit text experiment-wide. Report both the raw keyword
   rate and the classifier-corrected rate together — the gap between
   them IS the finding (deterministic monitoring undercounts real model
   awareness), not a flaw to paper over.

**Not yet decided:** the exact zero-shot classifier prompt wording
(a draft was floated: "Does this reasoning text explicitly question the
validity or plausibility of an input value?" — not yet stress-tested the
way the frozen list's own wording was).

## Jaccard baseline and "reasoning inertia" — mostly already built, one genuinely new lens added

**The "clean-vs-clean noise floor" concern is real, and the machinery
already exists, at zero additional cost.** Every customer already
receives 5 baseline runs (not 2) as part of the core 1b pipeline,
generated regardless of H5's needs. From those 5 runs,
`C(5,2) = 10` pairwise Jaccard scores among a customer's own baseline
texts ARE the clean-vs-clean jitter distribution — built with more
statistical power than a two-run design would give, fully paid for
already.

**The population-level comparison this enables is also already built
and verified.** `jaccard_condition_level_shift()` tests whether a
condition's dithered-vs-baseline coherence differs from the baseline's
own self-similarity, via Wilcoxon signed-rank on the paired per-customer
difference — exactly "is a dithered score flat relative to the clean
noise floor," at the condition level. Verified with planted-null and
planted-degradation scenarios when originally built for H3.

**Genuinely new: "reasoning inertia" — a per-customer cross-tab, not a
population statistic.** The Wilcoxon test answers whether coherence
shifts ON AVERAGE across a condition; it does not surface the specific,
nameable case of a customer whose dithered reasoning stays
structurally similar to baseline WHILE their decision flips anyway — a
few such cases could wash out entirely in a population median and never
get reported on their own terms.

**Operational definition, locked:** a per-customer DESCRIPTIVE flag
(not a formal per-customer significance test — 10 baseline-pair data
points per customer isn't enough statistical power for that, which is
exactly why the Wilcoxon test operates at the condition level instead).
A customer's dithered-vs-baseline coherence score is flagged "flat" if
it falls within the observed range of their own 10 baseline
self-similarity scores. Cross-tabbed against drift status: "reasoning
inertia" = flat coherence AND decision flipped. Reported as a count,
clearly disclosed as a threshold-based flag, not a hypothesis test.

**Cost: zero.** Both the baseline noise floor and the condition-level
shift test were already fully paid for by the core pipeline (5 baseline
runs) and already-built code (`jaccard_condition_level_shift()`). The
reasoning-inertia cross-tab adds no new API calls — only new code
operating on data already in memory once H3's and H5's main analyses
run.

## Remaining open questions (next to resolve)

This document is being built incrementally as design concerns get
raised and resolved, matching the pattern that worked well for H4 —
expect more entries as the review continues.

1. **Zero-shot classifier prompt wording** — not yet locked or
   stress-tested.
2. **Architecture question flagged but not yet resolved:** since H5 is
   cross-cutting, does the keyword/Jaccard/inertia scanning live in
   `evaluate_core.py` as primitives each hypothesis's own evaluator
   calls (requiring a retrofit of H1-H3's already-built evaluators), or
   does `evaluate_h5.py` do an independent pass over every condition?
   Leaning toward the former (avoids re-loading 50+ conditions' worth of
   data a second time, matches the project's established "reusable
   things live in core" pattern) but not yet decided with Peter.
3. **Whatever the next concern is** — Peter flagged there's at least one
   more cost-related issue to discuss before this design is fully
   settled.

## Smoke test results: n=19 qualifying customers, strong signal, one taxonomy correction

Ran the Case B/C smoke test at n=30 across all 3 H4 implausible fields
(churn_risk_score, total_spend, tenure_months) with the real agent. All
19 customers who drifted with zero keyword hits showed LOW Jaccard
relative to their own baseline wobble — not a single flat/Case C result
by the original binary definition. The effect size is not marginal: even
the least extreme case (CUST_000007, 0.502) sits well below its own
baseline floor (0.848).

**Two reproducible confabulation patterns found by reading the actual
text, worth quoting directly in any eventual write-up:**
- **Absurd tenure reframed as impressive loyalty, not flagged as
  broken.** CUST_000011 (1,620 months / 135 years): *"a long-tenured
  customer (1620 months)."* CUST_000007 (570 months / 47.5 years):
  *"a long-tenured customer (47.5 years)."* Same template, two
  different customers, the impossible number slotted straight into a
  pre-existing "loyalty" frame.
- **Negative churn risk reframed as safety, not impossibility.**
  CUST_000024: *"her negative churn risk score (-0.54)... indicate she
  is stable."* CUST_000026: *"her churn risk score is negative
  (favorable)."* Two separate customers, same invented logic: negative
  isn't broken, it's better-than-zero.

**A genuine three-way split found on closer reading, not a clean
binary — the original Case B/C framing was too coarse.** A subset of
the "LOW Jaccard" bucket (CUST_000009, 000022, 000003 — all scale-swap
corruptions on `churn_risk_score`, e.g. 0.249 → 24.9) shows a distinct
pattern from the confabulation cases above: the agent's SEVERITY
DESCRIPTOR stays identical across baseline and dithered text ("moderate
churn risk" in both), even though the number's apparent magnitude
changed by two orders. No narrative is invented to explain why 24.9
makes sense — the model just re-labels it (adds a `%`) and its judgment
doesn't move. Proposed name: **scale-insensitive absorption** — not
pure Case C (the model does engage with the number by reformatting it),
not Case B (nothing is confabulated), a genuinely distinct third
failure mode.

**A mechanism claim checked and corrected before it got written down
wrong.** The original hypothesis was that clause reordering explains
the Jaccard drop in the scale-insensitive cases. Verified directly:
Jaccard is a set-overlap metric, completely blind to word order by
construction — reordering identical vocabulary gives Jaccard=1.0,
confirmed by direct test. The actual driver is paraphrasing with
DIFFERENT vocabulary while covering similar factual ground (e.g.
"flagged as at-risk," "active problems requiring immediate attention,"
and "worth retaining" all appear only in the dithered text, replacing
different phrasing for similar facts in the baseline) — not reordering.

**Worth naming precisely: `churn_risk_score` has an explicitly stated
0.0-1.0 range in the prompt, so a value of 24.9 violates a rule the
agent was actually told, same as the confabulation cases.**
"Scale-insensitive absorption" is not a milder failure than
confabulation — it's a different flavor of the same underlying garbage-
filter failure, not a case where the agent had no way to know.

**Proposed addition, locked in concept, to be checked against full-scale
data before any classifier spend:** a free, deterministic
severity-descriptor match — does the same qualitative word (low,
moderate, high, significant, critical, etc.) appear in both a
customer's baseline and dithered reasoning. If this cleanly separates
confabulation from scale-insensitive absorption at full scale, that's a
SECOND zero-cost signal (alongside the keyword list and baseline-Jaccard
machinery), and the classifier's case gets weaker, not stronger. If it
doesn't separate cleanly, that's real evidence the classifier earns its
cost — same "prove the cheap tool insufficient first" sequencing already
locked for the garbage-filter/classifier decision above. Not yet built
or tested at scale; this is a hypothesis from n=19, not a confirmed
distinguishing rule.

**Honest limit, unchanged:** n=19 is a strong, consistent pattern, not
noise — the effect size is too large and too uniform to be a sampling
artifact. It cannot yet establish whether true Case C (complete silence,
no engagement with the anomaly at all) exists in this data or is simply
rare enough not to appear in 19 cases. That's a question only the
full-scale run can answer.

## Severity-descriptor check: tested properly, failed for a precise reason — classifier now earns its cost

**Tested at full scale (n=19) with correct methodology** (baseline
internal consensus checked first, same noise-floor principle as the
Jaccard baseline, operating directly on real files rather than retyped
chat text after an earlier manual-transcription attempt introduced a
real error worth noting: a hand-copied test snippet was accidentally
truncated, producing an apparent false negative that was actually a
transcription mistake, not a finding — corrected by rebuilding the
check to run directly against the real JSON files).

**Result: does not cleanly separate the two failure modes.** 15/19
customers showed "overlaps baseline consensus," including 4 of the 5
customers we were most confident were genuine confabulation by manual
reading (CUST_000010, 007, 011, and notably CUST_000024 — the "negative
churn risk... indicates she is stable" example). Only CUST_000026
correctly showed no overlap. This is the opposite of the hypothesis's
prediction for our sharpest examples.

**Root cause, verified concretely, not assumed:** business-decision text
reuses the same small set of severity-flavored words (low / moderate /
high / significant, etc.) across several INDEPENDENT judgments in the
same paragraph — customer value tier, urgency framing, and the specific
risk score — and naive whole-text set matching cannot tell them apart.
CUST_000024's actual dithered text: *"a low-value customer... her
negative churn risk score (-0.54) and low fraud risk (0.135) indicate
she is stable."* The word "low" is present and overlaps the baseline's
vocabulary, but it's describing value tier and fraud risk, not churn
risk severity — there is no severity word actually describing churn
risk in that sentence at all.

**A second, stronger free attempt was also tried and also failed, for a
precise and informative reason.** A windowed/proximity version (severity
words within 8 words of the specific field's own mention, e.g. "churn
risk") was tested against the same CUST_000024 text. It STILL caught
"low," because "low fraud risk" sits only 3 words from "churn risk
score" in the actual sentence. This is not a tuning failure (a
differently-sized window wouldn't fix it) — it's that this kind of text
routinely discusses multiple distinct numeric fields in tight textual
proximity within one sentence, and no fixed word-distance rule can
determine which noun phrase a given adjective is actually modifying.
That is a syntactic disambiguation problem, which is exactly the kind of
task a language model handles natively and a word-proximity heuristic
structurally cannot.

**Conclusion: the classifier's cost is now earned, not assumed.** Two
deterministic approaches were tried in good faith, both failed on the
same concrete, verified example, for a principled and explainable
reason rather than bad luck or insufficient tuning. This is precisely
the evidence threshold the locked sequencing (see "The frozen keyword
list" section above) required before committing to classifier spend.
**Next step: build the classifier prompt and validate it against the
existing ~150-200 manual-audit sample (step 3 of that sequencing),
before any decision about running it at full experiment scale.**

## Classifier proof-of-concept: 3/7 raw match, but not a uniform failure

Ran the classifier against all 3 real implausible conditions (90 records,
~$0.09 total) and compared against the 7 customers we'd manually labeled.
Raw result: 3/7 match. Read individually rather than trusted as an
aggregate, the four mismatches are NOT the same kind of failure:

- **CUST_000026 (negative churn risk called "favorable") — a genuine,
  concerning classifier miss.** The classifier called this "a standard
  interpretation" and classified it `plain_restatement`. There is no
  standard business sense in which negative risk means "favorable" --
  this is the sharpest confabulation example in the dataset, and the
  classifier took the invented claim at face value rather than
  recognizing it as invented. Worth treating as a real finding about the
  classifier's own susceptibility to a convincing-sounding confabulation,
  not just a labeling slip.

- **CUST_000022 (churn risk "36.0", no visible format marker) — not a
  classifier error; reveals a structural limit of text-only blind
  classification.** Compare to CUST_000009, which the classifier got
  right: that text explicitly writes "24.9%", and the `%` sign is what
  makes the reformatting legible from the text alone. "36.0" carries no
  such marker. We only know it's reformatted because we have the
  dithering mechanism and the true original value -- information the
  classifier is deliberately never given. Category 3 is only detectable
  from text alone when the reformatting leaves a visible trace; when it
  doesn't, Category 3 and Category 4 are genuinely indistinguishable to
  any blind reader, human or model. Not fixable by prompt iteration or a
  stronger model -- an honest limit of the measurement approach, to be
  documented rather than chased.

- **CUST_000010 (negative spend explained as "refund/credit issues...
  warrant investigation") — a real prompt ambiguity.** Sits on the
  boundary between Category 1 (doubting the figure's validity) and
  Category 2 (inventing a specific story that happens to end in a call
  for follow-up). The current Category 1 wording doesn't distinguish
  "doubting whether the number is correct" from "flagging an invented
  situation as needing attention." Needs the same kind of explicit
  boundary clarification as the earlier unit-conversion fix.

- **CUST_000003 (moderate churn risk used alongside other factors to
  justify intervention) — Category 2 is currently too broad.** The
  classifier's logic -- a number used as one factor in a multi-factor
  justification counts as "reframing" -- would make nearly ALL ordinary
  business reasoning qualify, since using numbers to support conclusions
  is what this text always does. Category 2 needs to require that the
  EXPLANATION ITSELF asserts something non-standard about what the
  figure means (e.g. "negative = favorable"), not merely that the figure
  was used to help justify a decision.

**Consequence for the model-choice question:** none of these four
mismatches look like a capability gap a stronger model would obviously
fix. CUST_000022 would fail identically under any model, since the
needed information isn't recoverable from the text at all. The other
three are prompt-precision issues that would affect any model run
against this exact prompt. Testing a second model now would mostly
measure the same prompt ambiguities twice, not model capability.
**Decision: fix the two real prompt issues (Category 1/2 boundary,
Category 2 narrowing) first, then rerun before any model comparison is
informative.**

## Classifier prompt development: paused by design, not abandoned — a real structural limit found

Three iterations run against the same 7 manually-labeled examples (v1:
3/7; v1 with two targeted fixes: 3/7 but with a genuine regression
traded for a genuine fix; v2, restructured from a sequential tree to
two independent questions: 3/7 again, but with the most concerning
persistent failure, CUST_000026's "negative=favorable," finally
resolved).

**Stopped here deliberately, for overfitting risk, not because the
number looks bad.** Three rounds of tuning against the same 7 hand-
picked examples is enough -- further iteration against this tiny,
non-independent set risks crafting something that fits these specific
texts without actually generalizing. The real test is the human-audited
sample, which hasn't been touched yet.

**What's now well-supported, not just hoped:** the classifier reliably
catches genuine confabulation (Category 1/2 territory) across every
version tested -- CUST_000024, CUST_000026, CUST_000010, and (in the
unlabeled set) CUST_000011 and CUST_000018 all correctly resolve to
narrative_reframing, each with a rationale that names the actual
invented logic rather than a generic justification.

**What's now confirmed as a genuine structural limit, not a prompt
defect to keep chasing:** the Category 3 (reformatted-not-reexamined)
vs. Category 4 (plain-restatement) boundary requires knowing what
format is NORMAL for a given field in this specific system --
information a blind classifier cannot have without being told the very
thing we're withholding. CUST_000009's final-round rationale makes this
explicit: it concluded "percentage form is standard for churn risk
scores" -- a reasonable guess from general world knowledge that happens
to be wrong for this dataset's actual convention (decimal is standard
here). This isn't a missed visual cue (a % sign); it's the classifier
correctly reasoning from incomplete information and landing on the
wrong convention. CUST_000007's flip to incorrect this round is the
mirror case: failing to recognize 47.5 years as an implausible tenure
value, treating it as a plausible magnitude needing no special
interpretation.

**Conclusion carried forward:** Categories 1 and 2 (explicit concern,
narrative reframing/confabulation) appear to be the classifier's real
strength and the most decision-relevant categories for H4's garbage-
filter question anyway -- they're what distinguishes "the agent noticed
and said so" from "the agent invented a story to explain away the
anomaly." The 3-vs-4 boundary may need a different approach entirely
(possibly telling the classifier the field's normal range, which
trades blindness for resolving power -- an open design question, not
decided here) or may simply need to be reported with this limitation
explicitly disclosed rather than solved. Revisit when returning to this
thread; not resolved further tonight per the overfitting-risk call.

## Correction: two of three "reformatted_not_reexamined" labels were engine-side, not agent-side

**Checked directly against dither_reference.json (value the agent was
shown) versus the reasoning text it wrote.** The earlier taxonomy
conflated what the dither engine did to a value with what the agent did
with it. We labeled by knowing the engine's rescaling; the classifier,
correctly, could only see the text.

- CUST_000003: shown 27.8 (original 0.278); text says "27.8". Verbatim
  echo of the corrupted value. No agent-side reformatting.
- CUST_000022: shown 36.0 (original 0.36); text says "36.0". Echo.
- CUST_000009: shown 24.9 (original 0.249); text says "24.9%". The
  agent appended a `%`. The only genuine agent-side reformatting of
  the three.
- CUST_000007: shown 570 (original 19); text says "47.5 years" and
  "long-tenured". Agent converted months to years AND attached an
  evaluative label. narrative_reframing label stands.

**Corrected labels:** 003 and 022 are plain_restatement. Re-scored
against the three classifier runs already completed: v1 4/7, v1 with
two fixes 5/7, v2 4/7 (previously 3/3/3 under the original labels).
These tallies are NOT validation: the relabeling was proposed after
seeing classifier outputs, on the same 7 examples used for tuning, and
differences of 4 vs 5 vs 4 at n=7 are noise. The relabeling itself
rests on a fact from the data (shown value vs written text), not on
classifier behavior.

**Consequences for earlier claims in this document:**
- The "regression" diagnosed in the second run (spillover from the
  Category 2 fixes into Category 3) was partly an artifact of the
  mislabeling: 003 and 022 moving to plain_restatement was correct
  movement. The v2 restructure was partly motivated by that reading;
  it still fixed CUST_000026, but the spillover explanation is
  overstated.
- "Scale-insensitive absorption" as described above (agent "re-labels"
  the number with a `%`) is accurate only for CUST_000009. For 003 and
  022 the agent echoed an out-of-range value verbatim while still rating
  it "moderate", which is consistent with silently treating it as a
  percentage, with no trace in the text. That is an inference from the
  severity word, not an observation.
- Remaining real misses depend on knowing a field's normal format or
  range (009: `%` is unremarkable without knowing decimals are the
  norm; 007: 47.5 years is unremarkable without knowing tenure is
  normally under ten years). CUST_000003's narrative_reframing call in
  v2 is a genuine over-call of ordinary multi-factor reasoning.

**Open, pending decision (not resolved here):** (1) merge Categories 3
and 4 into one "unremarked" category, since genuine agent-side
reformatting is rare in this data and not blind-detectable; (2)
whether agent-side reformatting is better measured deterministically
by comparing numbers in the text against the known shown value
(echo vs. converted form), a data-aware check that measures what the
agent did to the number rather than whether it noticed anything.

## Decisions locked: three-category judge, deterministic echo check, architecture

Agreed after the correction above:
1. **The LLM judge merges Categories 3 and 4 into `unremarked_usage`.**
   Final categories: `explicit_concern`, `narrative_reframing`,
   `unremarked_usage`. A blind judge cannot reliably tell "reformatted
   without comment" from "plain restatement" because that needs the
   field's normal format, which it is deliberately not given. Prompt
   rewrite NOT yet done; the judge prompt text must stay generic (no
   examples drawn from our own findings, no hypothesis vocabulary such
   as "explain away the anomaly").
2. **What the agent did to the number is measured deterministically.**
   `classify_value_echo()` in evaluate_core.py compares the value the
   agent was shown (dither_reference.json) against its reasoning text.
   Classes: echo, echo_pct_marker (same number, % attached), converted
   (with factor and the unit word that follows), magnitude_only (sign
   dropped), omitted. Descriptive only; "notable" is defined relative to
   the clean-baseline rate. Not blind by design: it measures what
   happened to the number, not whether the agent noticed anything.
3. **Architecture:** primitives live in evaluate_core.py; the
   deterministic H5 fields (keyword hit, echo class, mention flag) are to
   be computed inside load_condition() so H1-H4 get them without a retrofit
   or a second pass over the JSON; the LLM judge stays a separate script
   whose output joins by id; evaluate_h5.py aggregates. WIRED (see below).
   Caveat: the keyword scan hits anywhere in the text and cannot say which
   field was doubted, so it is a coarse proxy for H4's field-specific
   question; the judge is what attributes detection to a field.

## Echo check: results (n=30 dithered customers x 3 H4 implausible conditions, Haiku 4.5, temp 0)

Built and tested (31 original cases, then 17 more from real agent text),
then refined after reading real output: unit-word capture, ratio tokens
like NPS "5/10" skipped in conversion tests, "47+" read as at-least,
"$8.5K" notation. One refinement unmasked a case: CUST_000009's real
"87.5 months" had been hidden behind a false conversion match on the "10"
in "5/10", caught first. Hand-checked cases (003, 022, 009, 007) all agree
with the instrument.

| Condition | echo | echo + % | converted | omitted |
|---|---|---|---|---|
| churn, dithered (n=30) | 10 | 11 | 0 | 9 |
| churn, clean (150 runs) | 135 | 0 | 0 | 15 |
| spend, dithered (n=30) | 18 | 0 | 1 | 11 |
| spend, clean (150 runs) | 92 | 0 | 0 | 58 |
| tenure, dithered (n=30) | 4 | 0 | 11 | 15 |
| tenure, clean (150 runs) | 52 | 0 | 10 | 88 |

Wilson 95% intervals on the dithered rates are wide at n=30 (e.g. 37%
converted on tenure: 22-54%). The clean floor is 5 correlated runs per
customer: point estimates only.

**Findings, with the strength each one actually has:**
- **Churn: agents add a % sign to the corrupted value in 11/30 (37%)
  versus 0/150 clean runs.** Same customers, paired: exact McNemar
  p = 0.001. This corrects an earlier statement in this document that
  agent-side reformatting was rare; that statement rested on only the
  three hand-labeled cases. No agent wrote the value back on the 0-1
  scale (converted = 0), so none visibly self-corrected.
- **Tenure: 11/30 converted, in two different ways.** Six wrote a real
  unit change (e.g. shown 570 -> "47.5 years"). Five divided by 12 but
  KEPT the "months" label (shown 540/780/810/930/1050 -> "45", "65",
  "67.5", "77.5", "87.5 months"). The clean baseline has ten
  conversions, all /12 with "years" and none with "months". Same
  customers, 5 vs 0: exact McNemar p = 0.0625 -- suggestive, NOT
  significant at n=30. The two with a .5 decimal cannot be coincidence.
  Interpretation is open: the numbers match shown/12 exactly with a
  months label, but intent is unobserved (a rescale into a believable
  range versus a unit-label slip).
- **Omission is elevated only for churn (30% vs 10% clean);** spend (37%
  vs 39%) and tenure (50% vs 59%) show no difference. Suggestive; needs
  crossing with drift before calling it filtering.
- **Spend: 1/30 shown $500,255.00 and wrote "$5,002.55"**, exactly
  shown/100 to the cent. Verified since (see "Pending checks resolved"): no other input field
  carries 5002.55, so the agent derived it from the shown value.
- **Hypothesis from the tenure table, not a finding:** years-labeled
  cases were shown 150-630 and months-labeled cases 540-1050 -- larger
  values leaning toward the kept-months label, with overlap. Needs a
  dose-response look at n=1000.

**Pending checks:** both resolved, see "Pending checks resolved" below.

**Caveats:** one model and temperature; clean floor runs not independent;
the `low_specificity` flag over-warns on tenure (whole-number years are
genuine conversions: 3 of the 6 real unit changes tripped it) so it is a
"look at it" flag, not a verdict.

**Why this matters beyond itself (hypothesis for the full run):** H4's
garbage-filter pattern (lower drift on implausible than plausible values)
has two candidate mechanisms the drift rate cannot separate: the agent
notices and discounts the value, or it silently rescales it into a
believable range. Crossing echo class with whether the decision drifted
distinguishes them. At n=30 this cross is too small to read; at n=1000 it
is the main payoff.

**Open:** rewrite the judge prompt for three categories; run the echo x drift cross-tab on the full
run.

## Wired into load_condition()

load_condition() now adds three keys to every joined record:
`h5_keyword_detected` (bool), `h5_keyword_patterns` (list), and
`value_echo` (dict: dithered field -> classify_value_echo result), so
H1-H4, H7 and H8 get the deterministic H5 measurements with no retrofit
and no second pass over the JSON.

**Verified before wiring, because every evaluator depends on this
function:**
- The "shown value" assumption: dither_reference.json's value equals
  agent_input.jsonl's value for all 6,859 dithered (record, field) pairs
  across all 53 conditions (n=60, 25 distinct fields): 0 mismatches.
- Every old key is byte-identical in the new output across 3,180
  records; new keys present everywhere; non-numeric fields correctly
  `not_applicable`.
- Semantic check: all 687 records citing the shown value of their first
  dithered field classify as `echo`.
- Evaluator regression: evaluate_h1, h2, h3 and h4 run against the old and
  new evaluate_core on identical data produce byte-identical results, and
  h4 passes validate_h4_schema.py.

Limits to carry forward: the keyword scan is anywhere-in-text and cannot
attribute a doubt to a field; the echo check only covers numeric fields
(strings and booleans return `not_applicable`); and the clean-baseline
floor (agent runs against original values) is still computed only in
analyze_echo_check.py, so evaluate_h5.py will need a core helper for it.

## Pending checks resolved, two keyword detections read, and a ground-truth correction

**Correction: in the smoke-test directory, finalized_ground_truth.json is
the Tier-1 FAKE** (100 customers, random decisions; the real baseline has
30). Any `drifted` computed through load_condition() against it is
meaningless there. Real drift in these smoke tests = decision vs.
baseline majority. A snippet supplied during this work printed `drifted`
from the fake file; those values were discarded. h4_results.json
produced in that directory is structural validation only. The file
should be renamed so no evaluator reads it by accident.

**Spend (CUST_000003): verified.** Shown 500255.0, agent wrote
"$5,002.55" = shown/100 to the cent. No numeric input field equals
5002.55, and avg_order_value x total_purchases = 4894.56, so it was not
computed from other fields: the agent derived it from the shown value,
consistent with reading it as cents. n=1; reasoning unobserved.

**Tenure "months"-labeled cases: verified, with one clean regularity.**
True originals 18, 26, 35, 27, 31; shown 30x; agent wrote 45, 65, 87.5,
67.5, 77.5 months. Each is exactly 2.5x the true tenure (x30 from the
engine, /12 from the agent). All five fall inside the field's defined
1-120 range, so downstream reasoning sees a believable tenure that is
wrong by a constant factor. Still 5 vs 0 clean, exact McNemar p = 0.0625:
suggestive at n=30, not established. Whether it is a rescale or a
unit-label slip remains unobserved.

**The two keyword hits (both pattern "data quality", both genuine) both
detected by cross-checking another field in the record:**
- Spend CUST_000006: "The discrepancy between total_spend ($25,736) and
  lifetime_value_estimate ($339.91) suggests data quality issues."
  Nothing about $25,736 alone is absurd (the field allows $500,000); only
  the cross-reference exposes it.
- Tenure CUST_000008: "1380 months/115 years tenure indicates data
  quality issue, but account created 2021 shows ~3.5 years actual
  tenure." The agent flagged it and reconstructed a near-true tenure from
  `account_created_date` (if x30 produced 1380 the original was 46
  months, about 3.8 years; not directly printed).

**Real drift and stability for those two:**
- CUST_000006: baseline stable (5/5 MEDIUM). Detected, flagged, AND the
  decision drifted from baseline: the first real specimen of detection
  without protection (n=1; clean attribution because the customer is
  stable).
- CUST_000008: baseline deeply_boundary (3 MEDIUM / 2 HIGH). Its drift is
  NOT interpretable as a corruption effect: it flips on clean runs about
  40% of the time. Concrete H6 lesson: separate boundary customers before
  comparing detection to drift.

**Keyword floor:** 0 hits in 150 clean baseline runs vs. 2 in 90
dithered records. The scan does not fire on clean text here, so there is
no background rate to subtract. Two events cannot carry a test; no
p-value computed.

**Hypotheses from these cases (NOT findings; test at n=1000):**
- Detection may track REDUNDANCY in the record more than a stated range.
  churn_risk_score has an explicit 0-1 range in the prompt and drew 0
  keyword hits in 30; both detections came from fields with a redundant
  sibling in the record (total_spend <-> lifetime_value_estimate;
  tenure_months <-> account_created_date). This would cut against the
  stated-range detectability gradient in the H4 design notes. A
  redundancy table (which dithered fields have a cross-checkable
  counterpart) should be added to the design, and bears on external
  validity: real enterprise records are often redundant in this way.
- Detection does not guarantee protection (CUST_000006).

## Judge prompt v3: FROZEN as judge of record; mechanics verified, accuracy NOT validated; revisit after H6

**Frozen:** PROMPT_ID `v3-be665399de` (hash of the exact prompt text; every
output row from now on carries it), model claude-haiku-4-5-20251001,
temperature 0. Three categories (`explicit_concern`, `narrative_reframing`,
`unremarked_usage`), zero-shot with no examples, evidence-first JSON with a
machine-checked verbatim quote, a `figure_referenced` applicability flag,
and a priority rule (explicit_concern > narrative_reframing >
unremarked_usage). The v3 outputs generated so far predate the id tag; the
prompt text has not changed since. Decided deliberately NOT to iterate
further: four rounds of tuning on the same 7 texts each traded one miss
for another, which says the category boundaries are fuzzy for this task,
and further tuning on that set is overfitting.

**Revisit after H6**, with a larger and more comprehensive smoke test and
dataset. Rule for any later revision: v3 remains the judge of record for
the first full-run analysis. A v4, if built, runs in parallel on the same
data as a sensitivity check and the write-up reports whether conclusions
depend on the version. The risk is not bias inside the judge's outputs; it
is tuning the prompt after seeing which way headline counts move.

**Mechanics on the 90 existing records (30 x 3 H4 implausible
conditions):** 0 API errors, 0 parse failures. 0 non-verbatim quotes
across all 90 (grep -c '"evidence_verbatim": false' on the three output
files: 0 each), so the quoted evidence can be trusted for auditing. This
says nothing about category accuracy.
Category distribution: unremarked_usage 78, narrative_reframing 7,
explicit_concern 5.

**Cross-instrument checks (agreement between instruments is not accuracy):**
- Both known genuine detections (CUST_000006 spend, CUST_000008 tenure,
  the only two keyword hits) were labeled explicit_concern with verbatim
  evidence: recall 2/2 on the texts we know are real detections.
- Judge said "figure not referenced" 27 times; in all 27 the echo check
  also shows the value omitted (0 contradictions). Echo-omitted totals
  (9, 11, 15) match the earlier counts. 8 further cases are
  referenced-by-description with no number. One likely strictness error:
  CUST_000018 (churn) was judged unreferenced despite "despite low churn
  risk", which the prompt counts as a description-level reference.
- Of 5 explicit_concern calls: 2 clear true detections (006, 008); 1
  genuinely ambiguous (CUST_000010: supplies a cause AND flags it for
  investigation; the prompt's own priority rule favors explicit_concern);
  2 questionable (CUST_000013 "negative (favorable)", which is reframing
  by our reading; CUST_000028, "concerning" read as doubt when it is a
  severity adjective). None of the 3 judge-only explicit_concern calls (no
  keyword hit) is a clear detection: no sign here of the keyword list
  missing real explicit detections, but the judge has its own misses, so
  this is not evidence of keyword recall.

**Known weak spots (all small n):**
- **The central pattern is inconsistent.** Negative-value-presented-as-
  favorable: CUST_000024 -> narrative_reframing (matches our reading);
  CUST_000026 (same phrase as 013) -> unremarked_usage; CUST_000013 ->
  explicit_concern. One of three as expected. This is the confabulation
  signal for the x-1 implausible-down arm, so **judge-derived
  narrative_reframing counts are PROVISIONAL**; headline numbers should
  rest on the deterministic instruments (echo class, keyword scan, real
  drift) until the human audit calibrates the judge.
- **Over-call on contrast language:** CUST_000003 was called reframing in
  both the churn and spend conditions, triggered by tension phrasing
  ("given the severe disengagement signals", "despite being medium-
  value"). "Moderate" is exactly the standard qualitative word the prompt
  exempts.
- **Severity adjective read as doubt:** "concerning" (CUST_000028).
- **Text-derivable labels:** the judge sees only text, so oddness the text
  does not expose (silent rescale; "long-tenured (47.5 years)", CUST_000007,
  flagged borderline) comes back unremarked. That is why the echo check
  exists and why the two combine in the echo x judge matrix.
- Accuracy on our 7 hand-labeled texts: 3/7 (009, 022, 024 match). Fair
  accounting: 2 clear errors (003 over-call, 026 miss), 1 ambiguous (010),
  1 borderline-by-text (007).

**Planned human audit (replaces the original random-zero-hit design for
validating the JUDGE):**
- Stratified by the judge's label, not a random sample of zero-hit texts
  (that would be almost all unremarked_usage, too few reframing cases to
  estimate sensitivity). Include the implausible-down (x-1) arm on
  purpose, plus a slice of unremarked_usage to estimate misses.
- Labels must be derivable from the text alone; auditors work blind.
- Offer a "both / ambiguous" option and report its rate rather than
  forcing a pick (CUST_000010 is the model case).
- Use texts we have not already read: exclude the 19 qualifying customers
  and the tenure examples examined for the echo check.
- Report agreement with and without ambiguous cases, per category.

**Open design decision:** blind (current default) vs. range-informed
variant (tell the judge each field's documented range). Range knowledge
would help on oddness the text hides but ends blindness; evaluate as a
labeled variant on the same audit sample, not as the default.

**Remaining H5 work (not started):** core helper for the clean-baseline
floor (currently only in analyze_echo_check.py), needed by evaluate_h5.py;
redundancy table of dithered fields (which have a cross-checkable
counterpart in the record); evaluate_h4's garbage_filter_analysis() stub
can now be wired to the echo class, keyword scan and judge.

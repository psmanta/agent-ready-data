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
| spend, dithered (n=30) | 20 | 0 | 1 | 9 |   (was 18 / 11 before the magnitude-suffix fix below) |
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
- **RETRACTED: "omission is elevated for churn (30% vs 10% clean)."** See
  "Echo-check audit" below: the clean omissions are a few customers who never
  quote the score, and most dithered omissions still describe churn
  qualitatively. Spend (30% vs 39% after the fix) and tenure (50% vs 59%)
  show no difference.
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

## Design finding: the engine propagates corruption to some sibling fields (verified n=60)

`recompute_derived` defaults to True, and `_recompute_derived` covers only
two relationships: churn_risk_score -> is_at_risk (>= 0.60) and
support_tickets_open -> recently_contacted_support. Verified against real
engine output (clean record vs. dithered record, field by field):
- **H4 churn conditions:** is_at_risk changed in 25/60 (implausible) and
  32/60 (plausible) records. It agrees with the SHOWN (corrupted) score in
  60/60 and with the original score in only 35/60.
- **H4 spend and tenure conditions:** no field other than the targeted one
  changed. lifetime_value_estimate, account_created_date and the rest keep
  their original values, so a contradiction is available to a careful reader.

**Consequences:**
1. The detection contrast (0/30 on churn vs. the two cross-field detections
   on spend and tenure) is partly engine-made: churn's redundant sibling was
   made to AGREE with the corrupted value, while spend's and tenure's were
   left contradicting it. It is not independent evidence on stated-range
   validation vs. relational coherence. A second confound: a scale-swapped
   churn value of 24.9 can be read as 24.9% (11/30 agents added a % sign).
2. Churn drift in H4 conflates the corrupted score with a changed at-risk
   flag that agents explicitly cite ("flagged as at-risk").
3. The three H4 fields differ in propagated vs. isolated corruption. State
   this in the design section of any write-up; per-field reporting (already
   planned) is the right unit.
4. **Decided and built:** churn plausible/implausible with
   recompute_derived=False (2 conditions, about 2,000 calls, roughly $5)
   turns the redundancy hypothesis from an observation into a manipulated
   test. See "Isolated-corruption arm: built" below.

**Redundancy table (to build):** columns = dithered field, redundant
siblings present in the record, propagated by the engine (yes/no). Derive it
from the generator's field relationships and the engine's propagation list,
pre-specified BEFORE the full run. Never derive it from which fields
happened to produce detections; that would be circular.

## Plan (agreed order, with adjustments)

1. **Plumbing:** clean-baseline floor helper in evaluate_core.py (verify it
   reproduces the floor counts already obtained: churn 135 echo / 0 / 0 /
   15 omitted, spend 92 / 0 / 0 / 58, tenure 52 / 0 / 10 converted / 88, of
   150 runs each); the redundancy table; garbage_filter_analysis() wired
   once, stratified by the existing `stability_tier` already carried on
   every record (no need to wait for H6).
2. **H6:** tier labels already exist (aggregate_baseline.py ->
   finalize_ground_truth -> stability_tier on every record). To build:
   evaluate_h6.py (per-tier drift rates, ordering consistency, median
   ratio, Fisher check) and the adaptive refinement runner.
3. **The full run is the gate for everything "executed."** Amendment
   estimate before boundary expansion: about 58,000-59,000 calls,
   $135-137 standard / $67-68 Batch. The judge costs about $0.001 per
   record (roughly $6 for H4's six conditions, roughly $52 for all). Run the
   5,000-call baseline first (about $12): tiers gate every stratified
   analysis.
4. **Headline cross-tabs (echo x drift, detection x drift):** primary
   analysis on stability_tier == stable for clean attribution, with boundary
   strata reported SEPARATELY, not discarded (H6 asks about them directly,
   and dropping them changes the estimand).
5. **Stratified human audit** per the design above.

## Findings to carry into the final analysis, and what each needs to graduate

1. **Tenure rescale regularity.** Observed: 5/30 dithered tenure records
   divide the shown value by 12 but keep the "months" label, each landing
   exactly 2.5x the true tenure (x30 from the engine, /12 from the agent)
   and inside the field's 1-120 range; 0/150 clean runs do this (paired
   exact McNemar p = 0.0625). The arithmetic is solid; "implicit rescaling"
   is an interpretation (a unit-label slip fits equally well). Graduates
   with a replication at n=1000 and its magnitude dose-response.
2. **Redundancy hypothesis.** Reworded as a hypothesis, with the
   propagation confound above stated alongside it. Graduates only with the
   pre-specified redundancy table and either the n=1000 detection rates by
   table row or the no-propagation churn conditions.
3. **Deterministic text methods fail on this task.** Whole-text severity-word
   matching failed (vocabulary reused across independent judgments); the
   windowed-proximity variant failed on the same example ("low fraud risk"
   within eight words of "churn risk"). This is a solid NEGATIVE result for
   the two simple approaches, and it justified building the judge. It does
   not show that a classifier is required or sufficient: the judge gave
   three different labels to three near-identical phrasings, so claim only
   the negative result.

## Isolated-corruption arm: built (2 conditions; H4 now 8, total 55)

`h4_churn_risk_score_plausible_isolated` and
`h4_churn_risk_score_implausible_isolated` in dither_engine.py, with
recompute_derived=False and the SAME seeds as their propagating
counterparts. Verified at n=60 with all 55 conditions regenerated:
- the validity check passes for both;
- the matched pair gets the identical corrupted value and operator in
  60/60 records each, with no field other than is_at_risk differing;
- the isolated arm keeps the ORIGINAL is_at_risk in 60/60;
- the flag differs between arms in 32/60 (plausible) and 25/60
  (implausible): the contradiction-carrying records, matching the counts
  found earlier. In the other records the two arms' inputs are identical,
  a built-in negative control for agent run-to-run noise;
- regression: all 106 files (dither_reference.json and agent_input.jsonl)
  across the 53 original conditions are byte-identical to the
  pre-change generation.

Counts updated everywhere: 55 conditions (H4 = 8); generate_dithered_data.py
said "50 unconditional", already stale before this change, now 55. Amendment
updated (new subsection "Propagated vs. isolated corruption", condition
table, call counts, cost: about 60,000-61,000 calls, $139-142 standard /
$70-71 Batch before boundary expansion). Pre-specified analysis (paired
McNemar's on drift and detection, split by contradiction-carrying vs.
input-identical records) is written into the amendment.

**Not yet built:** the evaluator side. evaluate_h4.py still loads the nine
original conditions; the isolation analysis belongs with the
garbage_filter_analysis() wiring.

## Echo-check audit: reading the "omitted" bucket on both sides; one matcher bug fixed

**What the floor-helper check did and did not prove.** h5_verify_floor_helper.py
reproduces the earlier real-data counts exactly (churn 135/0/0/15, spend
92/0/0/58, tenure 52/0/10/88 of 150 runs; keyword floor 0/150): the helper
equals the old logic. It cannot show the old logic is right, because both
share classify_value_echo(), and every matcher miss lands silently in
"omitted". h5_audit_omitted.py reads that bucket on both sides.

**Hypothesis tested and NOT supported.** The matcher requires two significant
digits for rounded matches, so single-digit approximations ("nearly 4 years",
"roughly 2 years", "over 4 years", number words) score as omitted. Clean
tenure values are small and dithered ones large, so this could have flattered
the dithered-vs-clean conversion comparison. In the real data it does not:
of 88 clean tenure runs scored omitted, 78 do not discuss tenure at all; the
only 5 with a number-bearing sentence are one customer writing "9+ months of
inactivity" (a different quantity). The tenure conversion comparison (11/30
dithered vs. 10/150 clean) is not flattered by this gap in this sample.

**Matcher bug found by the audit, fixed.** Corrupted values are often huge and
agents abbreviate them. Two spend texts were scored omitted: shown $4,778,987,
wrote "$4.78M"; shown $4,000,427, wrote "$4M+". The matcher handled K but not
M. Now handles K/k, M, B and the words thousand/million/billion; lowercase "m"
is deliberately not a suffix (months). Predicted before the rerun and
confirmed: spend dithered echo 18 -> 20, omitted 11 -> 9; no other cell moved,
including both clean floors.

**Effective sample size.** Each condition shows 84 distinct clean texts among
150 runs (the same 30 customers' baseline texts every time): 44% of runs are
exact duplicates at temperature 0. The floors are point estimates; do not put
intervals on them.

**Churn omission: retracted as a headline.** The clean 15 omitted runs look
like exactly three customers (000011, 000027, 000008) omitting the score in
all five runs (inferred from sampled texts; confirm with customer-level
lists). At least two of them (000011, 000027) are also among the dithered
omitters, and only 1 of the 9 dithered churn omissions fails to discuss churn
at all: the rest describe it qualitatively ("negative (favorable)", "low churn
risk", "flagged as at-risk"). "Omitted" means the NUMBER is absent, not the
field; quoting vs. describing, not dropping. The judge's figure_referenced
flag is the right instrument for "field not mentioned".

**Spend omission:** about 30% dithered vs. 39% clean after the fix: no real
difference. All eight sampled clean candidates are sentences about the
lifetime value estimate: agents cite lifetime value instead of total spend.

**A lead, not a finding.** CUST_000020, shown 1,680 months (140 years), wrote
"long-tenured customer (14 years)": exactly /120, as if a zero were dropped
as a typo, landing on a believable value (the true tenure was 56 months). The
matcher does not test /120, so it sits in omitted. n=1. **Policy: freeze the
confirmatory factor list now** (x100, /100, /12, x12, plus K/M/B/word
suffixes). Any further factor (e.g. /120) belongs only in a separately labeled
exploratory scan at n=1000; adding factors after seeing one case is the same
forking-paths problem avoided elsewhere.

**Where each earlier echo claim now stands:**
- Solid: churn % sign 11/30 vs. 0/150 clean (paired exact McNemar p = 0.001);
  unaffected by the audit.
- Suggestive only: tenure kept-"months" label 5/30 vs. 0/150 (p = 0.0625);
  unaffected by the audit.
- Holds: the tenure conversion comparison is not flattered by the feared
  single-digit-years undercount in this sample.
- Retracted: churn omission elevated.
- No difference: spend and tenure omission.

**Open:** customer-level omission pairing (a "clean omitter" = at least 3 of 5
clean runs omit the value; pair against the dithered row); a slice of clean
baseline texts in the human audit as negative controls for the judge's
false-positive floor (about $0.15 of judge calls per field).

## Garbage-filter analysis: built (evaluate_h4.py; replaces the stub)

**What it computes**, per field and arm (drift, plausible, implausible; plus
the two isolated arms for churn), per stratum (stable / boundary / all):
drift, keyword detection, drift by detected vs. not detected, drift by echo
class, and drift by judge category when judge outputs are supplied
(`--judge_dir classifier_output`, joined by id `customer:field`). Also the
detection spike (paired McNemar's, implausible vs. plausible) and the
isolation analysis (isolated vs. propagated churn, split into
contradiction-carrying and input-identical customers, the latter a built-in
negative control). A design check reads both isolated arms' dither_reference
and reports whether the matched-pair assumption (identical corrupted churn
value per customer) actually holds. Reusable primitives live in
evaluate_core.py: stability_stratum, outcome_rate_by_group,
paired_binary_comparison, load_judge_results. The pre-specified primary tests
(5 per family, Holm-adjusted) are recorded in the amendment.

**Verified on a planted-effect environment** (real engine output for all 55
conditions at n=60, synthetic decisions with effects planted on purpose):
- recovers the planted detection spike on spend and tenure (16 vs 0 and 15 vs
  3 discordant pairs) and stays null on churn, where none was planted;
- recovers the isolation effect only among contradiction-carrying records
  (10 vs 0, p = 0.002 implausible; 7 vs 1, p = 0.07 plausible, correctly
  flagged low_power); the input-identical negative control shows exactly zero
  differences on keyword, drift and judge;
- shows lower drift on implausible only among stable customers, with boundary
  customers behaving as planted noise;
- drift by keyword, echo class and judge category follow the planted structure;
- failure paths: runs without a judge, without the isolated conditions, and
  reports a broken matched-pair assumption (one altered value flips design_ok
  to false);
- the four pre-existing analyses are byte-identical old vs. new on the same
  data; validate_h4_schema.py now checks the whole block (2x2 sums, ranges,
  interval/rate consistency, primary tags only on the stable stratum, Holm
  >= raw) and catches all six corruptions tried.

**Not verified:** anything on real agent output. The synthetic environment
proves the code recovers effects it is told exist; it says nothing about
whether those effects exist. The smoke-test directory is unsuitable for a real
run (mixed seeds and sample sizes, no real ground truth, no isolated
conditions), so the first genuine exercise is the full run.

**Still open from the plan:** redundancy table; customer-level omission
pairing; evaluate_h6.py and the refinement runner; the full-run spending
decision.

## H4 garbage-filter analysis: implemented in evaluate_h4.py; independently verified

`garbage_filter_analysis()` is no longer a stub. For each field and arm
(drift, plausible, implausible; for churn also the two isolated arms) and
each stratum (stable = clean attribution; boundary = lightly/deeply boundary
and tied_no_majority, reported separately and never dropped; all) it reports:
drift rate; keyword-scan detection rate; drift by detected/not-detected (the
four-way table); drift by echo class; and, when judge outputs are supplied,
drift by judge category. Plus the detection spike (paired exact McNemar's,
implausible vs. plausible, with the raw 2x2), the isolation analysis for churn
(isolated vs. propagated, split into contradiction-carrying and
input-identical records, the latter a built-in negative control), and the
primary tests gathered into a Holm-adjusted family per detector (keyword;
judge). Usage: `python3 evaluate_h4.py [--judge_dir classifier_output]`; the
isolated conditions are optional (skipped with a notice if not generated).

**Verified against known answers (the code was reviewed and tested, not
assumed):** with decisions planted at known drift and detection rates per arm,
identical outputs for input-identical customers, and boundary tiers, an
independent recomputation (plain loops, scipy exact binomial, statsmodels
Holm; none of the evaluator's code paths) matched on 1,021 checks: every
arm x stratum table, Wilson intervals, four-way tables, spikes, the isolation
split and its negative control (zero discordant by construction), exact
p-values, and the Holm-adjusted family. Edge cases: no events anywhere (no
crash; all p = 1.0, flagged low_power, no direction claimed); isolated
conditions missing (skipped cleanly); judge join with errored rows (dropped:
57 of 60 joined; judge spike matched an independent recomputation; unjudged
conditions carry no judge fields). validate_h4_schema.py caught seven
deliberate corruptions by name. The four unrelated sections of the H4 output
are byte-identical to the previous evaluator on the same data.

**Pre-specification gap to close before the full run:** the code tags five
tests primary (three detection spikes, implausible vs. plausible on stable
customers, one per field; two isolation comparisons on contradiction-carrying
stable customers; predicted direction a_only > b_only; Holm within each
detector family). The amendment pre-specifies only the isolation comparisons,
so the three detection-spike tests are not yet pre-specified anywhere.

**Limits carried forward:** keyword detection is anywhere-in-text and cannot say
which field was doubted; judge-derived numbers are provisional until the human
audit; most paired comparisons will be low-powered on rare events, so read
the raw 2x2 counts; records with a missing or unrecognized stability tier fall
in "all" only.

## Pending amendment edits: checklist for the full amendment pass

Captured now so nothing is lost; to be folded into 1b_DESIGN_AMENDMENT_1.md when
the full amendment is revised, not before. The H4 isolated-corruption arm,
condition counts and cost figures are already in the amendment.

**1. H4 statistical plan: pre-specified primary tests (decision: capture now,
fold in later; must be in the amendment BEFORE the full run).** Proposed text:

> **Pre-specified primary tests for the garbage-filter question.** Declared
> before the full run. (1) Three detection-spike tests, one per field: among
> stable customers, a paired exact McNemar's comparing explicit detection (the
> frozen keyword scan) under the implausible condition against the plausible
> condition for the same customers. Predicted direction: more detection under
> implausible. (2) Two isolation tests: among stable customers whose
> `is_at_risk` propagation would have flipped (the contradiction-carrying
> records), a paired exact McNemar's comparing keyword detection in the
> isolated arm against the propagated arm. Predicted direction: more
> detection in the isolated arm. The five tests form one family corrected
> with Holm; the judge's `explicit_concern` forms a parallel family once the
> judge is validated. Everything else in the garbage-filter analysis (drift
> and echo-class tables, boundary-stratum results, drift comparisons,
> judge-derived numbers before human-audit calibration) is exploratory and
> labeled as such. The raw 2x2 counts are reported with every test because
> most comparisons will be low-powered on rare events.

These correspond exactly to the tests tagged `primary` in evaluate_h4.py.

**2. H5 section rewrite.** The amendment still describes the earlier design.
To reflect: (a) the frozen keyword list: the amendment's regexes lack the
adverb forms; the seven adjective patterns (unusual, atypical, implausible,
odd, strange, suspicious, questionable) now include them, and the code in
evaluate_core.py is authoritative; (b) the three-category blind judge,
frozen as PROMPT_ID v3-be665399de, with the figure_referenced flag, the
machine-checked verbatim quote, and the priority rule; (c) the deterministic
echo check: its classes, and the frozen confirmatory factor list (x100,
/100, /12, x12, plus K/M/B and word suffixes), exploratory factors kept
separate; (d) the clean-baseline floors (echo and keyword); (e) the
stratified human-audit design for validating the judge (stratified by judge
label, implausible-down arm included, text-derivable labels, an ambiguous
option, texts not previously read, clean baseline texts as negative
controls); (f) judge-derived numbers provisional until that audit.

**3. Statistical Methodology note:** a new entry for paired exact McNemar's on
rare binary events (detection), reported with the raw 2x2 and a low-power flag,
with Holm-adjusted primary families.

**4. Cost:** the judge adds about $0.001 per record; scope (H4 only vs. every
condition) is still undecided, so the cost paragraph should state the unit
cost and leave the total open.

**5. Redundancy table (to be built, pre-specified before the full run):**
dithered field; redundant siblings present in the record; whether the engine
propagates the corruption to them. Derived from the generator's field
relationships and the engine's propagation list, never from which fields
produced detections.

## Repo vs. sandbox sync check: generator, engine, and a date-dependence finding

**Hash comparison (repo vs. my copies).** `validate_dama_dimensions.py` and
`generate_dithered_data.py` matched. The generator differed (709 vs. 704
lines) but only in the module docstring (a note about the post-1a refactor):
no code difference, so every earlier check against the field relationships
stands. The engine did NOT match: the repo's was the pre-isolated-arm version
(1,182 lines, hash 5d1b0982..., byte-identical to my earlier copies), so the
isolated-churn conditions never reached the repo engine even though the
amendment, `generate_dithered_data.py` and the commit message said 55
conditions; on the repo it built 53. Fix: replace it with the current engine
(1,208 lines, hash 917cf4e3b4ad452c): the isolated arm plus a stale
header-comment correction ("H4 reduced from 6 to 3", "Total: 50-51" ->
"55-56"). Verified: code identical outside comments; builds 55 conditions.

**Equivalence check.** All 55 conditions regenerated at n=60, seed 42, with the
repo's generator and the updated engine: 110 files byte-identical to the
earlier verified run once Faker's clock was pinned (see below).

**Finding: regenerated data is not reproducible across calendar dates.**
Faker's `date_of_birth` is computed relative to today's date, so the same seed
gives different `dob` values on different days (observed: shifted by exactly
the two days elapsed, in all 60 records; nothing else differed). The agent
sees `dob`. Every invocation of `generate_dithered_data.py`, including
`--baseline-only` and `--condition X`, regenerates the base customers and
overwrites the baseline input and `canonical_customers.json`. Consequences if
unaddressed: a baseline generated on one day and conditions generated on
another differ in `dob` (so "differs only in the dithered field" fails
slightly, for a field no hypothesis targets); a single condition regenerated
later is inconsistent with its siblings; the saved canonical file may not match
what the agent saw. The effect on decisions is almost certainly negligible; the
provenance problem is not. **Procedure until a code fix is decided:** generate
the baseline and all conditions in ONE invocation on one day, archive
`canonical_customers.json` with its hash, and never rerun any generation
command afterwards. The baseline-first gate is a gate on agent calls, not on
generation, so this costs nothing. Candidate code fix (decision pending): a
guard in `generate_dithered_data.py` that aborts when a stored canonical file
exists and the regenerated base differs, unless `--force-regenerate` is passed.
Changing how the generator produces `dob` is NOT recommended: it risks
shifting the random stream for every later field.

## Redundancy table (field_redundancy.py): DRAFT, pending review before freeze

Pre-specified table of which dithered fields have a cross-checkable sibling and
whether the engine removes the contradiction. Derived only from the
generator's formulas and the engine's propagation list. Tiers: **strict**
(exact deterministic function), **bounded** (hard constraint inferable from
what the fields mean), **approximate** (formula with a bounded random factor),
**soft** (a hard band by `customer_segment` in the generator, never stated to
the agent), none. "Class" = strongest tier among siblings the engine does NOT
recompute; reported for default conditions (recompute_derived=True) and for
the isolated arm (False).

Results that matter for H4: `tenure_months` is strict in both (uncorrected
`account_created_date`); `total_spend` is approximate in both (lifetime value;
purchases x average order value); `churn_risk_score` is **soft_only in the
default arm** (the engine rewrites the strict sibling `is_at_risk`) and
**strict in the isolated arm**: exactly the manipulation the isolated arm
exists to test. `support_tickets_open` behaves the same way (bounded by
default, strict isolated). Found by reading the generator, not previously
noted: `email` is derived from `name`, a strict pair, so `h1_individual_email`
dithers one half of a strict relationship (a cheap natural detection test).
Seven fields have no link at all (acquisition_channel, phone, address,
refund_rate, avg_resolution_time_hours, has_pending_order,
has_active_subscription): natural negative controls.

**Verification:** 21 relationships checked over 3,000 generated customers (no
failures; observed spend ratio 0.900-1.100 and LTV/spend 0.80-4.00 match the
stated bands); the engine probed on all 25 fields (changes exactly the field
plus the declared propagation, nothing else; declared propagations actually
occur); within-segment independence of the unlinked numeric fields (worst
|rho| about 0.10, threshold 0.20). The verifiers were sabotaged seven ways
(wrong rule, wrong band, wrong LTV band, missing propagation, phantom
propagation, stale field list, a hidden correlation); each was caught.

**Caveat:** direction matters. Strict relations expose every change; bounded,
approximate and soft ones expose a corruption only when it pushes the value
outside the allowed region (support_tickets_open dithered downward never
violates closed >= open). The table says which siblings exist, not that every
corruption is exposed.

**Freeze rule:** after review, record the module's hash here and in the
amendment; never edit the table after seeing which fields produced detections.
Because it covers all 25 dithered fields, the redundancy hypothesis can also
be examined across H1-H3 (exploratory): detection rate by class.

**Open review questions:** (1) Should `soft` count as redundancy? Recommendation:
primary stratification groups {strict, bounded, approximate} against
{soft_only, none}, with soft_only reported separately. (2) The spend triad
lists all three fields as one another's approximate siblings; confirm.
(3) Whether to pre-specify the `none` fields as negative controls.

## Agent file review (business_decision_agent.py, repo hash 07a86297d669948e)

Confirmed: model claude-haiku-4-5-20251001, temperature 0.0, max_tokens 1024;
output keys business_decision / agent_confidence / decision_reasoning /
key_factors; the agent sees everything but record_id (including
customer_segment and three date fields the glossary never describes:
account_created_date, last_purchase_date, next_renewal_date); input_hash is an
md5 of the displayed record. Everything the evaluators assume about the agent
holds.

**Prompt history of the key_factors instruction.** 1a: "list the 2-3 field
names", with a concrete example ["total_spend", "churn_risk_score"]. July 1b
draft: cap kept, placeholder example names. Current repo version: no cap, no
example (the six changed lines are all in the output-format section). This
removes a possible anchor (1a's example named two fields that were also among
its top self-cited fields), but it changes the self-report instrument, so H1's
replication of 1a's self-reported importance is a replication under a changed
instrument: compare rank order, not absolute citation rates (no cap means more
factors per decision). The module docstring lists the differences from 1a
(examples removed, glossary unchanged) but not this one. The pre-specified
within-1b stated-vs-revealed analysis is unaffected. Nothing collected under
the July wording should be pooled with runs under the current wording.

**Observations from reading the file (all verified):**
- `--max_records` is parsed and never used: passing it processes the whole
  file (a cost surprise, not a data problem).
- The glossary states a value range for only four fields: nps_score (0-10),
  email_open_rate, churn_risk_score and fraud_risk_score (0.0-1.0). None is
  stated for total_spend, tenure_months, lifetime_value_estimate, and the
  rest. So "implausible" means different things by field: for churn it
  violates a range the agent was told; for spend and tenure it is
  unrealistic but breaks no stated range. This partly confounds the
  field-by-plausibility interaction and should be a recorded attribute of
  each field (proposal: add `range_stated_in_prompt` to field_redundancy.py,
  derived by parsing the prompt itself and verified like the other claims).
  Consistent with the smoke finding that churn values like 24.9 were
  reformatted, not flagged, despite the stated 0.0-1.0.
- "If certain fields suggest conflicting priorities, weigh them..." is
  inherited from 1a. It concerns priorities, not data validity, but it is the
  one sentence telling the agent that fields can conflict, and the docstring
  says the prompt makes no reference to consistency. Same in every condition,
  so it cannot bias comparisons; it is a caveat on the word "organic" in H5.
- process_file has no try/except and opens the output with "w": no resume. The
  base agent (my copy; the repo's is unverified) retries 3 times with 1-2 s
  backoff and then raises, which aborts the file. At full-run scale a
  sustained rate limit would restart a condition from record 1.

**Open decisions:** (a) honor --max_records and add a resume mode (append,
skipping record_ids already present); (b) add `range_stated_in_prompt` to the
redundancy table; (c) document the key_factors change in the docstring and
the amendment; (d) wording for H5: detection is measured under a prompt that
mentions conflicts between fields. Still unseen: shared/agents/ (base_agent.py,
llm_factory.py); hashes requested.


## H1 and the leading-examples question (decision pending)

**What the 1a prompt contained** (restored in the August session): example
descriptions for each priority level (HIGH: "high-value customers at risk, VIP
customers with issues, ... early churn signals"; LOW: "inactive customers with
low engagement" ...); a concrete output example (decision "HIGH_PRIORITY",
confidence 0.85); and a key_factors instruction capped at "2-3 field names"
with the example ["total_spend", "churn_risk_score"]. key_factors was added to
1a (commit e6e7992) specifically for 1a's H4, the field-importance hypothesis
whose top 5 H1 now uses. 1a_RESULTS.md limitation 7 already names the
priority-level examples as a possible implicit constraint ("a prompt controlled
rerun is planned"): the 1b agent is that rerun, and the 1b docstring records
the principle. I found no transcript discussion of the key_factors lines
specifically; the current wording completes the removal of example values from
the output format.

**Evidence on contamination, from 1a's own results.** 1a's top 5 (identical at
all eight duplication levels): last_purchase_days_ago, churn_risk_score,
nps_score, lifetime_value_estimate, support_tickets_open. The key_factors
example named total_spend (NOT in the top 5: evidence against strong
anchoring) and churn_risk_score (rank 2: ambiguous, since it is also an
obvious driver). The priority-level examples use vocabulary (risk, value,
issues, inactivity) that loosely maps onto four of the five, but those are
also the obvious drivers of any prioritization; the two cannot be separated
from 1a's data. Conclusion: contamination is possible, not demonstrated.

**What H1 depends on.** H1's five individual conditions and Question A use 1a's
list. The revealed-importance measurement (drift when each field is dithered) is
valid however the list was chosen. What the instrument change affects is the
claim that the list is "the agent's self-report": the 1a list was produced
under a prompt with leading examples, the 1b baseline will produce its own
under the clean prompt. That second ranking is free (key_factors is in every
baseline decision).

**Proposal (to pre-specify BEFORE the baseline run):** (1) report the overlap
between 1a's top 5 and the 1b baseline's top 5 and top 8 by citation rank
(rank, not share: with no cap the 1b agent cites more fields), plus a rank
correlation over all fields; (2) a conditional rule: if at most 2 of 1a's top
5 are in the 1b baseline top 5, add up to 2 individual conditions for the
highest-cited 1b fields not already dithered (cost about $5), generated from the
stored canonical_customers.json as check_and_generate_h8b.py already does for
H8b, so the date-of-birth drift cannot affect them; Question A is then reported
on both lists, the 1b list primary. Selection uses only clean-baseline
citations, independent of any drift outcome, so it adds no forking path.
The overlap thresholds are the owner's call. Without the conditional rule,
Question A stands as designed, worded as a replication under a changed
instrument.

## Agent patch and redundancy-table update (done; hashes recorded)

**business_decision_agent.py** (patched; hash e647f56b543c5c67; was 07a86297d669948e).
SYSTEM_PROMPT is byte-identical. Changes: `--max_records` now works (only the
first N input records are considered); new `--resume` (keeps valid decisions
already in the output, sends only missing records to the model, retries
PARSE_ERROR lines and a truncated last line, writes one line per record; refuses,
leaving the file untouched, if any kept decision was made on different input
(input_hash), if the file holds decisions outside the current window, or if
record_ids repeat); overwriting an existing output without --resume now prints
a warning; the summary adds resumed_records, processed_this_session and
session_cost_usd (total_records and total_cost_usd keep their meaning); the
module docstring records the output-format differences from 1a. Tested offline
against the real base agent with the model call faked: 17 checks, including a
regression showing identical decision lines to the original file with no new
flags, crash-then-resume equal to an uninterrupted run, and the CLI flags; the
suite was mutation-tested (5 deliberate bugs, all caught).

**field_redundancy.py** (hash 677461949c0f6ee4; was aa603194c23023cc). New column and
verification: the value range the agent's prompt states per field (nps_score
0-10; email_open_rate, churn_risk_score, fraud_risk_score 0.0-1.0; none for
the rest), parsed from business_decision_agent.py and checked against clean
generated data; 5 sabotage tests all caught. Note: this module reads the agent
file from its own directory.

**Verified shared agent files:** base_agent.py (7b1b8f58...) and
llm_factory.py (a2633dd5...) match my copies, so the retry behavior described
earlier (3 attempts, 1-2 s backoff, then raise) is the repo's actual behavior.

## Amendment checklist: additions

6. Document the differences from 1a's prompt (priority-level examples removed;
   key_factors cap and example removed; output-format example values removed),
   and that key_factors comparisons with 1a are by rank only.
7. H5 wording: detection is measured under a prompt that tells the agent fields
   can conflict ("If certain fields suggest conflicting priorities...").
8. H1: the baseline replication rule above (decision pending).
9. Freeze list to record before the full run (hash each): agent file,
   generator, engine, field_redundancy.py, classifier prompt (v3-be665399de).
10. Date-of-birth reproducibility: guard in generate_dithered_data.py, or the
    procedure (generate once). Conditional conditions must load the stored
    canonical file.

## H1 conditional replication rule: DRAFT, pending approval of the changes marked (*)

Owner decisions so far: write a conditional rule into H1 now; trigger = top-5
overlap of at most 2. Pre-registration holds only once this text is in the
committed amendment BEFORE the 1b baseline starts (record the commit hash).

**Instrument.** key_factors from every baseline decision (5 runs x all
customers). A cited item counts only if it exactly matches a schema field name
(case-insensitive, trimmed); other items are not assigned and the unmatched
share is reported. (*) Citation rate of a field = fraction of decisions citing
it. Rank by rate; ties broken alphabetically; the rate margin between ranks 5
and 6 is reported.

**Always reported.** Top-5 and top-8 overlap with 1a's list (last_purchase_days_ago,
churn_risk_score, nps_score, lifetime_value_estimate, support_tickets_open);
Spearman rank correlation over every field cited in either; (*) a bootstrap over
customers (all of a customer's runs resampled together) giving the distribution
of top-5 overlap and the probability it is at most 2. The bootstrap is context,
not the trigger.

**Trigger: point-estimate top-5 overlap of 0, 1 or 2.**
- Overlap 3, 4 or 5: no change. Question A stands as designed, worded as a
  replication under a changed instrument. A 1-2 field swap is expected variation
  once the cap and the example field names are removed.
- Overlap 0-2: add an individual condition, with the same parameters as the
  existing twelve (15%, drift, correlated, recompute on), generated from the
  stored canonical_customers.json (as check_and_generate_h8b.py does), for
  every field in the 1b top 5 that does not already have one. (*) This is up to
  5 new conditions, not 2 (about $2.30 each at standard pricing, so at most
  about $12; roughly half on Batch). Twelve fields already have a condition and
  are reused for free: the five from 1a, email, is_vip, total_spend,
  tenure_months, avg_resolution_time_hours, refund_rate, payment_failures. A
  field the engine cannot dither (customer_segment, is_at_risk,
  recently_contacted_support, date fields) is skipped and the next eligible
  field takes its place; the skipped fields are reported. (*)
- Question A is then reported on both lists. The 1b list is primary; the 1a list
  is the legacy comparison. (*) For the 1b list the comparison group is the
  existing individual conditions for fields outside the 1b top 5, with a
  sensitivity analysis that also excludes fields ranked 6-8. Same tests as the
  existing plan (group-level Mann-Whitney secondary; pairwise comparisons).

**Wording of the rationale (*).** Low overlap means the 1a list does not
reproduce under the 1b instrument. It does not prove the example names caused
the difference: the instrument changed in several ways at once (priority-level
examples, output example values, the key_factors cap and example, different
customers). Likewise, overlap of 3 or more shows the list survived; it does not
show 1a was uncontaminated. False triggers are cheap and missed triggers weaken
H1, so a threshold at 2 is reasonable.

**Selection is independent of outcomes:** it uses only clean-baseline
citations, never any drift result.

**Why 5 and not 2 (correction to my earlier proposal).** Making the 1b list
primary requires dithering every field on it. An overlap of exactly 2 leaves 3
fields on the 1b top 5 that 1a never covered; a cap of 2 new conditions would
leave the primary analysis incomplete.


## H1 conditional replication rule: APPROVED; tooling built and verified

**Status.** The owner approved the rule as drafted above, including every
starred change (up to 5 new conditions, not 2; measurement definitions with
tie-break, matching and eligibility; the bootstrap as context only; the
comparison group for the 1b list; the softened rationale). The rule is
pre-registered only once its text is in the COMMITTED amendment before the 1b
baseline starts; record that commit hash.

**Tooling** (hashes recorded for the freeze list):
- `h1_baseline_replication.py` (68a4fd3f3475d2ee): counts key_factors citations across the
  five baseline runs, ranks fields, reports top-5/top-8 overlap with 1a, where
  each 1a field landed, a customer-level bootstrap, optional Spearman, applies
  the trigger (point-estimate top-5 overlap <= 2), and lists the conditions to
  add. Refuses a partial baseline. Records the hash of every run file and of
  canonical_customers.json. Warns if dithered-condition decisions already
  exist (the rule is meant to run before any).
- `generate_h1_replication_conditions.py` (7aad6d324afce92e): generates one individual 15%
  condition per entry, cloned from h1_individual_nps_score (only field, seed
  and id differ), from the SAVED canonical customers (refuses if their hash
  changed since the analysis: the date-of-birth hazard), using the main
  generator's own validity check and record-id functions; cross-checks the
  record_id set against an existing condition; never silently overwrites;
  idempotent. Condition ids h1_replication_<field>; seeds 500-524, assigned
  per field so they do not depend on the baseline outcome. Decisions go in
  conditions/<id>/decisions.jsonl, where the evaluators read them.
- `test_h1_replication.py` (7255ec45446f4e14): self-contained regression test (builds its
  own 200-customer dataset in a temp directory; never touches real outputs).

**Verification.** 37 check groups pass, every expected value recomputed
independently with plain loops over exactly-planted citation counts: seven
scenarios (overlap 5, 3, 2 with new fields, 2 fully covered, 2 with
customer_segment/is_at_risk in the raw top 5, 0, and a three-way tie at rank
5/6), the measurement rules, completeness refusals, bootstrap sanity, the CLI,
and twelve generator checks. Ten deliberate bugs (trigger off by one,
reversed tie-break, ineligible fields not skipped, covered fields re-added,
cap of 2, case-sensitive matching, double counting, PARSE_ERROR not excluded,
no canonical-hash check, silent overwrite) were each caught.

**Frozen constants** (in the script; do not edit after the baseline runs):
1a top 5 = last_purchase_days_ago, churn_risk_score, nps_score,
lifetime_value_estimate, support_tickets_open; trigger overlap <= 2; top-5 and
top-8; bootstrap B = 2000, seed 20261009; eligible = the 25 fields the engine
can dither; 12 fields already have an individual 15% condition.

**Sequence at the full run.** Generate everything once -> run the five baseline
runs -> run h1_baseline_replication.py with --canonical -> if it fires, run
generate_h1_replication_conditions.py and the agent on the new conditions ->
continue with the dithered conditions.

**Open items.**
1. Spearman needs 1a's FULL ranking; 1a_RESULTS.md reports only its top 5. If
   1a's decision files exist, one script can produce it (pass --a1a_ranking);
   otherwise the report shows where each 1a field landed in the 1b ranking.
2. evaluate_h1.py has not been extended: Question A on the 1b list (primary
   when the rule fires), with the comparison group of existing individual
   conditions outside the 1b top 5 and the sensitivity analysis excluding
   ranks 6-8, still needs building.
3. Amendment: fold in the rule text, the tooling, and the frozen constants.


## H1 tooling v2: frozen manifest (response to the owner's review; SUPERSEDES the tooling description above)

New hashes: `h1_baseline_replication.py` 2405ec63b09f7375; `generate_h1_replication_conditions.py` ca254aceb1e7eb27;
`test_h1_replication.py` b70e70dde8a1f39c. The output file is now
`h1_replication_manifest.json` (not h1_replication.json); the generator takes
`--manifest`; rule_version h1-replication-v2.

**The four edge cases raised, and the verdicts.**
1. *Deterministic tie-breaking.* Already satisfied, and now demonstrated: ranking
   is citation count DESC, field name ASC, and the sort key is now the integer
   count (not a float rate). A three-way tie at rank 5/6 produced byte-identical
   output under four different PYTHONHASHSEED values, and the suite does the same
   with two seeds. No set or dict iteration order reaches any output.
2. *Frozen manifest.* Adopted, and strengthened. The earlier output held most of
   the requested content but was not frozen: it could be silently regenerated,
   it carried machine-specific paths, and its bootstrap used NumPy's random
   stream, which is not guaranteed stable across NumPy versions. Now: the
   decision-bearing content (input hashes by file NAME, the complete citation
   table including never-cited fields, overlap k, the trigger, the conditions to
   add, and the pre-registered Question A groups) is covered by `decision_sha256`
   (canonical JSON, no paths, no timestamps); the `context` block (bootstrap,
   Spearman, cost estimate, sequencing warning) is outside it; the file is
   write-once (identical re-run leaves it untouched, different content or a
   hand-edited file is refused); the bootstrap uses Python's random.Random with
   exact integer arithmetic; the same inputs give the same hash in a different
   directory tree. The generator verifies the hash, refuses an edited manifest,
   and writes `h1_replication_generated.json` (manifest hash, canonical hash,
   hash of every generated file). **evaluate_h1.py must verify `decision_sha256`
   and consume `question_a_groups`; it must never re-evaluate the trigger.**
3. *"Ineligible" definition.* Two of three parts were already true; one is
   rejected. Eligible = a field the engine can dither (the 25); the three
   protected fields, dates, dob and preferred_categories are skipped, and the
   skipped fields are reported (tested). The agent-visible schema excludes
   record_id and every metadata field, so customer_id, record_id and _dither_*
   can only ever be unmatched citations (tested). REJECTED: skipping fields
   already in 1a's top 5. That would delete a surviving 1a field from the 1b list
   and defeat the replication it measures. Eligibility (can it be dithered) is
   deliberately separate from coverage (does it already have a condition): a
   covered field stays on the 1b list and is reused at no cost.
   Practical check: all 13 eligible fields that lack a condition pass the
   generator's validity check, so nothing the rule can select will fail at
   generation.
4. *1a ranking without a full Spearman.* Agreed. The report now shows
   "1a #k field -> 1b #m" for each of 1a's five, with the 1b rate. Spearman is
   optional context. Optional extra: 1a's rank and rate for total_spend and
   churn_risk_score, the two fields its key_factors example named, are the
   cleanest available test of anchoring, if 1a's decision files can be found.

**Question A groups, as encoded in the manifest.** Legacy 1a list: top = 1a's
five; comparison = exactly the amendment's six (email, is_vip, total_spend,
tenure_months, avg_resolution_time_hours, refund_rate). If the trigger fires:
1b list (primary): top = the eligible 1b top 5; comparison = every existing
individual condition outside that list; sensitivity = that comparison without
fields at raw 1b ranks 6-8. Note: the 1b comparison group includes
payment_failures and any 1a field that dropped out, neither of which is in the
amendment's six; that follows from the approved rule ("existing individual
conditions for fields outside the 1b top 5") and should be stated in the
amendment.

**Verification.** 52 check groups, expected values recomputed independently,
now including the bootstrap replicated in plain Python (exact match on 5
scenarios), the manifest's integrity, write-once and machine independence, the
groups, and the generator's record. 19 deliberate bugs, all caught. The mutation
testing exposed one weakness in my own suite: every scenario had margins so wide
that the bootstrap gave the same answer under any seed, so a changed seed went
unnoticed. A near-tie scenario (rank 5 vs rank 6) now makes the bootstrap vary,
and a changed seed or wrong resampling is caught.

**Open items.** (1) Optional: locate 1a's decision files for the
total_spend / churn_risk_score anchoring look. (2) evaluate_h1.py still to be
built against the manifest. (3) Amendment: the rule text, the manifest, the
group definitions (including the payment_failures note), the frozen constants
(now including LEGACY_COMPARISON).

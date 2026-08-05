# 1b Design Amendment — Full Hypothesis Review (H1–H8b)

**Status:** Pre-registration amendment. Written before evaluator code, before any
agent runs against dither conditions beyond initial smoke tests. This document
amends `1b_DESIGN.md` rather than replacing it. The original six hypotheses
(H1–H6) stand in spirit, their condition sets are restructured below following
a full stress-test pass, and H7–H8b are added as new hypotheses.

**Why an amendment rather than a silent edit:** `1b_DESIGN.md` is a
pre-registration contract with ourselves. Changing it after the fact without
a record would defeat the purpose. This document exists so anyone reading the
repo can see exactly what changed, when, and why before a single dither
condition was run through the agent.

**How to read this document:** each hypothesis section below follows the same
pattern including the original design, what stress-testing surfaced, the resulting
decision, and the final condition set. Several decisions are deliberately left
as *named placeholders* (a decision rule locked now, an exact value confirmed
once real data exists) rather than guessed numbers. This mirrors how the 1a experiment
handled uncertainty and keeps the document honest about what we actually know
versus what we're choosing to defer.

---

## A Note on Prompt Continuity

Before the hypothesis-by-hypothesis review, a full re-read of
`business_decision_agent_1b.py`'s system prompt was necessary, prompted specifically by H5's
detection-awareness concerns, surfaced one line worth flagging that applies
across multiple hypotheses rather than just one.

The prompt (inherited verbatim from experiment 1a) reads: *"IMPORTANT: Base your decision
on the DATA provided, not on assumptions. If certain fields suggest
conflicting priorities, weigh them based on their relative importance to
business outcomes."*

This line was inert in 1a, which never manufactured genuine internal
contradictions for the agent to resolve. In 1b, H3's uncorrelated dither
condition does exactly that, and this line gives the agent standing
instruction for handling it. This does not tell the agent to *notice*
inconsistency, only how to *weigh* conflicting signals once one is present,
so it does not appear to compromise H5's detection-awareness measurement. But
it does mean any "resolution behavior" observed in H3's uncorrelated arm is at
least partly prompted rather than fully emergent, and this must be disclosed
in H3's write-up.

**"Business outcomes" is never defined anywhere in the prompt.** Revenue
retention, support cost minimization, satisfaction, and long-term value could
each imply different resolutions to a genuine conflict, and the agent is never
told which one governs. This ambiguity is being left **unpatched**, fixing it
would introduce a second deliberate prompt divergence from 1a beyond removing
the illustrative examples, and unlike that removal (closing a documented 1a
limitation), defining "business outcomes" would add information the agent
never had in 1a, creating a fresh confound rather than resolving a bias. It is
documented instead as a named limitation: the ambiguity existed in 1a but was
inert there; H3's uncorrelated arm is the first place in the series that
actually exercises it.

---

## A Note on Protected Fields and Boolean Support (Engine Extension)

While building out H1's category-level conditions, an attempt to cross-
reference every category against the dither engine's actual field metadata
surfaced a real blocker, not just a documentation gap: three fields
(`is_at_risk`, `recently_contacted_support`, `customer_segment`) and an
entire field type (booleans: `is_vip`, `has_active_subscription`,
`has_pending_order`) were not supported by the engine at all. The engine's
own validation would have raised `ValueError` on several of the amendment's
own category-level conditions the moment anyone tried to run them.
In particular, `h1_category_account_status` could not have run at all, since
none of its five originally-listed fields were dither-capable.

This was resolved as three separate decisions rather than one bundled fix,
since the three unsupported items turned out to be three structurally
different problems wearing the same symptom.

### Boolean support: built

`is_vip`, `has_active_subscription`, and `has_pending_order` now dither via
**flip probability**, not percentage magnitude. `magnitude` for a boolean
field IS the probability of flip, directly. 0.15 magnitude means a 15%
chance any given customer's boolean gets flipped. This preserves the
principle held everywhere else in the engine: "magnitude" means the same
thing (how often/severely this customer's data gets corrupted) regardless of
field type, rather than becoming three unrelated concepts wearing one
parameter name. Verified against 1,000 synthetic customers at 0.15
magnitude: 13.5% observed flip rate, well within expected sampling noise of
the 15% target.

**A finding worth noting, not a flaw to correct**: Boolean fields with skewed base rates produce dramatic population level swings under uniform flip probability dithering, purely as a mathematical consequence of rarity, not an engine defect. `is_vip` in the ground truth data has only a 4.3% True rate (43/1,000 customers, gated to the high_value segment). At 15% flip magnitude, the True population nearly quadruples to 170 customers (131 new False→True flips against only 4 True→False flips). This mirrors a real phenomenon: in any production system, a small uniform per-record corruption rate will always inflate a rare positive class dramatically in relative terms, precisely because there are so many more negatives available to flip into it than positives available to flip out. Deliberately not corrected by scaling flip probability to a field's base rate, doing so would reintroduce the same per-field inconsistent magnitude problem we specifically avoided when rejecting 1/N-scaled flip probability for categorical fields with different option counts. The asymmetry is preserved as a genuine, measurable phenomenon rather than suppressed. See the new attribution analysis below for how it's isolated rather than allowed to contaminate interpretation.

Boolean fields ignore the `dither_type` (drift vs. entry_error) distinction —
there is no meaningful difference between a boolean "decaying" over time
versus an automation artifact flipping it, both route to the same flip logic.

**New analysis: per-field and per-direction attribution within multi-field conditions**

The `is_vip` base rate finding above surfaced a gap that applies well beyond one field: H1's original four sub-questions named Question C ("within a category, does one field carry disproportionate weight, or is the category's effect evenly distributed?") but no clean mechanism existed to actually answer it for any category. Individual field testing only exists for fields that happen to also be H4 top-5 or H2 fields, not systematically for every category.

This is resolved as a **free analysis**, not a new condition. Every multi-field condition already dithers its fields uncorrelated (each field independently rolls its own perturbation), and the `_dither_fields` metadata already captured per customer records exactly which specific field(s) were actually touched for that customer. The evaluator will use this to cross-tabulate drift rate by which specific field(s) changed within every multi-field condition, answering Question C for all six H1 categories, h8a pairs, and h7 breadth conditions at zero additional API cost.

**For boolean fields specifically, this attribution is extended to include direction** (False→True vs. True→False), not just whether the field changed. This directly answers the more interesting question the `is_vip` finding actually raises: does flipping a non-VIP customer to VIP status move the agent's decision differently than flipping a real VIP down to non-VIP? That's a legitimate behavioral question about the agent, not a confound to control away, and it's answerable from data already being generated.

### Derived and upstream fields: protected, not built

Experiment review revealed two structurally different reasons a field might need to be excluded from
direct dithering, both now enforced by the engine itself via an explicit
`PROTECTED_FIELDS` registry that raises a clear, explanatory error rather
than silently doing something incoherent:

**Derived fields** (`is_at_risk`, `recently_contacted_support`) are computed
*from* other fields after generation, and re-derived after any dither via
`_recompute_derived()`. Dithering a derived field directly would create an
internally incoherent record (e.g. `is_at_risk=True` with a
`churn_risk_score` that doesn't support it) with no defined semantics, and
no hypothesis in this experiment is designed to study derived-field mismatch
specifically. Adding one was considered and declined. It would add
complexity without adding insight proportionate to that complexity.

**Upstream fields** (`customer_segment`) are used *during* generation to
condition the sampling distributions of other fields. i.e. a customer's segment
determines the ranges `total_spend`, `churn_risk_score`, etc. are drawn
from. Dithering `customer_segment` directly after the fact, changing the
label while leaving the numeric profile exactly as originally generated,
breaks the segment/profile relationship **top-down**, with no defined
semantics for what that would even represent. This is excluded entirely.

**The bottom-up version of this same idea, however, is not excluded. It's
already happening, and worth measuring explicitly.** Dithering a customer's
numeric fields (via H2, H4, H7, or any other numeric-field condition) far
enough that they no longer match the typical range their *original* segment
assignment would predict is a legitimate and interesting effect that falls
directly out of dithering already being done. It requires no new dither
mechanism, only a new **analysis lens**.

### New cross-cutting analysis: segment/profile mismatch

For any condition that dithers numeric fields, the evaluator will check whether a customer's dithered profile still falls within their original `customer_segment`'s typical range and flag customers whose profile has effectively "left" their assigned segment's normal territory. **Computed empirically from the 1,000-customer ground truth dataset itself** (e.g. 5th–95th percentile per field per segment, grouped directly from `canonical_customers.json`), not by extracting the generator's internal literal bounds, which would require a refactor and would create a second, driftable source of truth. This is a free enrichment across H1, H2, H4, and H7, anywhere a numeric field gets dithered, not a new hypothesis or new engine mechanism.

**The same concept extends to boolean fields.** `is_vip=True` while `customer_segment != high_value` after dithering is a categorical instance of the identical phenomenon. Specifically, a dithered attribute no longer matching what the customer's protected, undithered segment assignment would predict. Folded into this same lens rather than treated as a separate concept.

### `preferred_categories`: a distinct, smaller deferral

Unlike booleans or single-select categoricals, `preferred_categories` is a
variable-length *subset* (1–4 categories sampled from a pool of 10), not a
single value among options. "Flip probability" and "plausibility tier" both
assume a clean single-value-to-single-value transition; neither concept maps
cleanly onto "corrupt a subset" as swapping one entry, adding or removing an
entry, and resampling the whole list are all meaningfully different
perturbation severities, not implementation variants of one idea. This is
judged a genuinely separate design problem from H9's boolean/categorical
magnitude question (which this amendment now resolves for booleans and
single-select categoricals) and is deferred on its own, not folded into H9's
scope. See Deferred Items below.

---

## H1 — Field Importance Validation (Restructured)

### The problem with the original 7 conditions

The original H1 design tested 5 individual top-5 fields plus two aggregate
conditions labeled "identity" and "behavioral." On review, "identity" mapped
cleanly to the prompt's own Identity & Contact section but "behavioral" was
a residual bucket assembled from five fields spanning three different
conceptual categories with no principled reason for that specific grouping
beyond "not in the top 5."

### The four sub-questions H1 is actually asking

- **Question A:** Does the agent reported top-5 field set (from experiment 1a, H4) produce more decision
  drift when dithered than fields the agent did not self-report as important?
- **Question B:** Does dithering behave differently depending on which
  conceptual category of data it hits?
- **Question C:** Within a category, does one field carry disproportionate
  weight, or is the category's effect evenly distributed?
- **Question D:** Does identity data, assumed decision-irrelevant, actually
  behave as inert as assumed?

### Restructured condition set (12 conditions)

**Five individual field conditions** (unchanged) — each 1a H4 top-5 field
dithered alone at 15% magnitude:

1. `h1_individual_last_purchase_days_ago`
2. `h1_individual_churn_risk_score`
3. `h1_individual_nps_score`
4. `h1_individual_lifetime_value_estimate`
5. `h1_individual_support_tickets_open`

**Six category-level conditions** — every field in one prompt section
dithered together, matched 15% magnitude, uncorrelated. **Field lists below
reflect engine-verified corrections** (see "A Note on Protected Fields and
Boolean Support" above for the full reasoning behind each exclusion)

6. `h1_category_identity` — name, email, phone, address. **`dob` excluded**
   as it is not currently dither-capable in the engine (no defined percentage-
   magnitude or plausibility-tier behavior for a date-of-birth field).
   This is an unflagged gap, not a protected field, worth a future engine extension
   if birth-date sensitivity becomes independently interesting.
7. `h1_category_purchase_behavior` — total_purchases, total_spend,
   avg_order_value, purchase_frequency_days, last_purchase_days_ago,
   lifetime_value_estimate
8. `h1_category_engagement` — nps_score, email_open_rate,
   last_login_days_ago, support_tickets_open, support_tickets_closed,
   avg_resolution_time_hours
9. `h1_category_risk_factors` — churn_risk_score, payment_failures,
   fraud_risk_score, refund_rate
10. `h1_category_segmentation` — acquisition_channel, tenure_months.
    **`customer_segment` excluded**. (protected, upstream field. see
    above). **`preferred_categories` excluded** (list-valued field,
    deferred separately see above).
11. `h1_category_account_status` — is_vip, has_active_subscription,
    has_pending_order. **`is_at_risk` and `recently_contacted_support`
    excluded** (protected, derived fields, see above). This category
    condition is now correctly scoped to the three fields that are
    genuinely independent account-status attributes rather than
    downstream consequences of fields living in other categories.

**One distributed condition**: one field per category, deliberately
excluding the H4 top-5, matched 15% magnitude, uncorrelated:

12. `h1_distributed` — email, avg_order_value, last_login_days_ago,
    refund_rate, acquisition_channel, has_pending_order

**Scope note:** `h1_distributed` is investigatory, not exhaustive. A null or
positive result scopes deeper combinatorial work into Phase 2 rather than
being treated as conclusive alone.

### Free analyses from H1

**Category impact ranking**: the six category conditions produce a ranked
list of which conceptual data category matters most when dithered, analogous
to 1a's decision-cliff framing but for category type rather than duplication
volume.

**Stated vs. revealed field importance**: every baseline decision already
returns `key_factors`. Comparing how often each field is self-cited (stated
importance) against actual drift rate when that field is dithered (revealed
importance, from the five individual conditions) tests whether the agent's
self-report about its own reasoning predicts what actually changes its
decisions. Zero additional API calls required.

---

## H2 — Magnitude Effects (Restructured)

### The problem with the original design

Testing magnitude on a single field (`churn_risk_score`) cannot distinguish
"how magnitude affects drift in general" from "how magnitude affects drift for
this one field", especially since that field also anchors H1, H4, and H8b.

### Field selection: deliberate spread

1. **`churn_risk_score`**: H4 top-5 anchor, kept for cross-hypothesis
   comparability
2. **`total_spend`**: not itself top-5, but closely related to
   `lifetime_value_estimate` (which is). This tests generalization to
   "important but not self-reported" fields
3. **`tenure_months`**: expected lower decision weight; genuine
   low-importance comparison point

### Field independence: examined, does not compromise H2

`churn_risk_score` and `total_spend` are correlated in the base generator
through shared segment-conditioned distributions, not through a direct
formula. This does **not** compromise H2, because H2 dithers one field at a
time so population-level correlation does not leak into a measurement that
only ever perturbs one field per customer per condition. (This correlation
*would* matter if dithering both fields simultaneously, which is what H8b
tests, not H2.)

**Analysis enrichment:** the evaluator will compute drift rate *within
customer_segment* for `total_spend` and `churn_risk_score`, to separate
"the field intrinsically matters" from "the field is a proxy for segment
membership."

### Magnitude ladder: 4 levels, 75% explicitly parked

5%, 15%, 40%, 100%. A fifth level (75%, closing the 40–100 gap) was
considered and **parked, not rejected**. Tripling the field count already
triples the condition count from 4 to 12; a 5th level would take it to 15 for
a value whose benefit is speculative. If the drift-vs-magnitude curve shows a
non-monotonic shape between 40% and 100% once real data exists, an
intermediate level may be added as an explicitly-flagged, post-hoc exploratory
follow-up, never folded into the pre-registered claim.

**5% retained deliberately**: most likely to surface a decision-cliff-style
finding analogous to experiment 1a's discovery that even 10% duplication produced
measurable inconsistency.

### Sample size: full 1,000 customers, not a subset

Subsampling within the same 1,000-customer draw does not guard against
population-specific artifacts. Those 1,000 customers all come from one
generator run with one seed regardless of how they're subsampled. What would
actually test that is a wholly separate ground truth population from a
different seed. **Decision:** full 1,000 for every H2 condition; seed
robustness deferred to Phase 2 as a distinct, explicitly-named replication
study (see Deferred Items below).

### Condition set (12 total)

13–16. `h2_churn_risk_score_mag{5,15,40,100}pct`
17–20. `h2_total_spend_mag{5,15,40,100}pct`
21–24. `h2_tenure_months_mag{5,15,40,100}pct`

---

## H3 — Internal Consistency (Restructured)

### Original design

3 field pairs + 1 triplet, each tested correlated (fields drift together,
staying internally plausible) vs. uncorrelated (fields drift independently,
breaking expected correlations), at matched 15% magnitude:

- Pair 1: churn_risk_score + last_purchase_days_ago
- Pair 2: total_spend + lifetime_value_estimate
- Pair 3: support_tickets_open + avg_resolution_time_hours
- Triplet: total_spend + last_purchase_days_ago + support_tickets_open

### What stress-testing surfaced

**Granularity of the outcome measure.** Due to a limited number of decision buckets,
binary drift (did the decision bucket change) cannot distinguish a confident 
correlated-arm shift from a visibly conflicted uncorrelated-arm shift landing on 
the same bucket. **Fix:** report three things per condition, not just the decision: 
(1) binary drift rate, (2) mean `agent_confidence` within customers who drifted, 
correlated vs. uncorrelated, (3) Jaccard similarity between dithered and baseline 
reasoning text, as a coherence signal (shared machinery with H5's secondary metric).

**An uncorrelated reference baseline.** Rather than treating "correlated vs.
uncorrelated" as the only axis, a genuinely unrelated field pair, one with no
real-world relationship and no shared segment-conditioning, gives a
reference point: "this is what dithering unrelated things looks like." Chosen:
**`avg_resolution_time_hours` + `refund_rate`**. Neither is segment
conditioned, neither derives from the other. The three genuinely-correlated
pairs' drift rates are read *against* this reference rather than in isolation.

**Directionality (the 2x2).** "Uncorrelated" as originally scoped didn't
distinguish (field A changed, B didn't) from (B changed, A didn't) from (both
changed). This matters if one field carries all the effect and the other is
inert, that's a different finding than genuine joint sensitivity. Checked
against existing conditions before assuming new ones were needed:

- **Pair 1** (churn_risk_score + last_purchase_days_ago): both individual
  conditions already exist in H1 so this is **free**, 2x2 complete at no added cost
- **Pair 2** (total_spend + lifetime_value_estimate): both individual
  conditions already exist (H2, H1) so also **free**
- **Pair 3** (support_tickets_open + avg_resolution_time_hours):
  support_tickets_open exists in H1; avg_resolution_time_hours does not exist
  anywhere so **one new condition required**
- **Reference pair** (avg_resolution_time_hours + refund_rate): neither
  exists individually so **two new conditions required**
- **Triplet:** full 2x2x2 factorial would require 6 additional pairwise
  conditions for a single triplet. **Declined** for cost purposes. Triplet remains
  correlated/uncorrelated only. Full factorial breakdown on 3+ field
  combinations explicitly deferred to Phase 2.

### Condition set (11 total)

25–26. `h3_pair1_churn_purchase_{correlated,uncorrelated}`
27–28. `h3_pair2_spend_ltv_{correlated,uncorrelated}`
29–30. `h3_pair3_support_{correlated,uncorrelated}`
31–32. `h3_triplet_{correlated,uncorrelated}`
33. `h3_individual_avg_resolution_time_hours` (fills Pair 3's 2x2)
34. `h3_individual_refund_rate` (fills reference pair's 2x2)
35. `h3_reference_uncorrelated` — avg_resolution_time_hours + refund_rate,
    dithered together, uncorrelated

---

## H4 — Dither Type Effects (Restructured)

### The problem with the original design

Testing dither type (`drift` vs. `entry_error`) on `churn_risk_score` alone
compounds two issues: the single-field problem already identified in H2, and
a growing perception risk. `churn_risk_score` was becoming the anchor field
for H1, H2, H4, and H8b simultaneously, raising a fair question about whether
findings reflect a general pattern or a spotlight effect on one field.

### The fix: reuse H2's field trio

Both concerns resolve with the same change. Specifically we test dither type across all
three fields already established in H2 (`churn_risk_score`, `total_spend`,
`tenure_months`) rather than one. This gives the single-field robustness
check H2 needed anyway, and de-centers `churn_risk_score` from being the sole
carrier of the finding.

### A third dither type considered, and deferred

Discussion of whether "entry_error" (currently modeling automation
artifacts: unit conversion, format mismatch, default persistence,
truncation) needed splitting into finer sub-mechanisms led to a genuinely new
question: should a **third, human-origin error type** exist, distinct from
both `drift` (passive time-decay) and `entry_error` (passive automated
transformation)?

The original `1b_DESIGN.md` explicitly excluded character-level typos as an
unrealistic model of modern data entry as most enterprise data entry is
automated rather than manually keyed. Revisiting that exclusion outright was
considered and **rejected** as a quiet reversal of a founding methodological
decision. Instead, if a human-origin type is built, it should model a
**discrete, deliberate human action that happened to be wrong** such as a rep
transposing digits while manually re-keying a value, or selecting the wrong
option from a dropdown during manual account setup as opposed to a character
substitution typo. This preserves the original reasoning (automation
dominates modern data entry) while still allowing for the moments where a
human does act directly on a record.

**Deferred, not built.** This is a new dither mechanism requiring engine
changes, comparable in scope to H9. Named and reasoned through here; build
decision revisited after the remaining hypothesis review and H4's two-type
results are in hand.

### Condition set (6 total)

36–38. `h4_{churn_risk_score,total_spend,tenure_months}_drift`
39–41. `h4_{churn_risk_score,total_spend,tenure_months}_entry_error`

(Drift-type conditions for these three fields overlap with H2's 15%-magnitude
conditions where applicable, to be confirmed at build time whether H4 can
reuse H2's existing 15% drift conditions directly rather than regenerating
them, which would reduce this to 3 new conditions instead of 6.)

---

## H5 — Detection Awareness (Fully Specified)

### Original design

Cross-cutting measurement across all conditions. Primary: keyword search for
detection language in `decision_reasoning`. Secondary: Jaccard similarity
against baseline reasoning.

### What stress-testing resolved

**Keyword list vs. LLM judge.** A second, independent evaluation agent was
considered and **rejected** as the primary mechanism. It would roughly
double API cost and introduce an unauditable meta-evaluation problem (how do
we know the judge's calls are reliable?) in exchange for uncertain recall
gains over a well-constructed keyword list.

**Hybrid approach adopted instead:** a frozen, broad, pre-registered keyword
list as the primary metric, paired with a **manual audit** after the real
run, a random sample (~150–200) of zero-hit reasoning texts reviewed by hand
to estimate the list's false-negative rate. This produces an honest,
reportable number ("estimated N% miss rate") without contaminating the frozen
metric or introducing a second non-deterministic system.

**Regex over plain string matching.** Plain keyword matching would miss
inflectional variants ("conflict" vs. "conflicts" vs. "conflicting"). Light,
fully transparent regex with word-boundary anchors closes this gap without
introducing a stemming library dependency that's harder to audit by reading
the code directly.

### The frozen keyword list (final, 17 patterns)

**Direct inconsistency language:**
```
\binconsisten(?:t|cy|cies)\b
\bdoes(?:n't| not) match\b
\bcontradict(?:s|ion|ing|ed)?\b
\bconflict(?:s|ing)?\b
```

**Plausibility/surprise language:**
```
\bunusual\b
\batypical\b
\bimplausible\b
\bseems? off\b
\bdoes(?:n't| not) add up\b
\bodd\b
\bstrange\b
\bsurpris(?:ing|e|ed)\b
\banomal(?:y|ies|ous)\b
```

**Doubt/verification language:**
```
\b(?:hard|difficult) to reconcile\b
\bquestionable\b
\bsuspicious\b
\bseems? wrong\b
\bappears? incorrect\b
\bmay be (?:an )?error\b
\b(?:possible|likely|apparent) error\b
\bdata error\b
\bmistake in (?:the )?data\b
```

**Explicit data-quality language:**
```
\bdata quality\b
\bdata issue\b
\bdata problem\b
```

**"Uncertain about" considered and removed.** Every other pattern is
data-forward, the data itself is the grammatical object of the doubt.
"Uncertain about" is object-ambiguous, equally readable as uncertainty
about the data or uncertainty about the decision itself. Since
`agent_confidence` and the H6 stability classification already measure
decision-level uncertainty directly and numerically, including this phrase
risked inflating H5's "detected a data issue" count with ordinary
boundary-customer hedging that has nothing to do with the data. Removed
outright rather than routed to manual audit, since this is a false-positive
risk baked into the phrase itself, not a false-negative recall gap auditing
could catch.

### Reporting structure

Not a single number. Cross-tabbed three ways per condition:
1. **Detected AND decision/confidence changed**: genuine signal-linked
   detection
2. **Detected, no behavioral change**: noticed but didn't act on it
3. **Not detected**: the blind spot, as found in experiment 1a

---

## H6 — Boundary Customer Vulnerability (Fully Specified)

### Original design

Cross-cutting: uses the `stable` / `lightly_boundary` / `deeply_boundary` 
classification from `aggregate_baseline.py`, broken out per condition.

**Terminology clarification:** "tiers" in H6 refers to this stability 
classification (how consistently the agent decided across the primary 
5 baseline runs), not to the HIGH/MEDIUM/LOW decision itself. A customer 
can be `stable` and always land on any of the three priority levels. 
Stability is about consistency of decision-making, independent of which decision was made.

### What stress-testing resolved

**Multiple comparisons.** 48 conditions × 3 tiers is up to 144 individual rate calculations. 
Declaring any single cell "significant" risks chance inflation. 

**Resolution:** the headline finding is whether the *ordering* (deeply_boundary 
drift rate > lightly_boundary > stable) holds *consistently across most of the 
48 conditions*, a repeated pattern is hard to get by chance. 
A single cell in isolation is not strong evidence on its own.

**"Disproportionate" quantified.** Defined as the ratio of drift rates between tiers, 
computed per condition, with the **median ratio across all 48 conditions** as the 
headline number rather than any single condition's ratio. Fisher's exact test 
(better suited than chi-square for small cells) used as a per-condition check. 
The overall claim rests on consistency of the median ratio and how often the ordering holds, 
not on per-cell statistical significance.

**Small boundary-tier sample size, and what to do about it.** A 3/2 split (deeply_boundary), 
or even a 4/1 split (lightly_boundary), from only 5 draws is a wide uncertainty read. 
Checking the actual Wilson interval width for every possible n=5 outcome confirms this 
is worse than it first appears: a 4-1 split has a width of 58.8 percentage points, 
barely tighter than 3-2's 65.2pp, and both sit far above our own ±15pp convergence bar. 
Even a perfectly consistent 5-0 ("stable") outcome has a width of 43.4pp. The discrete 
count based tier labels are a cheap triage heuristic, not a statistically precise partition.
A customer landing in `lightly_boundary` by chance may be nearly as genuinely uncertain 
as one landing in `deeply_boundary`.

**Resolution: expand baseline runs for both `deeply_boundary` and `lightly_boundary` customers**
Not `stable` (for a stable customer every observed run already agrees, so the practical reference 
decision is unambiguous regardless of abstract statistical uncertainty in the "true" rate), 
and not the ground truth majority vote used everywhere else in the experiment, which stays 
locked at the original 5 runs for comparability across all customers and conditions.

Runs are added adaptively, not to a fixed count: batches of 10 additional runs, with a 
Wilson-score confidence interval computed on the plurality proportion after each batch, 
recomputed fresh from *all* accumulated runs every time (never locked onto whichever decision 
led first. A customer's leading decision can and does flip between batches as more data 
comes in). Convergence is declared once the interval width is ≤30 percentage points (±15), 
with a hard cap at 60 total runs regardless of convergence. Non-convergence at the cap is 
itself reported as a finding, "this customer's true tendency could not be resolved to our 
precision bar even after 60 runs", not silently treated as resolved. Empirically, even a 
theoretical perfect 50/50 customer converges comfortably before the cap (24.5pp width at n=60), 
so a "did not converge" outcome should be rare rather than a common fallback.

**What this refined data is for, and what it deliberately is not for.** The primary 5-run vote 
remains the fixed, uniform yardstick every H1-H8b comparison measures against, deliberately. A `stable` 
customer (5 runs) and a boundary customer (up to 60 runs) are compared on equal computational 
footing regardless of how much extra effort went into refining the latter's estimate. Refined 
data never gets a vote on what counts as ground truth. It is a magnifying glass on the decision, 
not a replacement for it. Three intended uses:

1. **Primary-vote reliability disclosure**: The rate at which a boundary customer's refined 
plurality (after convergence or hitting the cap) agrees or disagrees with their original 
5-run majority vote, honestly reported even if that disagreement rate turns out to be high. 
A meaningfully high mismatch rate is itself an important, disclosed limitation on how much 
weight the primary baseline deserves.

2. **A continuous enrichment to H6**: Alongside the discrete 3-tier stability label, drift 
rate can be reported against each boundary customer's refined plurality *rate* as a 
continuous measure, a sharper and more statistically grounded version of the boundary 
vulnerability finding than the coarse tier label alone provides.

3. **A standalone "confidently wrong, round 2" finding**: Where a customer's primary run 
reported high self-reported confidence on the decision that became their majority vote, 
but the refined data reveals their true tendency is actually close to a genuine toss-up. 
A direct extension of experiment 1a's "confidently wrong" theme, surfaced purely from 
repeated sampling instability, with no dithering involved at all.

**General fragility vs. field-specific sensitivity.** The most important open question: are 
boundary customers vulnerable to dithering *in general*, or specifically to the field that 
made them boundary in the first place? This distinguishes "boundary customers are fundamentally 
fragile decision subjects" from "boundary customers are predictably sensitive to their one 
borderline signal." **Resolved as a free cross-tabulation**: H1's category level drift rankings, 
cross referenced against H6's tier based drift breakdown. If boundary customers show elevated 
drift even under conditions dithering categories H1 predicts are low-importance 
(e.g. `h1_category_identity`), that indicates general fragility. If elevated drift only 
appears under already important fields/categories, that's the more mundane predictable 
sensitivity story. No new conditions required.

**Confidence without bucket change enrichment**, same logic as H3: a stable customer's confidence 
may barely move under dither; a boundary customer's confidence may swing hard without ever 
crossing a decision bucket. Reported alongside binary drift, not instead of it.

### No new conditions

H6 uses the existing 48 conditions plus the boundary-tier run expansion described above (a diagnostic addition covering both `deeply_boundary` and `lightly_boundary`, separate from primary condition generation).



---

## H7 — Breadth Effects (New)

### Hypothesis

Decision drift will not scale linearly with the number of fields dithered
simultaneously. A working prediction that drift increases with breadth up
to a point, after which either a dominant remaining signal stabilizes the
decision, or accumulated conflicting signals produce something closer to
noise than directional shift.

### Design: fixed accumulation ladder

Each step adds fields to the previous one, producing a genuine accumulation
curve rather than unrelated combinations:

42. `h7_breadth_1field`: churn_risk_score only
43. `h7_breadth_3fields`: + last_purchase_days_ago, nps_score
44. `h7_breadth_6fields`: + lifetime_value_estimate,
    support_tickets_open, total_spend
45. `h7_breadth_all`: all H4 top-5 plus additional fields (final list
    confirmed at build time; target 10–12 fields spanning multiple
    categories)

All conditions matched 15% per-field magnitude, uncorrelated, full 1,000
customers.

---

## H8a — Category-Level Interaction (New)

### Hypothesis

Some category pairs will produce super-additive decision drift when
dithered together, greater than their individual effects predict, while
others remain purely additive or show no interaction.

### What stress-testing added

**A precise definition of "additive."** Naive summation of drift rates can
exceed 100% and doesn't reflect what "no interaction" should actually look
like statistically. **Adopted formula:** combined_drift ≈ rate_A + rate_B −
(rate_A × rate_B). This Inclusion-Exclusion Principle is the standard 
way of combining two independent probabilities of "at least one thing happened." 
Super-additive = combined result exceeds this prediction; sub-additive = falls short of it.

**Overlap with H8b and other hypotheses acknowledged.** Pair 2 (Purchase
Behavior + Risk Factors) touches fields already carrying weight in H2 and
H4. This pair and H8b examine the same territory at different granularities, 
not independent confirmations of each other, and the write-up must say so.

**Negative-control surprise handling.** If Pair 1 (Identity + Account
Status, predicted null) shows a *meaningful* interaction effect, that is
treated as a headline finding in its own right. An unexpected result on a
negative control is one of the more interesting things this experiment could
produce, not a footnote to a failed prediction.

### Conditions (2 new. "alone" comparisons already exist in H1)

46. `h8a_pair1_identity_account_status`
47. `h8a_pair2_purchase_risk`

---

## H8b — Field-Level Interaction (New, Conditional on H8a)

### Hypothesis

If H8a's Pair 2 shows category-level amplification, is it concentrated in
specific fields, or distributed evenly across every field in both
categories?

### What stress-testing changed

The original plan guessed a specific pair (`churn_risk_score` +
`total_spend`) to test. This was identified as a real blind spot. Pair 2
spans 10 total fields across both categories, and if the true amplification
driver is a different pair entirely (e.g. `fraud_risk_score` +
`avg_order_value`), the original guess would produce a false "no field-level
effect" conclusion purely from picking the wrong fields, not from the
absence of a real effect.

### The decision rule (locked now, outcome determined later)

This mirrors the `deeply_boundary` run-count approach, lock the *procedure*
before the data exists, not the *outcome*:

1. Run `h8a_pair2_purchase_risk` as designed.
2. Using the additive-baseline formula above, check whether combined drift
   exceeds the additive prediction by a pre-committed threshold (proposed:
   more than 20% relative excess).
3. **If no meaningful excess:** H8b does not run as a new condition. The
   finding is reported directly, no category-level interaction detected,
   therefore no field-level tracing was necessary. This is itself a complete
   and informative result.
4. **If meaningful excess found:** pull individual-field drift rates already
   available from H1 and H2 for every field in Purchase Behavior and Risk
   Factors, rank them, and take the top 2 by individual drift rate, rather
   than the original guessed pair, as the actual H8b condition.

**Sequencing dependency:** H8b cannot be generated in the same pass as the
other 47 conditions. It is the only condition in the 1b pipeline whose
existence depends on an earlier condition's analyzed result rather than being
generated upfront.

48. `h8b_[fields_determined_by_rule_above]`: generated only if the H8a
    threshold is met. Field pair determined mechanically per the rule above,
    not by pre-selection

---

## Deferred to Phase 2 — Considered, Documented, Not Built

### H9 (reserved, still fully deferred — one prerequisite cleared) — Field Type Sensitivity

**This hypothesis is NOT being added or promoted in this amendment.** No
conditions exist for it, no decision rule has been designed, and it remains
entirely out of 1b's active scope. What changed is narrower and easy to
misread if skimmed: one of the two original blockers to eventually building
H9 has been cleared as a side effect of unrelated H1 engine work, nothing
more.

**Hypothesis:** Fields of different types may carry disproportionate
decision weight even at "matched magnitude". A boolean flip, a numeric
percentage shift, and a categorical plausibility-tier change are not
obviously equivalent perturbations just because they share a magnitude
label.

**Status update:** the original blocker, no defined magnitude concept for
boolean fields, has been resolved. Boolean dithering (flip probability)
was built during H1's engine-verification pass (see "A Note on Protected
Fields and Boolean Support" above) because it was required to make
`h1_category_account_status` runnable at all, not as a deliberate early
pull-forward of H9. **What remains deferred is the comparative hypothesis
itself**. A dedicated study asking whether boolean, numeric, and
categorical fields at "matched" magnitude actually produce comparable
decision drift, or whether one field type is systematically more or less
disruptive than the others regardless of which specific field is tested.
That comparative question was not answered by simply making boolean
dithering possible, and remains genuinely unbuilt.

**Why still deferred:** answering it properly requires a dedicated
cross-type comparison design (matched-field-importance boolean vs. numeric
vs. categorical fields, controlled for how important each field is
independent of its type) which is a real design exercise in its own right, not a
byproduct of H1's engine fix.

### `preferred_categories` — Variable-Length Field Dithering

**What this would test:** how decision drift responds to corruption of a
variable-length subset field (1–4 categories sampled from a pool of 10),
where "corruption" could mean swapping one entry, adding or removing an
entry, or resampling the whole list. Each a meaningfully different
perturbation severity, not implementation variants of one concept.

**Why deferred:** genuinely distinct from H9's boolean/single-select-
categorical question. Neither "flip probability" nor "plausibility tier"
map cleanly onto "corrupt a subset of a list". This needs its own
magnitude concept designed from scratch, separate from both the boolean
mechanism just built and H9's cross-type comparison question. Surfaced
during the same engine-verification pass that resolved boolean support, but
judged a distinct enough problem to warrant its own deferral rather than
folding into H9's scope.

### Seed-Robustness Replication

**What this would test:** whether 1b's *pattern* of findings, which fields
drift most, whether H3's correlated/uncorrelated distinction holds, whether
H7's breadth curve shape repeats, replicates on a freshly generated ground
truth population from a different seed, or whether findings are artifacts of
this one specific synthetic draw (1,000 customers, seed=42).

**Why deferred:** re-running the full (now 48-condition) pipeline against a
second seed roughly doubles 1b's scope. Scoped as Phase 2 replication rather
than a requirement for initial findings.

### Human-Origin Error Dither Type (Fork B)

**What this would test:** whether decision drift differs when the underlying
error originates from a discrete, deliberate human action (a rep transposing
digits, selecting the wrong dropdown option) versus passive time-decay
(`drift`) or passive automated transformation (`entry_error`).

**Why deferred:** requires building a new dither mechanism in
`dither_engine.py`, comparable in scope to H9. Explicitly designed to avoid
reversing the original 1b_DESIGN.md's exclusion of character-level typos.
The mechanism, if built, models discrete human action, not typo-style
character substitution.

### Considered and Intentionally Omitted — Agentic Self-Report of Missing Fields

Asking the agent directly what additional field or information it believes
would improve its decision was considered and excluded. First, it is
introspective self-report rather than behavioral observation, a different
category of measurement than everything else in 1b, with real doubt about
whether an LLM has reliable introspective access to what would actually
change its output versus generating a plausible-sounding answer. Second, and
more critically, asking this question would likely signal to the agent that
its performance is being evaluated, directly undermining H5's methodology.
May resurface as an explicitly-labeled exploratory side study run separately
from the main pipeline, but does not belong in the core hypothesis set.

---

## Updated Condition Count and Cost Implications

| Hypothesis | Original | Amended |
|---|---|---|
| H1 | 7 | 12 |
| H2 | 4 | 12 |
| H3 | 8 | 11 |
| H4 | 2 | 6 |
| H7 (new) | — | 4 |
| H8a (new) | — | 2 |
| H8b (new) | — | 0–1 (conditional) |
| **Total** | **21** | **47–48** |

At n=1,000 customers per condition: 47,000–48,000 dither-condition agent
calls, plus the 5,000-call primary baseline, plus a diagnostic expansion for
`deeply_boundary` customers only (minimum 25 runs; exact added cost
confirmed once real population size is known, estimated at 1,000–2,000
additional calls). Total: roughly **53,000–55,000 agent calls**.

At 1a's observed per-record cost (~$0.00232/record): approximately
**$123–128 at standard API pricing, $62–64 with Batch API's 50% discount.**

---

## Summary of Changes in This Amendment

- **Dither engine extended:** boolean field support added (`is_vip`,
  `has_active_subscription`, `has_pending_order`) via flip-probability
  magnitude, verified against 1,000 synthetic customers at 0.15 magnitude
  (13.5% observed flip rate). `PROTECTED_FIELDS` registry added, raising an
  explanatory error rather than allowing incoherent dithers, `is_at_risk`
  and `recently_contacted_support` protected as derived fields;
  `customer_segment` protected as an upstream field that conditions other
  fields' sampling distributions. New cross-cutting analysis lens
  identified: segment/profile mismatch, checking whether dithered numeric
  profiles still fall within their original segment's typical range,
  free enrichment across H1, H2, H4, H7. `preferred_categories` (variable-
  length list field) identified as a distinct, separately-deferred problem
  from H9's boolean/categorical question.
- **Ground truth re-reviewed against new dither capability.** Confirmed 
  `avg_resolution_time_hours` and `refund_rate` (H3's reference pair) are 
  genuinely segment-independent. Identified that `is_vip`'s skewed 4.3% base 
  rate produces a ~4x population swing under standard 15% flip dithering. 
  Documented as a real phenomenon worth measuring, not an engine flaw to 
  correct. Resolved via two free evaluator side analyses rather than any 
  generator or engine change: per-field/per-direction attribution within 
  multi-field conditions (closing H1's previously-unbuilt Question C for 
  all categories), and empirically-computed (not hard-coded) segment/profile 
  mismatch detection, now explicitly extended to boolean fields.
- **H1** restructured: 7 → 12 conditions. Ambiguous "behavioral" aggregate
  replaced with 6 category level conditions matching the prompt's own
  taxonomy, plus 1 distributed condition. Category field lists corrected
  against actual engine capability. `dob`, `customer_segment`, and
  `preferred_categories` excluded from their respective categories with
  reasons documented; `h1_category_account_status` corrected to its 3
  genuinely-independent fields after removing 2 derived-field impostors.
  Two free analyses identified: category impact ranking, stated vs 
  revealed field importance.
- **H2** restructured: 4 → 12 conditions. Single-field design replaced with
  a 3-field spread (churn_risk_score, total_spend, tenure_months). Field
  correlation concern examined and resolved (single field per condition
  design isolates the measurement). Within segment breakdown added as
  enrichment. 75% magnitude level parked, not rejected. Full 1,000-customer
  population confirmed over subsampling; seed-robustness deferred instead.
- **H3** restructured: 8 → 11 conditions. Confidence and Jaccard enrichments
  added for measurement granularity. Uncorrelated reference pair
  (avg_resolution_time_hours + refund_rate) added as a baseline comparison
  point. Full 2x2 directionality achieved for all three pairs (mostly free,
  reusing H1/H2 conditions); full factorial declined for the triplet.
- **H4** restructured: 2 → 6 conditions. Reuses H2's field trio, resolving
  both the single field problem and the churn_risk_score over concentration
  concern. Third dither type (human origin error, Fork B) considered,
  designed at a conceptual level, explicitly deferred.
- **H5** fully specified: frozen 17-pattern regex keyword list (with
  "uncertain about" deliberately excluded), Jaccard secondary metric shared
  with H3, manual audit methodology for false negative rate estimation,
  three-way cross tab (detected+changed / detected+unchanged / not detected).
- **H6** fully specified: terminology clarified (tiers = stability
  classification, not decision buckets), median ratio + Fisher's exact
  approach to avoid multiple comparisons inflation, deeply_boundary run
  expansion to a minimum of 25 (exact number pending real population size),
  general fragility vs field specific sensitivity crosstab identified as
  free (reuses H1).
- **H7** (Breadth Effects) added. 4-condition accumulation ladder testing
  whether drift scales linearly with number of simultaneously dithered
  fields.
- **H8a** (Category-Level Interaction) added. 2 conditions, additive
  baseline formula defined precisely, negative control surprise handling
  specified.
- **H8b** (Field-Level Interaction) added as conditional. Decision rule
  locked now (run H8a first, use existing individual field data to
  empirically select fields if amplification is found), outcome dependent
  field selection rather than a pre-registered guess.
- **H9** (Field Type Sensitivity) remains fully reserved for Phase 2, not
  added or promoted in this amendment. One of its two original blockers
  (boolean magnitude support) was cleared as a side effect of unrelated H1
  engine work; the comparative hypothesis itself has no conditions, no
  decision rule, and no active scope in 1b.
- **Seed robustness replication** and **human origin error dither type**
  named explicitly as Phase 2 follow ups.
- **Agentic self report of missing fields** considered and excluded from the
  core hypothesis set, with reasoning documented.
- **Prompt continuity note added:** the "conflicting priorities" instruction
  and the undefined "business outcomes" objective, both inherited from 1a,
  are flagged as inert until now and first meaningfully exercised by H3's
  uncorrelated arm. Left unpatched to avoid introducing new confounds.
- Total condition count increased from 21 to 47–48; cost re-estimate
  (~$62–64 at Batch pricing) flagged as needed before full pipeline
  execution.

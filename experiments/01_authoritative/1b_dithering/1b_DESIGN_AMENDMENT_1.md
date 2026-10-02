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

---

## A Note on Perturbation Validity

An exposure audit, checking what SHARE of perturbed customers actually
had their field change, not just whether the mechanism ran without
error, found two real bugs, neither of which had raised an exception.

**`acquisition_channel` never changed at any magnitude, ever.** The
categorical dither dispatcher had no branch for it and silently returned
the value unchanged. `h1_category_segmentation` had been testing
`tenure_months` alone since the category was defined — its second field
was never actually dithered. Fixed by adding the missing branch, and by
making the dispatcher raise an error on any future categorical field with
no implementation, rather than silently pass it through.

**Small-integer fields barely changed at all.** Numeric drift is
multiplicative (value × ~15%); for a count like 1 or 2, that rounds back
to itself, and 0 stays 0 regardless of magnitude. At 15% magnitude,
`support_tickets_open` — one of H1's self-reported top-5 fields — changed
for only 7.9% of customers; `payment_failures` for 2.8%, versus 94–100%
for every continuous field. A field's drift rate was partly measuring how
many customers got touched, not how much the agent cares — exactly the
kind of confound this project has repeatedly hunted down elsewhere.
Fixed with a minimum ±1 step: any customer selected for perturbation now
moves at least one unit. The same bug existed a second time, in
`entry_error`'s fallback path, found by checking whether the pattern that
broke `drift` existed elsewhere — fixed identically.

**One residual, expected limit, measured rather than hidden:** a value
already at its floor (e.g. 0 tickets) that is drawn to move DOWN cannot
move regardless of the fix — there's nowhere to go. The engine now
records this as `_dither_blocked` per customer per field, and the
generator's validity check excludes structurally-blocked customers from
the exposure denominator, so a genuine floor case is not confused with a
broken mechanism. Both are measured and reported separately.

**`generate_dithered_data.py` now runs a validity check on every
condition before writing it** — per-field exposure (with the blocked-move
adjustment above), and for H3's coupled conditions, coherence of the
joint movement. A condition that fails is not written, the full report is
saved to `validity_report.json`, and generation exits with an error. This
runs before any API call, at zero cost, and would have caught both bugs
above immediately.

**Cross-field drift comparisons must use exposure-adjusted drift, not raw
drift.** Raw drift rate averages over untouched customers; when fields
differ in exposure (boolean fields by design flip ~15% of customers at
15% magnitude, versus ~100% for most continuous fields), raw rates aren't
comparable across fields. `evaluate_core.py`'s
`compute_effective_drift_rate()` restricts to customers actually
perturbed (using `_dither_fields`, which lists only fields that changed —
no new metadata needed). Conditioning on "was perturbed" does not
introduce the selection-bias problem raised for H3's confidence metric:
perturbation is set by the dither's own random draw (treatment
assignment), never by the agent's outcome. H1's Question A now uses this
adjustment throughout.


---

## A Note on Statistical Methodology

`evaluate_core.py` uses seven statistical tools across the evaluator. Each
was chosen only after checking whether the more obvious, textbook first
tool actually fits 1b's specific data structure ,several times, it
doesn't. Collected here in one place so the reasoning behind every
methodological choice is visible without hunting through individual
function docstrings.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| How uncertain is a customer's true decision rate from a small run count? | Wald interval (p̂ ± z·√(p̂(1-p̂)/n)) | Collapses to zero width at p̂=0 or p̂=1, a perfectly consistent 5/5 customer would read as 100% certain, which is absurd from only 5 draws | **Wilson score interval**: derived by inverting a hypothesis test rather than centering on p̂'s own (unreliable at small n) variance. Verified against hand calculated values for every n=5 partition shape. |
| Is one new confidence observation unusual vs. a customer's baseline? | Standard one sample t-test (tests whether a *sample mean* differs from a hypothesized value) | Answers a different question than ours, we have one new point, not a competing sample, so the standard SEM-based denominator understates the real uncertainty | **Prediction-interval-style t-statistic**: (`s·√(1+1/n)` in the denominator, not `s/√n`). Verified: the standard formula gives t=15.9 on our test data (absurdly inflated); the correct formula gives t=-6.5 (properly calibrated) on identical input. |
| Does dithered reasoning look less coherent than a customer's own baseline wobble? | Two sample t-test, or Mann-Whitney with the normal approximation | Jaccard scores are bounded [0,1] and often skewed; the normal approximation is unreliable at our sample sizes (as few as 5 dithered scores vs. 10 baseline-pair scores) | **Exact Mann-Whitney U** (`method='exact'`). Verified against both a "looks like normal wobble" scenario (correctly non-significant, p=0.44) and a genuinely degraded scenario (correctly sharp, p=0.0007). |
| Does drift rate differ between two stability tiers within one condition? | Chi-square test | Unreliable with small cell counts, which `tied_no_majority` and `deeply_boundary` frequently produce | **Fisher's exact test** — exact even at small cells, per-condition only (H6's actual claim rests on the median ratio holding across all conditions, not any single p-value). |
| Does field X cause more drift than field Y? | Standard two-proportion z-test, treating each field's customers as independent samples | Every condition dithers the *same* 1,000-customer population — a customer's drift-under-X and drift-under-Y both depend on the same underlying profile, violating the independence assumption | **McNemar's exact test** (paired binary outcomes, discordant pairs only). Verified against both a balanced-discordance null scenario (p=0.49) and a genuinely asymmetric one (p<0.000001). |
| How similar is dithered reasoning to baseline reasoning? | Frequency-weighted term comparison (TF-IDF-adjacent) | The specific weighting scheme considered (favor higher-frequency words) is backwards from standard practice — common words like "the," "customer," "priority" would get amplified, not the substantive differences that matter | **Unweighted Jaccard, stop-word filtered.** Deliberately simple and deterministic — the same reasoning 1a used to choose Jaccard over an LLM judge in the first place: no discretionary parameters, fully auditable, reproducible by anyone reading the code. |

**The pattern across all six:** in every case, the naive tool isn't wrong in general — it's wrong for *this specific data's shape* (small samples, paired observations, skewed bounded distributions, or genuine independence violations). Every substitution was verified against known-correct reference values before being trusted by any hypothesis-specific evaluator file, not assumed correct from memory.

**A seventh tool, added after the six above were already documented, worth a note on how it surfaced:** A condition level Jaccard coherence comparison (was `jaccard_dispersion_test`, pooling every customer's scores into two Mann-Whitney groups) shared the exact same paired data flaw the McNemar's vs. two proportion table above already identifies for drift rate. Namely, every condition dithers the *same* 1,000-customer population, and the same baseline texts feed both sides of the comparison for a given customer, violating Mann-Whitney's independence assumption at the population level.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| Does a dithering condition degrade reasoning coherence across the whole customer population? | Mann-Whitney U on pooled per customer scores (same tool as the per customer diagnostic, just aggregated across customers) | The same baseline texts feed both a customer's dithered vs baseline scores and their own self similarity scores. Pooling across the population treats correlated data as independent, the identical mistake McNemar's was built to avoid for drift rate | **Wilcoxon signed-rank test** on each customer's own paired difference (dithered coherence minus baseline coherence), reduced to one number per customer before testing. The per customer Mann-Whitney diagnostic (`jaccard_dispersion_test`) remains valid *within* a single customer's own two small score sets. It was never wrong, only wrong to pool across customers. |

**The generalized principle, stated plainly:** Any comparison across two conditions applied to the *same* 1,000 customer population is a paired comparison, not an independent samples comparison. Binary outcomes need McNemar's, continuous outcomes need Wilcoxon signed-rank. This applies beyond Jaccard: any future comparison across magnitude levels or dither types for the same field (H2, H4) needs the same treatment, not a naive two-sample test.

 **An eighth tool.** H2's magnitude ladder and H7's breadth ladder both compare more than two paired conditions on the same population — McNemar's only handles pairs.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| Does drift rate differ anywhere across a >2-level paired ladder for the same field? | Running all pairwise McNemar's directly | Multiple comparisons inflate the family-wise error rate; the ladder is ordered, so an omnibus gate followed by adjacent-step tests is both more rigorous and more interpretable | **Cochran's Q** (chi-square approximation — its asymptotic validity depends on customer count, ~1,000, not condition count, unlike every other exact-over-approximate choice here) as an omnibus gate, adjacent-step McNemar's only if significant. Verified via a proven identity: reduces exactly to McNemar's uncorrected chi-square at k=2 (confirmed to 9 decimal places), plus a cliff-then-plateau scenario confirming the gate localizes rather than masks a real effect. |

**A ninth tool.** H3's reference comparisons needed a binary difference-in-differences test.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| Does breaking a real correlation cost MORE drift than breaking a pair with no relationship? | (a) McNemar's comparing a real pair's uncorrelated arm directly to the reference's; (b) Wilcoxon on per-customer double differences | (a) confounds correlation-breaking with field leverage. (b) ~60% of customers land at exactly zero under realistic drift rates (verified by simulation), gutting power | **Binary DiD via GEE** (logit link, clustered by customer, robust sandwich SEs). Verified via planted-effect simulation: null stays silent (OR=1.09, p=0.55), a true effect is recovered with a CI containing the planted value (OR=2.17 vs. true ≈2.43), and a sign-reversal check confirms direction is meaningful. A deliberate reversal of the position taken for Cochran's Q — `statsmodels` is added to `requirements.txt` because no simpler tool answers this question correctly. |

 **A tenth entry, not a new tool — a correction to how the existing tools are fed.** An exposure audit found fields differ enormously in how many customers a condition actually touches (booleans ~15% by design, some small-integer counts as low as 2.8% before an engine fix, versus ~100% for continuous fields). Raw drift rate partly measures exposure, not agent sensitivity. **Exposure-adjusted drift rate** — computed only among customers actually perturbed — is used for every cross-field comparison (H1's Question A throughout). Conditioning on "was perturbed" does not introduce selection bias: perturbation is set by the dither's own random draw, never by the agent's outcome.

**An eleventh tool.** H4 needed to compare three mechanisms (drift, plausible entry error, implausible entry error) at once, not just two.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| Does entry-error style (plausible vs. implausible) affect drift differently than drift itself, as two separate questions? | A 3-level factor with standard treatment coding (one reference level, two dummy variables) | Gets "plausible vs. drift" and "implausible vs. drift" directly, but "plausible vs. implausible" — H4's actual central question — has to be derived as a difference of two coefficients rather than read off directly, and a library's default contrast-coding convention is an easy place for a sign to silently flip | **Two hand-built orthogonal contrasts** (style: drift vs. average of both error types; plausibility: plausible vs. implausible directly), with an explicitly locked sign convention. Verified against three planted fixtures (Garbage Filter, Outlier Vulnerability, null) plus a direct orthogonality check — the style coefficient stayed flat while the plausibility coefficient swung from +1.60 to −1.57 across the two opposite scenarios. |

**A twelfth entry, a gate rather than a tool.** H4's three fields each use a structurally different entry-error operator (scale swap, unit conversion, unit swap/default seeding) — pooling the eleventh tool's contrasts across fields risks averaging three different mechanisms into one misleading number.

| Question | Naive first choice | Why it fails here | What we use instead |
|---|---|---|---|
| Is it safe to report one pooled plausibility effect across all three H4 fields? | Fit the pooled model and trust its p-value | Verified by simulation: a pooled coefficient near zero (mean −0.09) was statistically significant in 19 of 20 replicates despite describing neither of two genuinely opposite-sign fields (+1.3 and −1.7) — the danger isn't a null result, it's a confident, misleading small effect | **A joint Wald test on each contrast's interaction with Field**, fit in the same model as the pooled contrasts. Fired on 20/20 opposite-sign replicates and 30/30 mild-heterogeneity replicates, while firing falsely in only 2/40 replicates under a genuine null — confirming it's calibrated, not just trigger-happy. When it fires, per-field contrasts are the primary report; the pooled number becomes a footnote. |


## Forward-Looking Note: Paired Comparisons Beyond H1/H3

The generalized principle above (same-population comparisons need paired
tests, not independent samples tests) applies to at least four places not
yet built:

- **H2's magnitude ladder**: drift rate at 15% vs. 40% for the same
  field is the same 1,000 customers under two treatments. McNemar's, not
  a two proportion test.
- **H3's core comparison**: correlated vs. uncorrelated arms, same
  population. McNemar's again.
- **H4's dither-type comparison**: evolved beyond this note's original
  prediction once built. McNemar's per field remains the primary test
  (drift vs. plausible, drift vs. implausible, same population), but H4
  also needed a 3-level mechanism comparison (drift, plausible,
  implausible) and a Field × Mechanism interaction gate to decide
  whether pooling across fields is safe — see "A Note on Statistical
  Methodology," eleventh and twelfth entries, and the full H4 section.
- **H7's breadth ladder**: genuinely different from the other three:
  four paired conditions (1 field, 3, 6, all), not two. McNemar's only
  handles pairwise comparisons. The correct generalization to more than
  two paired conditions is **Cochran's Q test**.

**Deliberately unresolved for now, to be decided when `evaluate_h7.py` is
actually built, not guessed at in advance:** if Cochran's Q returns a
significant result (drift rate genuinely differs somewhere across the
breadth ladder), what's the follow up? Candidates include pairwise
McNemar's across all six condition pairs with a multiple comparisons
correction (Bonferroni, Holm, or similar), or a different post-hoc
approach entirely. This is intentionally left open rather than committed
to now. Building the correction logic before we've seen real H7 data
risks the same premature commitment mismatch H8b's field selection was
specifically designed to avoid by deferring to real data instead of a
pre-registered guess.


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

For any condition that dithers numeric fields, the evaluator will check whether a 
customer's dithered profile still falls within their original `customer_segment`'s 
typical range and flag customers whose profile has effectively "left" their assigned 
segment's normal territory. **Computed empirically from the 1,000-customer ground 
truth dataset itself** (e.g. 5th–95th percentile per field per segment, grouped 
directly from `canonical_customers.json`), not by extracting the generator's internal 
literal bounds, which would require a refactor and would create a second, driftable 
source of truth. This is a free enrichment across H1, H2, H4, and H7, anywhere a 
numeric field gets dithered, not a new hypothesis or new engine mechanism.

**The same concept extends to boolean fields.** `is_vip=True` while `customer_segment != high_value` 
after dithering is a categorical instance of the identical phenomenon. Specifically, a dithered 
attribute no longer matching what the customer's protected, undithered segment assignment would 
predict. Folded into this same lens rather than treated as a separate concept.

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

A note on condition numbering: Earlier versions of this document numbered conditions 
sequentially across the whole experiment (1–48). This is dropped as of this revision. Every 
time a hypothesis's condition count changed, it required renumbering everything downstream, 
which happened often enough to become its own source of error risk. Conditions are now 
referenced by condition_id alone, matching how they're actually identified in code.

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

### Restructured condition set

Five individual field conditions, each 1a H4 top-5 field dithered alone at 15% magnitude:
`h1_individual_last_purchase_days_ago`, `h1_individual_churn_risk_score`, `h1_individual_nps_score`, 
`h1_individual_lifetime_value_estimate`, `h1_individual_support_tickets_open`.

Two comparison group individual field conditions (added): filling the only two categories with zero 
individually-tested fields anywhere in the 44-condition set, at matched 15% magnitude: `h1_individual_email` 
(Identity, predicted near-zero drift, a genuine test of the null hypothesis, analogous to H8a's negative control 
framing) and `h1_individual_is_vip` (Account Status: also isolates whether `h1_category_account_status` bundled 
drift is driven by `is_vip` specifically or spread across its three fields, using the per-field 
attribution machinery already built in evaluate_core.py).

Six category-level conditions: every field in one prompt section dithered together at matched 15% magnitude, 
uncorrelated. Field lists reflect engine-verified corrections. See "A Note on Protected Fields and Boolean Support" 
above for the full reasoning behind each exclusion:

`h1_category_identity` — name, email, phone, address (dob excluded, not currently dither capable)
`h1_category_purchase_behavior` — total_purchases, total_spend, avg_order_value, purchase_frequency_days, last_purchase_days_ago, lifetime_value_estimate
`h1_category_engagement` — nps_score, email_open_rate, last_login_days_ago, support_tickets_open, support_tickets_closed, avg_resolution_time_hours
`h1_category_risk_factors` — churn_risk_score, payment_failures, fraud_risk_score, refund_rate
`h1_category_segmentation` — acquisition_channel, tenure_months (customer_segment excluded. Protected, upstream field; preferred_categories excluded, list-valued, deferred separately)
`h1_category_account_status` — is_vip, has_active_subscription, has_pending_order (is_at_risk and recently_contacted_support excluded and protected, derived fields)

One distributed condition: one field per category, deliberately excluding the H4 top-5, matched 15% magnitude, 
uncorrelated: h1_distributed: email, avg_order_value, last_login_days_ago, refund_rate, acquisition_channel, has_pending_order.

### Question A methodology — group-level test and its limitation
Testing "do the H4 top 5 fields produce more drift than fields the agent didn't self-report as important" runs into a real 
statistical trap worth naming precisely: pseudo-replication. Each field's drift rate is well-powered on its own 
(~1,000 customers per field), but the group level question, top-5 fields as a group vs. comparison fields as a group, has 
a true sample size equal to the number of fields tested, not the number of customers. Customers dithered under the same 
field are not independent replicates of "what happens when a top-5 field gets dithered"; pooling them would artificially inflate apparent power.
**Two complementary analyses, not one:**

1. **Group-level check (secondary):** Mann-Whitney U comparing the 5 self reported top 5 field drift rates against the 6 comparison 
field drift rates (`email`, `is_vip`, added specifically as predicted null comparison fields, plus `total_spend`, `tenure_months`, 
`avg_resolution_time_hours`, `refund_rate`, reused from H2/H3's individual conditions). Reported honestly as low powered given 
n=5 vs n=6 fields. A clean separation is still interpretable, but a null result here does not mean "no effect," only 
"not enough fields tested to detect one at this sample size."
2. Per-field check (primary): Each individual field's drift rate (n≈1,000, well-powered) tested via pairwise McNemar's exact test against 
each comparison field, not a standard two-proportion test, since every field level condition dithers the same 1,000-customer population, 
and a naive two proportion test would wrongly treat those customers as independent samples across conditions (see "A Note on Statistical 
Methodology" above). With 5 self-reported top-5 fields and 6 comparison fields, this produces 30 pairwise tests. The headline 
finding is whether the top 5 fields' apparent advantage holds consistently across those pairwise comparisons, not whether any single 
comparison clears significance in isolation. This is the same "consistency across many checks, not one p-value" framing already used for H6's tier ordering.

**Comparison group note:** Drawing `total_spend`, `tenure_months`, `avg_resolution_time_hours`, and `refund_rate` from H2 and H3's 
individual conditions is intentional cross-hypothesis data reuse, consistent with the pattern already established 
for H2/H4's field-reuse and H3's free 2x2 directionality. `evaluate_h1.py` reaches into H2 and H3's condition folders 
for this one analysis rather than duplicating data generation.


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
breaking expected correlations), at matched 15% magnitude.

### What stress-testing surfaced

**Granularity of the outcome measure.** Due to a limited number of decision
buckets, binary drift (did the decision bucket change) cannot distinguish a
confident correlated-arm shift from a visibly conflicted uncorrelated-arm
shift landing on the same bucket. **Fix:** report three things per condition,
not just the decision: (1) binary drift rate, (2) confidence, handled
separately below, (3) Jaccard similarity between dithered and baseline
reasoning text, as a coherence signal (shared machinery with H5's secondary
metric).

**Assumed correlation that didn't exist.** Two of the three originally-
specified pairs (`churn_risk_score`+`last_purchase_days_ago`,
`support_tickets_open`+`avg_resolution_time_hours`) sounded plausible but had
essentially zero actual correlation in the base generator (r≈+0.017,
r≈−0.014 respectively) — statistically indistinguishable from a
deliberately-null reference pair (r≈−0.001). See "A Note on Assumed vs.
Forced Correlation." Replaced with fields verified to correlate through
genuine common-cause structure (shared segment conditioning):

- Pair 1: `churn_risk_score` + `nps_score` (r=−0.72)
- Pair 2: `total_spend` + `lifetime_value_estimate` (r=+0.99 — direct
  derivation, documented as a borderline case rather than glossed over,
  judged acceptable because it mirrors a standard real-world CRM heuristic)
- Pair 3: `support_tickets_open` + `payment_failures` (r=+0.51)
- Triplet: `total_spend` + `support_tickets_open` + `churn_risk_score`
  (r=−0.28, −0.58, +0.59 — every pairwise direction reinforces the
  "disengaging high-value account" narrative)

**Correlation mechanism, redefined after an audit found the original
definition non-functional.** The original design let each field drift in
its own independently-assigned "natural" direction (e.g. `churn_risk_score`
biased upward, `nps_score` with no defined direction), on the theory that
both fields moving "naturally" would look coherent together. Measuring
actual joint coherence found this did not work: Pair 1 was 52.5% coherent
in the "correlated" arm versus 52.0% in "uncorrelated" — statistically
identical — because `nps_score` has no natural direction to assign. The
triplet's "correlated" arm measured LESS coherent (10.8%) than its own
uncorrelated arm (27.3%), since pushing `total_spend` and `churn_risk_score`
both "up" directly contradicts their verified r=−0.58. The mechanism also
gave the two arms different marginal odds (85/15 bias vs. 50/50),
confounding coherence with directional bias.

Replaced with an explicit coupling mechanism: each field set gets a
coupling sign per field (its verified correlation sign relative to the
set). Correlated arm — one fair coin per customer sets a shared direction;
each field follows it times its coupling sign. Uncorrelated arm — each
field gets its own independent fair coin. Both arms give every field
identical 50/50 marginal odds, so the only thing differing between arms is
whether the fields move together. Re-verified after the fix: every coupled
set's correlated arm shows ≥99% coherence; every uncorrelated arm sits
within a few points of chance (50% for pairs, 25% for the triplet).

`recompute_derived=True` in BOTH arms, not only the uncorrelated arm as
originally specified — leaving derived fields stale in only one arm added
a second, unintended difference between arms and contradicted the later
decision that no hypothesis studies derived-field mismatch specifically.

**An uncorrelated reference baseline.** Rather than treating "correlated
vs. uncorrelated" as the only axis, a genuinely unrelated field set — no
real-world relationship, no shared segment-conditioning — gives a
reference point: "this is what dithering unrelated things looks like."
Reference pair: `avg_resolution_time_hours` + `refund_rate` (r≈−0.001).
Reference triplet: adding `tenure_months` (all three pairwise |r| < 0.022;
`last_purchase_days_ago` was considered and excluded from this role
specifically because it is *bounded by* `tenure_months` in the generator,
producing a real if weak r=0.138 — not a clean null). The 3-field reference
matches the test triplet's perturbation volume (k=3), so a reference
comparison never conflates correlation-breaking with the sheer number of
fields touched.

**Direct comparison against the reference was found to be confounded and
replaced with difference-in-differences.** Comparing Pair X's uncorrelated
arm directly to the reference's uncorrelated arm conflates two different
things: how broken the correlation is, and how much intrinsic decision-
weight the agent places on Pair X's specific fields. Resolved via DiD:
compute each set's own within-set delta (drift under decorrelation minus
drift under correlation), which nets out field-leverage as a level effect
shared by both arms of the same set, then compare that delta against the
reference's own delta (which should be ~0). Implemented via
`binary_did_gee()` — a GEE with a logit link, clustered by customer,
testing the group × arm interaction directly on the raw binary outcome
(see "A Note on Statistical Methodology"). Applied three times (each real
pair or the triplet against its matching reference), so the headline
finding is whether all comparisons consistently show a larger decorrelation
cost than the reference, not whether any single test clears p<0.05 in
isolation. Both arms of both reference sets are required — GEE's
interaction term is not estimable without a reference-correlated cell.

**Confidence-within-drifted-customers: methodology.** Comparing agent
confidence between arms specifically "among customers who drifted" is a
post-treatment conditioning problem — which customers count as "drifted"
differs in composition between arms, since the arms differ in how potent
they are at inducing drift. Comparing those two subsets directly would
compare non-equivalent populations. Resolved as two complementary tests:

- **Primary: full-population, unconditional paired comparison.** Every
  customer contributes one paired difference (confidence under correlated
  minus confidence under uncorrelated), regardless of drift status, tested
  via `wilcoxon_signed_rank_test()`. Conditions on nothing, so the
  selection-bias problem does not arise. Answers "does the type of
  dithering shift confidence overall."
- **Secondary, explicitly scoped: the "always-drifters" principal
  stratum.** Same test, restricted to customers who drifted under BOTH
  arms — holding customer-level susceptibility constant. Explicitly
  disclosed as describing that subpopulation, not customers in general.
- A mixed-effects model with a Treatment × Drift interaction term was
  considered and declined: Drift here would be a covariate explaining a
  DIFFERENT outcome (confidence) measured at the same time — a mediator,
  since Drift is itself caused by treatment. Using it as a covariate does
  not escape post-treatment conditioning, and no simple closed-form
  verification case exists for it, unlike every other primitive in this
  project.

**Directionality (the free 2×2).** Checked against existing conditions
before assuming new ones were needed:

- Pair 1 (`churn_risk_score`+`nps_score`): both individual conditions
  already exist in H1 — **free**
- Pair 2 (`total_spend`+`lifetime_value_estimate`): both already exist
  (H2, H1) — **free**
- Pair 3 (`support_tickets_open`+`payment_failures`): `support_tickets_open`
  exists in H1; `payment_failures` does not exist anywhere — **one new
  condition required**
- Reference pair: neither exists individually — **two new conditions
  required**
- Reference triplet: `avg_resolution_time_hours` and `refund_rate` already
  exist individually; `tenure_months` exists via H2 — **free**
- Triplet: full 2×2×2 factorial would require 6 additional pairwise
  conditions. **Declined** — remains correlated/uncorrelated only; full
  factorial breakdown on 3+ field combinations explicitly deferred to
  Phase 2.

### Condition set (15 total)

`h3_pair1_churn_nps_{correlated,uncorrelated}`
`h3_pair2_spend_ltv_{correlated,uncorrelated}`
`h3_pair3_support_payment_{correlated,uncorrelated}`
`h3_triplet_{correlated,uncorrelated}`
`h3_reference_pair_{correlated,uncorrelated}`
`h3_reference_triplet_{correlated,uncorrelated}`
`h3_individual_avg_resolution_time_hours`
`h3_individual_refund_rate`
`h3_individual_payment_failures`


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

### The matched-definition decision: 100% prevalence, realistic intensity, not matched magnitude

Two ways of calibrating "entry_error" against `drift`'s 15% magnitude were
considered and rejected. **Matched prevalence** (apply entry_error to only
15% of customers, matching drift's per-customer magnitude as if it were a
rate) was rejected: it throws away 85% of the sample for no benefit, since
prevalence and conditional drift risk are separable
(`P(drift) = prevalence × conditional_risk`) and conditional risk is the
property worth measuring precisely — practitioners can rescale by their own
organization's prevalence afterward. **Matched population MAPE** (calibrate
entry_error's average error across the population to ~15%) was rejected: at
n=1,000, hitting a 15% population average with realistic (large) per-row
errors means corrupting only ~2 customers, with no statistical power and a
meaningless "15% average" blending a handful of huge errors with thousands
of untouched rows.

**Locked instead: 100% prevalence in both the drift and entry-error arms,
with realistic, uncapped intensity for entry_error.** Forcing a realistic
operator (unit conversion, scale swap) to a 15% numeric move would produce
an artifact that doesn't exist in real pipelines — a 15% unit-conversion
error isn't a thing. The comparison is explicitly "realistic entry error
vs. realistic drift," not "same intensity, two mechanisms," and the
resulting intensity mismatch is reported as context (see Metrics below),
not hidden or corrected for.

This is a deliberate feature, not an accepted flaw: if entry errors — much
larger, uncapped — produce LESS drift than drift-type corruption, that is
strong evidence the agent implicitly filters implausible values while
missing small plausible drift, a genuinely informative finding either way
the result lands.

### The plausibility split: a 2×2 design, not a single operator

A single-operator, 3-condition design (one realistic entry-error operator
per field) was expanded to 6 conditions — a plausible (in-bounds) and an
implausible (out-of-bounds) operator per field — after recognizing a
3-condition design couldn't distinguish two different explanations for a
null or negative result: "entry errors disrupt less because the agent
filters obvious junk" vs. "entry errors just don't disrupt agent reasoning
in general." Splitting plausible from implausible turns H4 into a
mechanism study that directly answers which explanation is correct.

This split also surfaced a genuine detectability gradient across the three
fields, confirmed against the actual prompt (not assumed): `churn_risk_score`
has an explicit stated range in the field glossary ("0.0-1.0, higher = more
risk"); `total_spend` ("Lifetime spending amount") and `tenure_months`
("How long they've been a customer") have no stated units or bounds. A
scale-mismatch error on `churn_risk_score` violates a rule the agent was
actually told; an equivalent error on the other two fields is only
detectable by cross-referencing other fields or general implausibility.

### The operator matrix (locked)

| Field | Plausible — Up | Plausible — Down | Implausible — Up | Implausible — Down |
|---|---|---|---|---|
| `churn_risk_score` | Reset to `0.85` *(constant)* | Reset to `0.15` *(constant)* | ×100 *(personalized)* | ×−1 *(personalized)* |
| `total_spend` | ×2.0 *(personalized)* | ÷10.0 *(personalized)* | ×100.0 *(personalized)* | ×−1 *(personalized)* |
| `tenure_months` | ×1.3 *(personalized)* | ÷1.3 *(personalized)* | ×30 *(personalized)* | ×−1 *(personalized)* |

Direction is a fair coin per customer, applied WITHIN each condition — not
a separate condition axis. This does not change the condition count: 6
conditions total (plausible + implausible per field), each producing a
roughly balanced internal mix of up/down outcomes.

### The direction confound: found and fixed

The original single-operator design (one operator per field, always the
same direction — e.g. `total_spend`'s only entry-error operator always
moved spend down) meant any observed plausibility effect could really have
been a direction effect, confounded further with decision-boundary
position (a customer already at the top priority bucket can't be pushed
further up). Balancing direction independently within BOTH the plausible
and implausible arms removes this: the two arms now differ only in
plausibility, never systematically in which way values move.

**Direction is derived from the observed before/after comparison, never
logged from the coin's intended label.** For constant-target operators
(`churn_risk_score`'s plausible resets), a customer whose value already
sits on the far side of the target experiences the opposite of what the
coin intended — verified concretely: a customer at 0.880 who drew the "up"
coin (target 0.85) actually decreases, and roughly 20% of customers sit far
enough from the midpoint for this to matter in practice, not a rare edge
case. The engine computes before/after values for every dithered field
regardless, so deriving direction this way required no new infrastructure
— only using data already produced.

### Mechanism style: tracked as metadata, not forced into false uniformity

`churn_risk_score`'s plausible cells are unavoidably constants — "system
default persists" IS a fixed value in real life, and personalizing it
would stop it from representing that failure mode. Its implausible cells
are naturally personalized (a real scale-mismatch bug multiplies whatever
the true value is, it doesn't emit the same wrong number for everyone).
This gives `churn_risk_score` a style asymmetry between its plausible and
implausible cells that `total_spend` and `tenure_months` don't have.
Rather than forcing artificial uniformity that would misrepresent the real
failure mode, `_dither_mechanism_style` (`constant`/`personalized`) is
tracked as per-customer metadata, available to check as a candidate
explanation if `churn_risk_score`'s results ever look anomalous relative
to the other two fields.

### Bounds bypass

Every dithered value is normally clipped to the field's defined min/max.
The implausible arm bypasses this clip entirely — a scale-swap to 39
silently clamped back into [0,1] stops being a scale-swap and becomes
just another mild drift. The bypass applies ONLY to implausible
conditions; the plausible arm keeps normal clipping, since respecting the
field's contract while still being wrong is exactly what makes it
"plausible." The bypass skips RANGE validation only, not TYPE — an
integer field stays integer-typed even out of range, matching how a real
database enforces schema without enforcing business rules (the actual
reason these errors reach production data at all).

**A field-specific finding from testing this against real generated data,
tracked rather than "fixed":** `total_spend`'s implausible-up operator
(×100) only escapes the field's own 500,000 ceiling for customers whose
original spend exceeds $5,000 — about 26% of the population, verified
directly against 1,000 generated customers. `churn_risk_score` and
`tenure_months`'s implausible-up operators escape bounds for ~98-99% of
customers, because those fields' ceilings are small relative to typical
values, while `total_spend`'s ceiling is large relative to its typical
values (median ~$2,476). Inflating the multiplier to force a higher
escape rate would sacrifice realism — a genuine dollars-vs-cents bug is
×100, not some artificially larger factor chosen to guarantee a
statistical property — so this was not changed. No new engine tracking
was needed either: whether a given customer's dithered value escaped
bounds is fully derivable from the existing before/after values and the
field's own known bounds, so the evaluator computes and stratifies by
this directly rather than the engine needing to record it separately.

### Statistical plan

**Primary, per field: McNemar's**, same shape as every other
same-population comparison in this project — `drift_vs_plausible` and
`drift_vs_implausible` separately, since every condition dithers the same
1,000-customer population.

**Secondary: `gee_style_plausibility_test()`** — two orthogonal contrasts
on the 3-level Mechanism factor (drift / plausible / implausible), fit
via GEE with a logit link, clustered by customer, using HAND-CONSTRUCTED
numeric contrast columns rather than a statistics library's built-in
Helmert contrast class — removes any risk of a silent sign flip from an
unfamiliar internal convention. Style contrast (`drift=-2, plausible=+1,
implausible=+1`): drift vs. the average of both entry-error types.
Plausibility contrast (`drift=0, plausible=+1, implausible=-1`): plausible
vs. implausible directly. **Locked sign convention:** positive &
significant = Garbage Filter Effect (plausible errors slip through, cause
MORE drift than implausible); negative & significant = Outlier
Vulnerability (implausible errors disrupt the agent MORE than plausible
ones). Verified with three synthetic fixtures: a planted Garbage Filter
scenario came back positive and significant (p≈4×10⁻¹⁶⁰); a planted
Outlier Vulnerability scenario came back negative and significant
(p≈3×10⁻¹⁵²); a null scenario came back non-significant (p=0.61).
Orthogonality between the two contrasts confirmed directly: the style
coefficient stayed near zero across both planted scenarios while the
plausibility coefficient swung from +1.60 to −1.57.

**`gee_field_mechanism_interaction_gate()` decides whether pooling across
fields is safe.** Each field uses a structurally different operator, so a
pooled effect risks averaging qualitatively different mechanisms into one
number. Verified by simulation: on a genuinely opposite-sign two-field
scenario, the pooled plausibility coefficient came out near zero (mean
−0.09, against true per-field effects of about +1.3 and −1.7) yet was
statistically significant in 19 of 20 replicates — the failure mode is
NOT "pooled model reports no effect," it is "pooled model confidently
reports a small effect that describes neither field." The interaction
gate fired in 20 of 20 replicates on that scenario, 30 of 30 on mild
same-sign heterogeneity, and — the case that confirms it is calibrated
rather than just over-triggering — fired falsely in only 2 of 40
replicates under a genuine null where every field has the identical
effect, matching the expected ~5% false-positive rate at α=0.05.
**Consequence: per-field contrasts are H4's primary report whenever the
gate fires; the pooled contrast is a footnote, not the headline.**

### Condition set (6 total)

`h4_churn_risk_score_plausible`, `h4_churn_risk_score_implausible`,
`h4_total_spend_plausible`, `h4_total_spend_implausible`,
`h4_tenure_months_plausible`, `h4_tenure_months_implausible`.

Drift-type conditions for these three fields are served directly by H2's
existing 15%-magnitude conditions (`h2_{field}_mag15pct`) — confirmed at
build time these can be reused directly rather than regenerated, reducing
H4 to 6 new conditions, not 9.

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

### The frozen keyword list (final, 25 patterns)

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
| H1 | 7 | 14 |
| H2 | 4 | 12 |
| H3 | 8 | 15 |
| H4 | 2 | 6 |
| H7 (new) | — | 4 |
| H8a (new) | — | 2 |
| H8b (new) | — | 0–1 (conditional) |
| **Total** | **21** | **53-54** |

(H1=14 + H2=12 + H3=15 + H4=6 + H7=4 + H8a=2 + H8b=0–1)


At n=1,000 customers per condition: 53,000–54,000 dither-condition agent calls, plus the 5,000 call primary baseline (5 runs × 1,000 customers).

Boundary expansion cost is a genuine open unknown, not a placeholder estimate. The mechanism now covers three tiers 
(`deeply_boundary`, `lightly_boundary`, `tied_no_majority`, per the `aggregate_baseline.py` tied vote fix) with an adaptive 
Wilson interval convergence loop (batches of 10, ±15pp precision threshold, hard cap at 60 total runs per customer), 
not the earlier flat "minimum 25 runs" estimate this section previously cited. Exact added cost depends on how large the 
combined boundary population turns out to be once the primary baseline actually runs. See `RESEARCH_NOTES.md`'s open 
question tracking this same population split as a stochasticity finding in its own right.

Total: roughly **58,000–59,000 agent calls before boundary expansion**, with boundary expansion itself unknown until real baseline data exists.

At 1a's observed per-record cost (~$0.00232/record): approximately **$135–137 at standard API pricing before boundary expansion, $67–68 with Batch API's 50% discount**.

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
- **H4** restructured: 2 → 6 conditions. Reuses H2's field trio (resolving
  both the single-field problem and the churn_risk_score over-concentration
  concern), but the design went substantially further during a dedicated
  design session: expanded from one entry-error operator per field to a
  plausible/implausible split (directly testing whether an agent implicitly
  filters obvious junk while remaining vulnerable to stealthy, plausible
  corruption), after which a direction confound was found and fixed
  (direction is now derived from observed before/after values, never
  logged from the coin that selected an operator), mechanism style
  (constant vs. personalized) tracked as metadata rather than forced into
  artificial uniformity, and a field-specific bounds-escape finding for
  `total_spend` documented rather than "fixed" by sacrificing realism.
  Two new GEE-based statistical tools built and verified
  (`gee_style_plausibility_test()`, `gee_field_mechanism_interaction_gate()`
  — see Statistical Methodology note). Third dither type (human origin
  error, Fork B) considered, designed at a conceptual level, explicitly
  deferred, unchanged from the original restructuring.
- **H5** fully specified: frozen 25-pattern regex keyword list (with
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
- Total condition count increased from 21 to 46–47; cost re-estimate
  (~$62–64 at Batch pricing) flagged as needed before full pipeline
  execution.
- **H1** expanded from 12 to 14 conditions: two individual field conditions 
added (`email`, `is_vip`) to close the only two categories with zero 
individually tested fields. Question A's group level test explicitly flagged 
as low powered (pseudo-replication: true n is number of fields, not number of customers). 
A per-field two proportion test against the comparison group's average adopted 
as the primary analysis instead. Global sequential condition numbering dropped 
throughout the document in favor of `condition_id alone`, removing a recurring 
source of renumbering errors.

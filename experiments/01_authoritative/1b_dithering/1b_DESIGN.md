# Experiment 1b: Dithering — Design Document

> **The Agentic Data Contract · Phase 1 · Experiment 1b**
> Status: Pre-registration. Written before any code. This is a contract with future-self.
> Model under test: `claude-haiku-4-5-20251001`, temperature `0.0`
> Predecessor: see [`../1a_duplication/1a_RESULTS.md`](../1a_duplication/1a_RESULTS.md)

---

## 1. Research Question

1a asked: *what happens when the agent sees the same customer more than once?*
1b asks: **what happens when the data isn't duplicated, but it's wrong?**

1a's H4 finding gave us a list. The agent, in its own reasoning text, consistently cited five fields when making prioritization calls: `last_purchase_days_ago`, `churn_risk_score`, `nps_score`, `lifetime_value_estimate`, `support_tickets_open`. That was a *self-report* — what the agent says it cares about. 1b turns that self-report into a falsifiable claim. If the agent really leans on those fields, dithering them should move decisions more than dithering anything else. If it doesn't, the self-report was decoration.

This matters because every observability story for agentic systems currently rests on stated reasoning. We read the chain-of-thought, we score it, we trust it. The interpretability literature has shown repeatedly that stated reasoning can function as post-hoc rationalization rather than a faithful generative driver of the answer. Turpin, Michael, Perez & Bowman ("Language Models Don't Always Say What They Think," NeurIPS 2023) documented this directly: *"When we bias models toward incorrect answers, they frequently generate CoT explanations rationalizing those answers. This causes accuracy to drop by as much as 36% on a suite of 13 tasks from BIG-Bench Hard."* 1b is a small, domain-grounded test of the same gap: does the agent's stated field importance predict its actual field sensitivity?

The framing for the series: **Decision Quality** is the dependent variable. Data conditions are the independent variables. Each experiment isolates one pillar of the Agentic Data Contract. 1a was Uniqueness. 1b is the Accuracy/Consistency pillar — what happens when the fields the agent reads are present, well-formed, plausible, and quietly wrong.

The aphorism that frames this experiment: *Duplication makes the agent contradict itself. Dithering makes the agent confidently wrong about a different customer than the one in front of it.*

---

## 2. Hypotheses

Six hypotheses. Each is stated as a directional prediction, with the operational measurement called out so future-me can't quietly redefine "drift" after seeing results.

### H1 — Field Importance Validation

**Prediction.** Dithering the agent's self-identified top-5 fields will produce more decision drift, per unit of dither magnitude, than dithering identity fields (name, email, phone) or random non-H4 behavioral fields.

**Why this matters.** This is the load-bearing hypothesis of the series. 1a's H4 result is a *stated* importance ranking. H1 tests whether the stated ranking matches the *revealed* ranking. If it doesn't, every downstream interpretability claim in the series gets weaker, and the practitioner takeaway — "read the agent's reasoning to know what it depends on" — gets a footnote it didn't have before.

**Conditions.** Seven dither conditions plus the clean baseline:

1. `dither_last_purchase_days_ago` — single field, top-5
2. `dither_churn_risk_score` — single field, top-5
3. `dither_nps_score` — single field, top-5
4. `dither_lifetime_value_estimate` — single field, top-5
5. `dither_support_tickets_open` — single field, top-5
6. `dither_identity_aggregate` — name + email + phone together
7. `dither_behavioral_aggregate` — non-top-5 behavioral fields aggregated (e.g., `account_age_days`, `preferred_channel`, `product_category_count`)

Each tested at matched magnitudes (see H2). Single-field conditions isolate per-field sensitivity; aggregates establish a comparison floor for "stuff the agent says it doesn't care about."

### H2 — Magnitude Effects

**Prediction.** Decision drift scales monotonically with dither magnitude within each field. Not linearly — almost certainly not — but monotonically.

**Numeric magnitude tiers.** For numeric fields: ±5%, ±15%, ±40%, ±100% of the field's baseline value (clipped to the field's plausible range so we don't violate validity constraints — see §5).

**Categorical/identity magnitude tiers.** Three plainly-defined tiers, because semantic distance for strings is inherently subjective and I am not going to pretend otherwise:

- **Minimal / typo-equivalent**: `bob@marley.com` → `bob.marley@gmail.com` (same person, format change)
- **Moderate / format shift**: `bob@marley.com` → `b.marley@gmail.com` (plausible alternate)
- **Significant / persona shift**: `bob@marley.com` → `jimbo123@gmail.com` (would a human reviewer say "wait, is this the same person?")

I'm not claiming these tiers are quantitatively spaced. I'm claiming a competent human reviewer would order them the same way every time. That's the bar.

### H3 — Internal Consistency

**Prediction.** When two correlated fields drift *together* in a plausible direction (the customer "looks consistently different"), decision drift will differ from when the same two fields drift *independently* (the customer "looks internally contradictory"), even at matched per-field magnitudes.

**Direction of effect not preregistered.** Two competing intuitions:
- *Correlated dither produces more drift*: a coherent shift moves the customer's mental model to a different bucket; the agent confidently rebuckets them.
- *Uncorrelated dither produces more drift*: contradictory signals destabilize the agent's reasoning into different categories on each run.

I genuinely don't know which wins. That's the point of running it.

**Field pairs and the triplet:**

| Set | Fields | Plausible correlation |
|---|---|---|
| Pair 1 | `churn_risk_score` + `last_purchase_days_ago` | Both rise together as customer disengages |
| Pair 2 | `total_spend` + `lifetime_value_estimate` | LTV is partially derived from spend; should track |
| Pair 3 | `support_tickets_open` + `recently_contacted_support` | Open tickets imply recent contact |
| Triplet | `total_spend` + `last_purchase_days_ago` + `support_tickets_open` | All three move when a high-value account disengages |

For each, two arms: `correlated_dither` (all fields drift in the plausible joint direction) and `uncorrelated_dither` (fields drift independently, breaking the expected joint distribution).

### H4 — Dither Type Effects

**Prediction.** Data drift dither and modern data entry error dither produce different decision drift patterns, even when matched for per-field magnitude.

**Data drift.** Directional, asymmetric, time-driven. Modeled on the operational reality of CRM data aging: `last_purchase_days_ago` only grows (the clock only ticks forward), `churn_risk_score` trends upward for disengaged accounts, `total_spend` stays flat or grows but rarely shrinks, `nps_score` regresses toward the population mean as the most recent survey response ages. This is the asymmetry that RFM (Recency-Frequency-Monetary) frameworks operationalize: recency monotonically increases until a new event resets it, and customer-data decay is one-directional under normal pipeline conditions.

Industry survey data anchors the realism of this model. Landbase's *Data Decay Rate Statistics* report (January 2026), drawing on a tracking study of 1,000 business contacts, puts annual B2B contact-data decay in the range of **22.5%–70.3% per year**, with 70.8% of tracked contacts experiencing at least one data change within 12 months. Validity's *State of CRM Data Management in 2025* (released July 10, 2025, surveying 602 CRM users and administrators across the U.S., U.K., and Australia) reported that **76% said less than half of their organization's CRM data is accurate and complete**, and 37% of CRM users reported losing revenue as a direct consequence of poor data quality. That is the steady-state condition this experiment is approximating.

**Modern data entry errors.** Automation artifacts. The error model I am *not* using is character-level typos ("Jonh" for "John"). In modern enterprise systems, character-level typos are increasingly rare because data entry is dominated by automated pipelines, form validation, dropdowns, and API integrations rather than free-text manual entry. The errors I *am* modeling:

- **Date format mismatches** — MM/DD/YYYY vs DD/MM/YYYY ambiguity. CloverDX's pipeline literature explicitly calls date formats "a classic source of error" in modern enterprise integration. `Jan 5` becoming `May 1` on the wrong side of a locale boundary is the canonical example.
- **Unit conversion failures** — dollars vs cents, days vs hours, scores on a 0–10 vs 0–100 scale.
- **Default value persistence** — a field that was seeded with a default during onboarding and never updated. The system thinks it's data; it's a placeholder that aged into "looking like" data.
- **Truncation at integration boundaries** — a field that got cut at a column-width limit during ETL.
- **Schema drift / null increase** — Microsoft Research's *Auto-Validate by-History* (arXiv:2306.02421, 2023) documents schema drift and *"a sudden increase of null, empty strings, or special values (e.g., -1) in a column due to external factors"* as primary failure modes in recurring pipelines, including a Google production incident where null values rose sharply because an upstream module hit RPC timeouts.

Same magnitude in absolute terms, very different distributional fingerprint. H4 asks whether the agent treats them differently.

### H5 — Detection Awareness

**Prediction.** The agent's reasoning text will not reliably indicate awareness of data quality issues, regardless of dither magnitude or type. This extends 1a's "agentic blind spot" finding from duplication (where the agent never noticed it had seen the customer before) to dithering (where the agent will not flag implausible values).

**Primary measurement.** Keyword / topical search across the agent's reasoning output for detection language. The seed list (will be expanded during pilot, then frozen before main run):

- "data inconsistent", "inconsistency", "contradicts"
- "unusual value", "seems implausible", "doesn't seem right"
- "contradictory signals", "conflicting", "suspicious"
- "outlier", "anomalous", "atypical"
- "verify", "double-check", "confirm"
- "data quality", "data issue", "missing context"

The detection rate is the fraction of dithered runs where the reasoning contains at least one phrase from the frozen list. Baseline detection rate (clean data) gives the false-positive floor.

**Secondary measurement.** Jaccard similarity between the dithered run's reasoning and the matched-customer baseline reasoning. Jaccard is a coarse token-overlap metric — not semantic — and I'm using it precisely because of that coarseness. If the agent meaningfully changes how it reasons about a dithered customer, Jaccard should drop noticeably. If Jaccard stays flat while the *decision* changes, that's the blind spot rendered numerically: same words, different verdict.

### H6 — Boundary Customer Vulnerability

**Prediction.** Customers classified as boundary-baseline (whose 5 clean-data runs do not all agree) will show disproportionate decision drift under dithering compared to stable-baseline customers (whose 5 clean-data runs all agree).

This is the direct generalization of 1a's boundary-customer finding. 1a showed that variation-driven inconsistency disproportionately hit customers near decision thresholds. H6 asks whether dither-driven inconsistency hits the same customers. If yes, "boundary customer" is a property of the customer × agent pair, not of the perturbation mechanism — and that property predicts where Decision Quality will degrade first as data quality slips.

---

## 3. Ground Truth Methodology

There is no oracle. Synthetic customers don't have an external "correct" priority level out in the world. Ground truth for 1b is defined operationally:

**Multi-run baseline.** Each customer is classified five times by the agent on the clean baseline data, at temperature 0.0, using the same prompt that will be used in evaluation. The majority vote across the five runs becomes the customer's baseline decision. The full distribution (e.g., 5-0, 4-1, 3-2) is preserved separately for boundary classification.

**Five runs at temperature 0.0.** Yes, temperature 0.0 should be deterministic. Empirically, on the Anthropic API, it is not perfectly deterministic — across model replicas, batch routing, and internal sampling stochasticity, identical prompts can still yield slightly different outputs. Five runs gives us enough signal to distinguish stable customers from boundary customers without burning budget on diminishing returns. This is self-consistency sampling (Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," arXiv:2203.11171, Google Research Brain Team, 2022, which reported gains of +17.9% on GSM8K, +11.0% on SVAMP, and +12.2% on AQuA over greedy-decoding CoT) applied to operational classification rather than to math problems.

**Baseline stability classification.**

| Class | Pattern across 5 runs | Interpretation |
|---|---|---|
| `stable` | 5/0 — all five agree | Agent has a confident, consistent read |
| `lightly-boundary` | 4/1 | One run dissents; agent is mostly settled |
| `deeply-boundary` | 3/2 | Near a decision threshold; small perturbations could flip the call |

(Theoretically 3/1/1 splits are possible across three categories. I expect these to be rare; if they appear, they get folded into `deeply-boundary`.)

**Same prompt for baseline and evaluation.** This is the most important methodological choice in the document and the one most worth being explicit about. The temptation is to use a richer or more careful prompt for ground-truth establishment, then a "production-like" prompt for the evaluation runs. That would be wrong. Production agents never get a heads-up that data is degrading. If we tell the baseline agent it's establishing ground truth, we're testing a different agent than the one we're claiming to study. The baseline must be the same agent under the same instructions, just on clean data.

**Temperature held at 0.0 for baseline.** Same reasoning. The classical self-consistency literature elevates temperature for the baseline runs to surface response diversity. We are not measuring response diversity in the abstract. We are measuring *this agent's* behavior under realistic operational conditions. Elevating temperature for baseline would establish ground truth for a different agent — one the user never deploys.

**Frequency distributions are evaluation-layer only.** The agent never sees the distribution. The 5-vote pattern is computed and stored after the fact and used downstream for stability classification. The agent on any given run sees one customer and one set of fields.

**On the 1a prompt design limitation.** 1a's prompt included illustrative examples in the priority level definitions ("e.g., a HIGH_PRIORITY customer might look like…"). On reflection, those examples may have anchored the agent's behavior in ways that contaminate the field-importance signal. 1b is a fresh start. I am going to remove or restructure those illustrative examples before the baseline run is committed. The specific revision will be documented in the prompt file's commit message before any baseline data is generated.

---

## 4. Data Generator Architecture

Refactor decision, made now and not later.

**Shared base generator.** A single canonical clean-data generator lives at `shared/data_generation/` (or wherever the repo's conventions settle). It produces a deterministic customer dataset given a seed. This becomes the input to every Phase 1 experiment.

**Experiment-specific extensions.** Each experiment owns a transformation layer that takes clean base data and produces experiment-specific conditions:

- `experiments/01_authoritative/1a_duplication/duplicator.py` — turns N customers into N+k records with controlled duplication patterns
- `experiments/01_authoritative/1b_dithering/ditherer.py` — turns N customers into N customers with controlled field-level perturbations

Same pattern, different transformation. Future experiments (1c missingness, 1d conflicting authorities, etc.) plug in the same way.

**Methodology note on 1a.** 1a's generator did not pass through this shared base. 1a results stand because the duplication mechanism didn't depend on the underlying base data being internally consistent — duplicating an inconsistent record is still duplication. But it does mean 1a's baseline customers and 1b's baseline customers will be different, and the cross-experiment comparison story needs to acknowledge that. The composite Decision Quality score is normalized within-experiment specifically so this kind of generator difference doesn't poison cross-experiment trend lines.

**Fixed seed.** The primary 1b run uses a single fixed seed, recorded in the experiment metadata file (`1b_metadata.yaml`). Robustness checks use 2–3 alternate seeds. The primary seed's results are the headline; the alternates exist to make sure the headline isn't a coincidence of one customer population.

**Strict internal consistency in the base data.** The base generator enforces internal correlations as hard rules (see §5). The reason: cleaner baseline data means cleaner signal-to-noise for the dither effects. Real enterprise data is noisier than this, and I will say so in the limitations. But noisy baseline plus noisy dither is two confounded variables; clean baseline plus controlled dither isolates the dither.

---

## 5. Data Quality Dimensions

Operational definitions for the base generator. These are principles, not field-level rules. Field-level enforcement (which exact ranges, which exact correlations) will be specified in the generator code and documented there, not pre-locked here. The six DAMA-canonical dimensions, interpreted for this synthetic context:

- **Completeness.** No null values in any field the agent considers. Real systems have nulls; missingness gets its own experiment (1c, deferred).
- **Consistency.** Internal correlations hold. If `is_at_risk == true`, then `churn_risk_score` must be above the at-risk threshold. If `last_purchase_days_ago > 365`, the customer cannot also be marked `recently_contacted_support = true` with a same-day timestamp. The generator enforces these as preconditions; no record ships with internal contradictions.
- **Validity.** All values fall within the field's defined plausible range and conform to format constraints (email regex passes, dates parse, scores are within their declared scale).
- **Accuracy.** For synthetic data, accuracy reduces to internal consistency plus distributional realism. There is no external reality to be accurate to. I'll say this out loud in §9.
- **Uniqueness.** No accidental duplicates in the base data. (1a deliberately introduced duplicates; 1b does not.)
- **Timeliness.** All timestamps reflect a coherent "as-of" date. The dataset has a single notional snapshot moment; no record is from the future or from before the customer's account creation.

These dimensions trace to the DAMA UK Working Group on Data Quality Dimensions, *The Six Primary Dimensions for Data Quality Assessment: Defining Data Quality Dimensions* (October 2013), led by Nicola Askham with contributors from Lloyds Banking Group, Aviva UK Health, Aston Martin, and Microsoft Corporation. It remains the de facto industry framing.

---

## 6. Dither Model

Two dither types, defined at the principle level. Field-by-field implementations live in the ditherer code; this section is the contract those implementations have to honor.

### Type A — Data Drift

Directional, asymmetric, time-driven. Modeled on what happens to a customer record sitting in a CRM that hasn't been touched in three months while the customer keeps existing in the real world.

- `last_purchase_days_ago` only grows.
- `churn_risk_score` tends to grow for disengaged accounts; the prior is upward.
- `total_spend` is monotone non-decreasing (a customer's lifetime spend can't shrink).
- `nps_score` regresses toward the population mean as the last response ages.
- Identity fields (email, phone) don't drift gradually — they either still work or they don't. So data drift for identity is modeled as binary staleness rather than gradual.

### Type B — Modern Data Entry Errors

Automation artifacts, not character-level typos. The five canonical mechanisms:

1. **Date format mismatch** — locale ambiguity producing `Jan 5` ↔ `May 1` swaps.
2. **Unit conversion miss** — dollars/cents, days/hours, 0–10/0–100 scale swaps.
3. **Default value persistence** — fields stuck at their onboarding seed value.
4. **Truncation** — field cut at an integration column-width boundary.
5. **Schema-drift surrogate** — for numeric fields, this surfaces as a sudden value-range shift; the closest in-distribution analog to what Auto-Validate by-History documented in production pipelines.

Magnitude tiers (§H2) apply orthogonally to both types. A 15%-magnitude data-drift dither and a 15%-magnitude data-entry-error dither will move the field's value by the same absolute amount, with different mechanisms generating that move.

Implementation details — which exact RNG distributions, which exact thresholds — get documented in `ditherer.py` and its module docstring, not here.

---

## 7. Evaluation Approach

How each hypothesis gets tested. Concrete metrics, with the analysis spec written down before any data exists.

**H1 (Field Importance).** Decision drift rate per condition, where decision drift = `P(dithered_decision != baseline_majority_vote)`. Compare drift rate across the seven H1 conditions at matched magnitudes. The top-5 fields should occupy the top of the drift ranking. If they don't, the stated-importance hypothesis fails.

**H2 (Magnitude).** Decision drift rate as a function of magnitude tier, within each field. Monotonicity tested via rank correlation (Spearman's ρ) between magnitude tier and drift rate. Plotted as drift-vs-magnitude curves per field.

**H3 (Internal Consistency).** Decision drift rate compared between correlated-dither and uncorrelated-dither arms at matched per-field magnitudes, for each pair and the triplet. The contrast effect (correlated minus uncorrelated, or vice versa) is the headline number.

**H4 (Dither Type).** Decision drift rate for data-drift dither vs. modern-data-entry-error dither, at matched magnitude tiers, holding field constant. Tested per field for fields where both types make sense (numeric fields). Also tested aggregated.

**H5 (Detection Awareness).** Primary: fraction of dithered runs whose reasoning contains at least one phrase from the frozen detection-language keyword list. Reported broken out by dither type, magnitude, and field. Secondary: mean Jaccard similarity (over token sets, lowercased, stopwords removed) between dithered-run reasoning and the customer's baseline reasoning, broken out the same way. A decoupling of decision change from Jaccard change is the empirical signature of the blind spot.

**H6 (Boundary Customers).** Decision drift rate broken out by baseline stability classification (`stable` / `lightly-boundary` / `deeply-boundary`). Expectation: monotonically higher drift in the boundary classes.

**Composite Decision Quality score.** Continues to be calculated. Specific component weights may shift relative to 1a because the things 1b can measure (e.g., per-field sensitivity) are different from what 1a could measure (e.g., per-record consistency). Cross-experiment comparison is maintained at the *principle* level — Decision Quality is always normalized so that 1.0 represents the agent's behavior on clean baseline data and 0.0 represents random assignment across priority classes.

---

## 8. Scope Boundaries / Deferred Items

What 1b is explicitly **not** doing:

- **System-of-record disagreement.** Two systems showing different values for the same customer field is its own problem and gets its own experiment. Deferred to **1d Conflicting Authorities**.
- **Multi-pillar combination effects.** What happens when data is *both* duplicated *and* dithered? Real-world data is, but isolating those effects requires the individual pillars first. Deferred to **Phase 1.5 cross-pillar synthesis**.
- **Multi-domain replication.** Customer prioritization is one domain. Whether 1b's findings generalize to, say, transaction triage or document review is a Phase 2 question.
- **Ground truth calibration against external reality.** Synthetic data only. The agent's behavior is measured against an internal baseline; we are not claiming the baseline decisions are "right" in any external sense.
- **Multi-model comparison.** Single model (`claude-haiku-4-5-20251001`), single temperature (`0.0`). Whether GPT-5.x or Sonnet 4.x or Gemini behave the same way is interesting, expensive, and out of scope here.
- **Prompt sensitivity sweep.** One prompt, frozen at baseline-run time. Prompt-as-variable is its own research program.

---

## 9. Known Limitations

Pre-emptive, in the spirit of writing a contract with future-self before the results bias the disclosure:

1. **Ground truth is an internal-consistency baseline, not external reality.** The baseline-majority-vote definition of "right" is circular in the sense that it's the same agent that we're then measuring against. This is a deliberate choice — we're studying the agent's stability under data perturbation, not its correctness against an oracle. But it bounds the claims we can make. We can say "dither moves the agent's decisions"; we can't say "dither makes the agent's decisions worse against ground truth."

2. **The modern data entry error model is a simplification.** Real automation failures are uglier, more correlated across fields, and more frequently undetected than my five-mechanism taxonomy captures. The five mechanisms are chosen to be representative and reproducible, not exhaustive. Microsoft Research's pipeline-failure taxonomy is broader; I am sampling from it, not implementing it.

3. **Strict consistency enforcement produces cleaner baseline data than real enterprise data.** Validity's 2025 CRM survey of 602 administrators found 76% reported less than half of their CRM data was accurate and complete. My base dataset is 100% internally consistent. That makes the dither signal cleaner, and it makes the dither effects an *upper bound* on what you'd see in real noisy enterprise data, where the agent is already partially adapted to background noise.

4. **Fixed-seed customer population limits generalizability.** Different customer populations — different industry mixes, different customer-lifecycle stage distributions — may show different field sensitivities. Robustness checks across alternate seeds will quantify this within the synthetic-customer space; cross-population validity is a Phase 2 question.

5. **Single-model behavior may not generalize.** Everything in this experiment is `claude-haiku-4-5-20251001`-specific. Other Haiku versions, other Claude tiers, other vendors will plausibly show different field-importance and detection-awareness patterns. The methodology is the deliverable; the specific numbers are illustrative of one model's behavior at one moment.

6. **Self-reported field importance from 1a is itself a measurement artifact.** The H4 top-5 from 1a is what the agent *cited*, which is not necessarily what the agent *used*. H1 partially tests this — but H1's "control" fields (identity, non-top-5 behavioral) are themselves a researcher choice, not an exhaustive comparison set. A field could be important without being cited; that case won't be detected here.

---

## 10. Methodology Decisions Made

Summary, for the future self who is going to read only this section and then start writing the generator:

| Decision | Choice | Why |
|---|---|---|
| Prompt for baseline | Same as evaluation prompt | Production agents never get a heads-up |
| Temperature | 0.0 throughout, no elevation for baseline | We want baseline for *this* agent, not a different one |
| Baseline runs per customer | 5 | Enough to classify boundary vs. stable; diminishing returns above |
| Aggregation | Majority vote, full distribution preserved | Vote drives ground truth; distribution drives stability class |
| Baseline customer population | New for 1b, not reused from 1a | Refactored generator; 1a generator didn't meet new consistency standard |
| Seed | Fixed primary seed + 2–3 robustness seeds, all documented | Reproducibility plus a check that headline isn't a one-seed artifact |
| Base data quality | Strict internal consistency enforcement | Clean baseline → clean signal-to-noise on the dither effects |
| Generator architecture | Shared base + experiment-specific transform extensions | One refactor now, paid back across every Phase 1 experiment |
| API mode | Anthropic Message Batches API | 50% cost discount on input and output tokens; latency tolerable for offline evaluation |
| Hypothesis count | 6 | Up from 1a's 4; H5 (detection awareness) is now first-class |
| Detection metric | Frozen-keyword reasoning search + Jaccard secondary | Coarse on purpose — the question is "does the agent notice at all" |
| Boundary classification | Stable / lightly-boundary / deeply-boundary | Direct generalization of 1a finding |
| Cross-experiment comparison | At the principle level (normalized Decision Quality) | Generator changes break naïve numeric comparison |
| Pre-registration | This document, frozen before any code | Contract with future-self |

---

*Last edit before code: any change to this document after the first ditherer commit must be recorded in `1b_DESIGN_AMENDMENTS.md` with a timestamp and a one-line justification. The whole point of preregistration is that the contract holds even when the results are inconvenient.*
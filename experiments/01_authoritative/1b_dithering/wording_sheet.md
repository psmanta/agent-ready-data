# Wording sheet, version 1: plain and technical, side by side

*Written before any results exist. Number slots in [brackets] are filled after the run. Every public statement about 1b comes from this
sheet; if a claim is not here, it is not said (macro-review Pass 3).*

**How to use it.** Lead with the plain wording. Use the technical wording where the reader will check it. Define a term the first time it
appears (the glossary says how). Every claim is scoped to "this agent, these fields, this data" and never says "proves".

---

## 0. Scope of the evidence (put this on page one of every piece)

- **A minimal agent.** One model call per customer; no tools, no memory between calls, no retrieval: the decision step of an agent, isolated. We
  chose this on purpose: tools and memory add variables, and in an agent with tools the tool outputs *are* the data (an agent that looks its
  customer tier up in master data inherits the master data's errors).
- **One model, one prompt.** `claude-haiku-4-5-20251001`, temperature 0, a fixed prompt that gives no decision criteria. The design works for any
  model; our numbers are for this one.
- **Synthetic customers.** 1,000 generated customers; no real people.
- **No outcome labels.** We measure how much decisions move, not whether they are right.
- **Funding and independence.** Self-funded with the author's personal API credits; no sponsor. The author works at Salesforce and uses the findings in his work, but the research is independent of it. [Confirmed by the author, 22 October 2026; final wording to approve before publication.]

---

## 1. The experiment in one paragraph

**Plain.** We gave an AI agent a thousand customer records and asked it, one at a time, to decide how urgently each customer needs
attention: high, medium or low. Then we made the data wrong in controlled ways (a number a little off, a number that makes no sense, two
fields that disagree) and watched what the agent did, compared with its own decision on the clean record. We did not ask whether its
decisions were right. We asked how much they depend on the data being right.

**Technical.** Experiment 1b of the Agentic Data Contract. 1,000 synthetic customers; `claude-haiku-4-5-20251001`, temperature 0, a fixed
prompt (hash recorded). The reference for each customer is the agent's own decision on the clean record, taken from five baseline runs spread
across different time windows. 55 dithered conditions vary the field (H1), the size of the error (H2), its consistency (H3), its believability
(H4), how many fields at once (H7) and which combinations (H8); H5 asks whether the agent voices doubt, H6 which customers are most exposed. A
zero-dither control is run in the same randomized blocks. The outcome is decision drift from the reference. A pilot of 130 customers (two small populations) informed the protocol,
and every change it prompted is logged with its reason; the design and the analysis rules are hash-frozen before the confirmatory run, on
customers who are not in the pilot; and every pre-registered result is reported.

---

## 2. The claims

### Claim 1: data quality earns its budget by protecting a decision

**Plain.** Fixing your data doesn't necessarily fix an AI agent's decisions. Data quality earns its budget where it protects a decision someone
depends on. For this decision, [x] fields carried it and the contact details did not [when we spoiled the contact details, only [x]% of decisions
changed; spoiling [field] changed [y]%]; for compliance, billing or fraud, those same fields may be everything. Map your decisions to your data,
fund quality where decisions depend on it, and keep the upstream gate for what the agent cannot catch.

**Technical.** H1, field-level and category-level conditions: drift under the identity condition (name, email, phone, address) was [x]%
(95% interval [a, b]) against [y]% ([a, b]) for the highest-ranked individual field. The wording is "does not matter *for this decision*":
we corrupt data and do not clean it, so we cannot say cleaning is worthless in general.

**Will not say:** that data hygiene is unnecessary; that cleaning a field is worthless; that any share of a record is "decoration" in general;
anything about fields we did not test; whether agents should or should not be deployed.

### Claim 2: the open slot, filled by whichever result is strongest

*(a) The cliff.* **Plain:** A cliff, not a slope: any error starts the movement, and bigger errors add little. **Technical:** H2, drift at the
smallest error size was [x]% and at the largest [y]%. *Say only if the 1a pattern actually recurs.*

*(b) The filter.* **Plain:** The agent catches the absurd but swallows the plausibly wrong. **Technical:** H4, drift under believable errors
[x]% against impossible errors [y]%; "impossible" means realistic automation artifacts (a flipped sign, a unit slip), not arbitrary nonsense.

*(c) Valid but wrong.* **Plain:** Data can pass every column-level check and still mislead the agent: a record that contradicts itself, or a
value that looks plausible, changes the decision. **Technical:** H3 (a consistent lie versus a contradiction between fields that normally move
together) with H4's believable arm: drift [x]% against [y]%; the confidence the agent reported was [lower/unchanged].

**Will not say:** that the agent "understands" anything; we see only what it does.

### Claim 3: unless you ask, the agent usually does not tell you

**Plain.** Unless you ask it to, an AI agent that decides from the data it is handed probably won't tell you the data is wrong. When the data was wrong, the agent said so in [x]% of cases, against [y]% when the data was clean. [It rarely flags it; it often
explains the odd value away.]

**Technical.** H5, explicit doubt in the reasoning on dithered records compared with clean baseline records. Confidence is reported
separately and labelled soft (model confidence is weakly calibrated). Refusal was not available: the agent returned a priority and a
confidence. Judge-based labels are provisional until a human audit.

**Will not say:** that the agent *cannot* detect problems. The claim is about unprompted behavior when no flag option or instruction to check is offered; what happens if you ask is untested (backlog: Experiment 1c).

### Claim 4: rank by impact, then decide what to fix first

**Plain.** To decide what to fix first you need three numbers: how much a wrong value in a field moves the decision (the impact), how often that
field is wrong in your data (the prevalence), and what it costs to fix. We measure the first. Work in the order of impact times prevalence, divided
by cost; the worksheet below leaves your two columns blank. Start with a map of which data each decision depends on and how much each field moves it
(a decision-to-data dependency map). Our impact ranking for this decision: [ranked fields].

Check the data before the agent reads it (at the point of use), not after it decides. Think in layers: check at the point of use; hand the agent only
the fields the decision needs (an idea we have not tested); keep a canary set, a few known-clean records mixed in with live traffic (a flip on a canary is the model's variance; a failed check is the data); and keep a person on the borderline and high-stakes cases, which are hit first [if H6 holds]. You
can measure the impact term on your own decisions with the same method, at small scale.

**As a contract, it has two sides and a third owner.** The data side (the engineer) asserts the payload: types, ranges, referential integrity, cross-field
consistency, freshness. These are deterministic, and the dependency map ranks them: blocking for the fields the decision depends on, warnings for the rest. The
decision side (the agent's owner) reports a stability figure measured on the canary set, versioned with the model snapshot, the prompt and the platform; when any
of them changes the map is stale and the sensitivity test is re-run. The business owner sets how much autonomy the agent has and where a person reviews. A
flip on a clean canary record is the model's variance; a flip that follows a failed assertion is the data.

Cost: [x]% of the money spent on decisions bought a decision the bad data changed, and finding the bad ones means re-running [y] decisions, about $[z].

| Field | Impact (measured by us) | How often it is wrong (yours) | Cost to fix (yours) | Impact x how often / cost |
|---|---|---|---|---|
| [field] | [x]% of decisions move | [ ]% | $[ ] | [ ] |

**Technical.** Impact ranking by H1 drift per field (exposure-adjusted: drift among the customers whose value actually changed), by error size (H2) and
type (H4); boundary customers (H6) show [x]% drift against [y]% for stable ones; breadth (H7) [saturates at ~k fields / grows steadily]; combinations
(H8) [amplify / add]. Direct costs: corrupted-decision spend and rework cost, from the logged token cost per decision. The control's drift of [w]% is
the no-dithering reference. Impact ranking only, for the fields tested; prevalence and remediation cost are not measured.

**Will not say:** how common these errors are in anyone's data or how costly they are to fix; rankings beyond the fields we tested; that a ranking by impact alone is a priority list; that removing a field from the agent's input is safe (untested; an ablation would test it); that a check prevents bad decisions; that a payload is "AI-ready"; why identical input sometimes gets a different decision (we do not know).

### Also found: the window effect

**Plain.** The same agent, with the same instructions and the same data, gave a different decision for [3 of 30] customers depending on when it
ran, and was perfectly consistent within each time window. [At 1,000 customers the control showed [x]%.]

**Technical.** Identical input hashes and token counts, the same dated snapshot, prompt and settings; 3 of 30 customers' majority decisions
changed between sessions (exact 95% range 2.1% to 26.5%), deterministic within a window. Not our code (the pre-patch agent gave the same
answers), not the libraries, not the data change under test. Cause unknown (serving side). One model, one API.

**Will not say:** why it happens; that it affects other models; anything stronger than "in this small check" until the control reports.

---

## 3. Glossary of assumed knowledge

| Term | Plain meaning | Technical meaning |
|---|---|---|
| AI agent | A program that reads information and decides what to do, here one customer at a time. | `business_decision_agent.py`: one model call per record, no memory between calls, returns a priority, a confidence and reasoning. |
| Priority decision | The agent's answer: high, medium or low urgency for a customer. | `business_decision` in {HIGH_PRIORITY, MEDIUM_PRIORITY, LOW_PRIORITY}. |
| Clean record / baseline | The customer's data exactly as generated, before we spoil anything. | `baseline_customers.jsonl`; five runs per customer, spread across windows. |
| Reference decision | What the agent decided on the clean record; the yardstick. Not "the right answer". | The majority decision of the baseline runs. |
| Dithering | Deliberately making the data wrong in a controlled way. | The perturbation engine; each condition fixes the fields, size, believability and prevalence. |
| Drift (a flip) | The agent's decision changed because the data changed. | Decision differs from the reference decision for a perturbed customer. |
| Perturbed | The value really changed. A value that could not move (a zero drawn downward) is not counted. | A record whose `_dither_fields` is non-empty; blocked records are excluded from every table. |
| Boundary customer | A customer the agent can't make up its mind about: the same clean record gets different decisions. | Baseline runs split across decisions (4-1 lightly, 3-2 deeply, or a tie). |
| Stable customer | One the agent decides the same way every time. | All baseline runs agree, across windows. |
| Window | A stretch of time during which the agent behaves consistently. | A scheduling block; behavior can differ between blocks for identical input. |
| Window effect | The same input, model and prompt giving a different decision at a different time. | Cross-window disagreement; measured by the clean re-run control. |
| Clean re-run control | The clean data run again, mixed in with everything, to measure how much the decision moves with nothing changed. | `h0_clean_rerun_1`: zero dithering, identical input hash, same randomized blocks. |
| Noise floor | How much the agent's answer moves even when nothing is wrong. | Within-baseline disagreement, and the control's drift. |
| Explicit doubt | The agent says in words that something looks wrong. | A frozen keyword scan of the reasoning text, with adverb forms. |
| Echo check | Did the agent repeat the odd number it was shown, convert it, or leave it out? | Per field: echoed, converted (x100, /12 and so on), or omitted. |
| Judge | A second AI that reads the first one's reasoning and labels it. Provisional until a person checks. | A frozen classifier prompt (PROMPT_ID recorded), applied to the 8 H4 conditions. |
| Pre-registration | Writing down what we will test and how, before we look. | The amendment and ledger, committed before data collection. |
| Freeze | Locking the files so nobody can quietly change the rules afterwards. | `freeze_manifest.py`: a hash of each file, written once. |
| Exact interval / "range" | How far the real rate could plausibly be from what we saw in a small sample. | A Clopper-Pearson 95% interval; Wilson for rates in the evaluators. |
| Temperature 0 | The setting that makes the model give its most likely answer. It does not guarantee the same answer across sessions. | Sampling temperature 0.0. |
| Model snapshot | A dated, fixed version of the model. | `claude-haiku-4-5-20251001`. |
| Token spend | What the AI calls cost, counted in words processed. | `cost_usd` logged per decision. |
| Check before vs after | Checking the data before the agent reads it (at the point of use), versus reviewing after it decides or hoping the AI notices. | A validation step against a contract at the agent's input assembly (preventive), versus post-decision review or reliance on the agent's own flags. |
| Canary set | A few known-clean records mixed in with live traffic, like a canary in a mine. A flip on a canary is the model's variance; a failed check is the data. | A control set with a stored reference decision per record, interleaved with live traffic; re-run on any model, prompt, platform or data change. The production form of 1b's clean re-run control. |
| Data quality vs decision quality | Data quality is the input; decision quality is the outcome. We measure one thing in between: how much the decision moves when the input is wrong. | Input conformance versus the dependence of decisions on that input; decision stability is necessary for decision quality, not sufficient. |
| Sensitivity vs correctness | How much decisions move versus whether they are right. We measure only the first. | Drift from the agent's own clean-record decision; no external ground truth. |
| Consistency (DAMA) | Every value fine on its own, but the row contradicts itself (country US, currency Euro). | A row-level rule across fields; our H3 and the isolated churn test. |
| Decision-to-data dependency map | For each decision an agent makes, which data it depends on and how much each field moves it. | A table of decisions by fields, filled with measured drift per field (H1) and by error size and type (H2, H4). |
| Decision stability | How little the decision moves when the data is wrong. The thing we measure. | One minus drift from the reference decision, for customers whose value actually changed. |
| Impact, prevalence, cost | The three numbers behind "what to fix first": how much an error moves the decision (we measure), how often it happens, what it costs to fix (yours). | Impact x prevalence / cost; our contribution is the first term. |
| Point of use | The moment an agent receives the data for one decision, however tangled the systems behind it. | The agent's input-assembly step: the record or context passed to the model. |
| Minimal agent | An AI that is handed one record and decides, with no tools, no memory and no lookups. The decision step of an agent, on its own. | One model call per record; no tools, no retrieval, no memory between calls; fixed prompt and output schema. |
| Agentic data contract (practical form) | A short, machine-checkable list of rules about the data an agent reads for one decision, ranked by how much the decision depends on each field, with a named owner on each side. | Payload assertions (deterministic: types, ranges, referential integrity, cross-field consistency, freshness), blocking or warning by measured dependency; a decision-stability figure on a canary set, versioned with the model fingerprint; an autonomy level set by the business. |
| Synthetic data | Customers we generated; none are real people. | Generated records; any resemblance of an email to a real address is coincidence. |

---

## 4. What can be quoted today, and what cannot

**Quotable now, scoped as a small check:** in a 30-customer check, 3 of 30 customers' decisions changed between sessions with nothing changed
(exact 95% range 2.1% to 26.5%); changing the email format changed 0 of 30 (range 0.0% to 11.6%).

**Not quotable until the run and the analysis:** every number in a [bracket] above.

## 5. Style rules

1. Say "the decision changed" or "the agent's decision moved", never "the decision was wrong".
2. Scope every claim: "this agent, these fields, this data".
3. "Consistent with", not "confirms", when comparing 1a and 1b: they share an agent, a prompt and a model.
4. A result that goes the other way gets the same space and the same wording, not a smaller footnote.
5. Define each glossary term the first time it appears in any piece.
6. **The compression test.** Squeeze every claim to one sentence and check it cannot be used to argue the opposite ("most of the record is
   decoration" fails; "data quality earns its budget by protecting a decision" passes). Claims get shortened and inflated by whoever retells them.
7. "Vanity metric" is used only about *counting* data quality in isolation (a dashboard showing 100% deliverable addresses), never about the work.
8. **Vocabulary (D33).** Name what we measure "how much the decision moves" (plain) or "decision stability" (technical). "Decision quality" is the
   frame and always carries "stability is necessary, not sufficient". Never "damage", "break", "hurt" or "wrong decisions" for a decision that changed.
9. **Beat 5 (D42)** reads "How do you decide what to fix first?" and the answer is the three-term worksheet, never a ranking by impact alone.
10. **Name the agent (D44).** Say "an AI agent that decides from the data it is handed", or "a minimal agent". Never "your AI agent" without that
    qualifier, and never imply the results cover agents with tools, memory or retrieval.
11. **Name the model (D46).** "One model (Haiku 4.5), with a design that works for any." Never "any model" or "AI agents" as if we tested them all.
12. **Never "certified AI-ready" (D47).** Say "the payload conforms to the stated rules". A data contract promises the data, not the decision.

---

## 6. What this does and does not mean for data-quality budgets

**It does not mean** that data hygiene, master data management, governance or catalogs are unnecessary, or that any part of a customer record is
unimportant in general. We measured one decision, made by one agent, on synthetic data. The same field can be irrelevant to one decision and
essential to another (an address to a priority score, and to a KYC check).

**It does mean** that the agent will not reliably catch bad data for you (to be confirmed by H5), so the control has to sit between the data and the
decision, which is what your data programs provide; that the value of data quality can be argued decision by decision, with evidence; and that you
can find out which of your fields your decisions actually depend on, rather than assume.

---

## 7. What a check at the point of use can and cannot do

**It can:** stop or flag a payload that breaks stated rules; be prioritized by which fields a decision depends on; produce evidence that the payload conformed;
and, with a canary set, separate "the data was wrong" from "the model moved on clean records".

**It cannot:** make the model deterministic; catch a flip caused by the model's own variance; guarantee that a decision is right; or certify a payload "AI-ready".

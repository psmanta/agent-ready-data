# Macro review: are we still answering the question we posed?

A plain-language review done before the freeze. **Ground rules:** no code and no statistics; anything that needs
them is parked as a design question. Every decision flows into the amendment and the ledger before anything is
frozen. The review has five passes; this file is the running decision log.

## The premise

Experiment 1a asked: *what happens to agentic decision-making when the data is duplicated?*
Experiment 1b asks: *what happens when the data doesn't tell the truth?*

A deceptively simple question hides three that a business reader actually has:

1. **Will my agent's decisions change?**
2. **Will it tell me?**
3. **What should I fix, and in what order?**

## Pass 1: the premise and the claim (decided)

**D1. The claim, in one sentence.** *"We measure how an AI agent's priority decision for a customer changes when the
data it sees is altered, compared with its decision on the clean record."*

**D2. The non-claim, in one sentence.** *"We do not measure whether the agent's decisions were right."* ("Right" means
different things to different people and agents, so we do not ask it.)

**D3. What "tell the truth" means here.** A value is made wrong by altering it from what the record originally said.
The reference is the agent's own decision on the clean record. So 1b measures **sensitivity to wrong data**, not how
often decisions end up wrong.

**D4. Priority guidance, and its limit.** The third question is "What should I fix? In what order?" We provide
**priority guidance only**: a ranking by how much each kind of wrongness moves decisions. We do **not** say how
common an error is in anyone's data or how costly it is to fix. That is a stated non-claim, so no one quotes us as
saying more.

**D5. The headline-claim rule.** Claims are what we want people to quote; the story is how we earn them.
- At most **four headline claims**.
- The story may have **as many charts as it needs** (1a: six charts earned three claims), under one rule: **every
  chart must serve a claim**. A chart that serves none is cut or moved to the appendix.
- Each claim should have the shape that worked in 1a: a comfortable belief overturned, what it costs, and what to do.
  The content is decided by the results, not before them.

**D6. Decision quality without defining "correct".** The framing "data quality is an input; decision quality is the
outcome" is fair, with one assumption made explicit and one word changed. The assumption: a decision that changes
only because the data was wrong is a decision the business did not intend, so the agent's decision on the unaltered
record is the best available reference. That is not a definition of a correct decision. The word changed: we show
that good data practices **protect** decisions from being changed by bad data; we cannot show they **improve**
decisions beyond what the clean data would have produced. Wording: *"Data quality is an input; decision quality is
the outcome. We measure how much the outcome depends on the input, using the agent's own decision on the true record
as the reference. We do not define what a correct decision is."* One subtlety for the red team: an agent that ignores
a field looks perfectly robust to errors in it, so "no change" does not automatically mean "good"; H1 helps read
those results.

**D7. The enterprise myth.** The belief that has actually been observed is the **umbrella belief**: *"improving data
quality automatically improves agent decisions"* (the "vanity metric" point: perfectly deliverable addresses do
nothing for a decision that never uses them). A **supporting belief**, *"agents naturally filter small or plausible
errors"*, has not been heard directly and is split into two testable parts: small and plausible errors do not change
decisions (H2; H4's plausible arm), and the obviously absurd is caught or filtered (H4's implausible arm; H5). The
claim has two kinds of evidence and they stay separate: the **empirical** claim is testable in this experiment; the
**sociological** claim (people believe it) is not, and rests on practitioner observation alone (the
poll idea was dropped, D32). The identity condition in H1 (name,
email, phone, address, all visible to the agent) is a direct test of the vanity-metric example, with one caveat: we
corrupt data, we do not clean it, so the wording is "does not matter *for this decision*", never "does not matter".
Candidate four-beat story (beats, not claims; Pass 2 makes each conditional on results): the belief; "it depends on
which data and how wrong"; "and the agent usually does not tell you" (H5); "so fix by decision impact, in this order."

## How the eight hypotheses answer the three questions

| Business question | Hypothesis | Plain-language question | What it resembles in real life |
|---|---|---|---|
| **Will my agent's decisions change?** | H2 | How wrong does it have to be? | stale data that slowly decays |
| | H4 | Plausible mistakes or impossible ones? | placeholder fills vs. unit and sign errors |
| | H3 | Do contradictions matter? | two systems merged badly |
| | H7, H8 | How many fields at once, and in combination? | a bad migration |
| **Will it tell me?** | H5 | When data is wrong, does the agent say anything, or carry on as if it's fine? | the last line of defense: does it raise a hand? |
| **What should I fix, in what order?** | H1 | Which fields matter most? | where to put data-quality effort |
| | H6 | Which customers are most exposed? | borderline accounts, where a small error tips the decision |

H5 and H6 are not kinds of wrongness. They describe the response (does anyone notice; who is exposed), and they are
the most likely to carry headline claims.

## What we do and do not test

**Tested (kinds of wrong):** values drifted by 5 to 100 percent; plausible entry errors (defaults, resets);
impossible values (x100, negatives); fields that contradict each other; many fields wrong at once.

**Not tested:** missing data; duplicates (that was 1a); wrong customer matching; mislabeled fields; deliberate
manipulation.

## Decisions from the Pass 2 discussion

**Decided by the owner:**
- **D8. Report every result.** The search for headlines is itself a bias; presenting every pre-registered result lets the
  headlines rise on their own.
- **D9. The beats are questions; the answers come from the data.** The structure holds whatever the results are; only the
  answers and their size change. If the answer is "no effect at all", the same beats still tell the story.
- **D10. Add a cost beat.** Between "does the agent tell you" and "what do you fix first": *what does it cost?* Cost can be
  direct or an opportunity cost (a valuable customer called low priority because the data was wrong). See the cost beat
  below; the analysis it needs is NOT yet pre-specified and is a Pass 5 item.
- **D11. H3 (consistency) is a headline candidate.** Pass 2 is revised with that emphasis.
- **D12. Field-ranking limits are written up, not hidden.** We rank the fields we tested; readers test their own fields.

**Proposed, awaiting the owner's confirmation:**
- **D16. The cost beat uses the simpler direct cost** (token spend). The richer opportunity-cost modeling needs financial
  assumptions about lost customers (a rabbit hole), so it is parked as an exploratory add-on and documented so that a
  future experiment can pursue it.
- **D17. Fix the email evaluation-awareness cue with Option 1.** Replace the `@example.com` emails with realistic
  consumer-provider emails with modest randomness (for example `bthomas4471@gmail.com`), written up as synthetic: any real
  address is coincidence and the data is never used for outreach. Option 2 (a prompt note about standardizing emails)
  was rejected: it points the agent at data manipulation and changes a frozen, hint-free prompt. **Why it matters more than
  it first seemed:** the engine's own wrong emails already use gmail, outlook, yahoo and icloud, so corrupting an email
  *removes* the example.com cue, a confound inside H1's identity test. This mitigates the cue; it does not prove the agent
  has no awareness (names, addresses and phones remain Faker-generated).
  **Implemented and verified:** consumer providers only (gmail, yahoo, outlook, hotmail, icloud, aol; no corporate domains,
  which could add a business-versus-consumer signal); the address is derived from a hash of the customer, so the random
  streams are untouched. Regenerating 200 customers with the old and new generator, exactly one field differs (the email);
  regenerating all 55 conditions, 50 are identical apart from the email, and the 5 that corrupt email draw different but
  equally valid values for their other fields (the engine's domain swap consumes a different number of random numbers).
  The name-to-email link moves from strict to bounded (the local part contains the surname).
- **D19. Email A/B smoke test, result.** 30 customers x 5 runs per arm, both arms run the same day, plus the original smoke as a
  drift check. **No sign the new emails mattered:** mean confidence 0.812 / 0.813 / 0.812, output tokens 248 / 245 / 244, and
  zero test or domain wording in any arm. The agent was near-deterministic within a session (0.0% disagreement among the five
  runs in both arms), so the one customer (1 of 30) whose decision differed between arms differed in all five runs; the exact
  95% range for the share of customers the email could change is 0.1% to 17.2%. **For context:** 2 of 30 customers changed
  between the original smoke and today's re-run with NO change at all (0.8% to 22.1%). The email change stays, for the
  confound reason as well as the awareness one. A 30-customer test can only rule out large effects. **Superseded in part by D24: the one customer that changed between the arms did not replicate, so the email effect observed is 0 of 30.**
- **D20. DECIDED (owner): add a clean re-run control.** None of the 55 conditions is a zero-dither re-run, and the baseline and the
  dithered conditions will be run at different times. D19 shows identical input can give a different decision in a different session, so
  some measured drift could be session wobble and not dithering. A zero-dither condition (1,000 calls, about $2.40; or two, one at each
  end of collection, about $5) measures session wobble directly and gives every drift metric a true no-dithering reference; the baseline
  and conditions are to be run in a tight window. Needs the engine (56 conditions), the amendment counts and cost arithmetic, the
  checker, the freeze list, and an analysis home for the control (likely with H6). Design details (one run or two, placement, how the
  evaluators treat it) wait for D21.
- **D21. OPEN: why did identical-looking inputs give different decisions?** Two distinct observations from the A/B run. (1) Same day:
  changing only the email flipped one customer from HIGH x5 to MEDIUM x5; all five runs in each arm agreed, so it is not sampling noise.
  The reasoning diverges from the first sentence ("a valuable customer" versus "a medium-value customer") and never mentions the email,
  which fits sensitivity to an irrelevant change in a close call, though stated reasoning is not proof of what drove a choice.
  (2) Then versus now, IDENTICAL input: two customers flipped from the original smoke to today's re-run, both toward HIGH, both now
  opening with "shows significant churn risk" instead of "presents a moderate priority case", and today's two arms agree with each other.
  That looks systematic. Free check: a provenance block (input hash, input token count, model string, timestamps) in the comparer.
  Cheap decisive test: a 30-call interleaved replication on the three customers (about $0.08). Implication to carry forward: "stable"
  in our design means the same answer across five same-session runs; this shows that is not the same as robust to an irrelevant change
  or to a different day. Irrelevant-field dithering (H1 identity, the unlinked fields) measures the first at scale; the clean re-run
  measures the second.
- **D22. Provenance result, and what it does and does not rule out.** Original smoke: 2026-10-03, 12:06 to 12:13. Re-run arms: B (new
  emails) 2026-10-14, 23:09 to 23:16; A (old emails) 23:17 to 23:24, eight minutes later. For identical records, OLD and A have the same
  input hash and the same input token count (30 of 30) and the same model string (claude-haiku-4-5-20251001). That rules out a change in
  our prompt text or data. It does NOT rule out what the rows do not record: temperature and other settings, the libraries that build
  the request, or how the model is served. A clue: today's runs are perfectly repeatable (no split votes in any of the 60 customer-arm
  sets), while the October 3 runs had split votes (for example 3 MEDIUM, 2 HIGH on one customer), so the two sessions differed in
  repeatability as well as in answers. The public model lifecycle shows no deprecation or swap of the snapshot. Still to check (free):
  git history of the agent and shared/agents since October 2; how the first smoke was launched (temperature); library versions and install
  dates. Still to run (about $0.08): the interleaved 30-call replication of the three flipped customers.
- **D22b. Free checks, results.** (1) Library installs: anthropic 0.121.0 and langchain-anthropic 1.5.4 were installed on August 9, before the
  October 3 smoke, so neither was upgraded in between (langchain-core's install date was not checked). (2) No --temperature flag appears in
  the shell history for the agent, so the default (0.0, as built) most likely applied both times; history only covers this shell. (3) One
  commit touched the agent or shared/agents since October 2: 3ff1fee on 2026-10-07, "honor --max_records, add --resume ...". So OUR AGENT
  CODE DID CHANGE between the smoke (Oct 3) and the re-run (Oct 14). The patch was intended to touch only the processing loop and
  arguments (the prompt is byte-identical and its tests passed), but that is not proof that nothing in the request path moved.
  Next: a static check of the commit (does the diff touch the lines that build the request?) and a bisect (the pre-patch agent versus the
  current agent on the same input, today, about $0.12). If the pre-patch agent reproduces the October 3 answers and the current one does
  not, the patch changed behavior and that is a bug to find; if both give today's answers, it is time or serving, not our code.
- **D23. DECIDED (owner): add a run fingerprint.** The decision rows record the model but not temperature, max tokens, the system-prompt hash,
  library versions or the code version, which is exactly what this diagnosis needed, and which a future model change and regression test
  will need too. Before the full run, write a small metadata file beside every decisions file (model snapshot, temperature, max tokens,
  system-prompt sha256, input file hash, anthropic and langchain-anthropic versions, python version, git commit, command line, start
  time). Held until the bisect finishes, so we test exactly the agent that produced the A and B runs. Needs a change to the agent (a
  collection-stage artifact) and its test.
- **D24. The bisect, and what it overturned.** *Static check:* commit 3ff1fee touched only business_decision_agent.py (not shared/agents); the diff lines
  matching the request path are a refactor of the input-hash computation (now a helper, same result: the input hashes matched on all 30
  records) and the new --resume code; nothing touching call_llm, the prompts, temperature, max tokens or the model. *Replication, three
  flagged customers, five runs per arm:* (1) BD0D: HIGH x5 in the original smoke, in A, in A-again, and with the PRE-patch agent;
  MEDIUM x5 only in B; and B-again (same input, today) HIGH x5. **The email "flip" did not replicate**; both emails give HIGH today.
  (2) 1725 and 6D9F: MEDIUM-leaning on October 3 (MEDIUM x3 / HIGH x2, and MEDIUM x5), and HIGH x5 in EVERY arm since October 14, including
  the pre-patch agent. **Conclusions:** our patch is exonerated (the pre-patch agent gives today's answers); the email had no detectable
  effect (0 of 30, exact 95% range 0.0% to 11.6%); and the model's behavior is deterministic within a window but differs between windows
  (October 3, October 14 at 23:09, October 14 at 23:17, today) for identical input, model string, prompt and settings: 3 of 30 customers
  changed between windows with no change in input (exact 95% range 2.1% to 26.5%). Cause unknown and not ours: it is on the serving side.
- **D25. DECIDED (owner): a window-aware collection design.** The five baseline runs are not five independent looks: they are one window, repeated.
  So (a) "stable" (five agreeing same-window runs) overstates robustness: 1725 would be "boundary" by the October 3 runs and "stable" by
  today's; (b) drift is the dithered decision in window X versus the baseline decision in window Y, so window effects are confounded
  with dithering unless the schedule prevents it. Remedies, cheapest first: spread the five baseline runs across different windows (no
  extra cost; stability tiers then reflect cross-window variation too); randomize the order in which conditions are run and interleave
  them (a randomized block design, so window effects become noise and not a per-condition bias; the agent is stateless, so this is
  feasible); the clean re-run (D20), at the start and the end; optionally a small clean control embedded in each batch. Needs a run
  scheduler and a splitter that routes rows back to their conditions. A Pass 5 decision, and a Pass 4 red-team question: "is your
  baseline itself reliable?"
- **D26. BUILT: D20, D23 and D25, together.** *Clean re-run (D20):* one control, h0_clean_rerun_1, interleaved with everything in the
  randomized blocks, so no separate start and end runs are needed; the hypothesis evaluators skip controls; its analysis is built with H6.
  *Fingerprint (D23):* a best-effort .meta.jsonl beside every decisions file, appended per session; 33 check groups in the agent's test,
  including settings recorded as used and a fingerprint that can never stop a run. *Window-aware collection (D25):* plan_collection.py
  (a randomized complete block: 48 blocks by default; every condition spread over all blocks to within one task; each customer's five
  baseline runs five blocks apart in different windows; tasks shuffled inside a block; deterministic from a seed; written once and hashed),
  run_collection.py (resumable, shardable, retries, needs --yes), collect_results.py (routes rows back with the original record_id plus
  collection_block; writes nothing for an incomplete condition), collection_common.py, and the control in the generator. A simulation in
  the tests shows the point: with conditions that have NO real effect, drift differs between conditions far less under this scheduling
  (spread 0.014) than when each condition has its own window (0.048), and subtracting the control recovers "no effect" only under the
  randomized schedule. About 61,000 calls; at 20 calls a minute that is about 50 hours for one worker, so shard it. **Remaining:** how the
  control is used in the primary tests (Pass 5), and its analysis, built with the H6 evaluator.
- **D13. Claim 2 is an open slot.** Its content is whichever "how the agent reacts" finding is strongest (see Claims below).
- **D14. "Raising a hand" is defined three ways, and the H5 claim is scoped to *unprompted* behavior.**
- **D15. "Absurd" means realistic automation artifacts** (a flipped sign, a unit slip), not arbitrary nonsense.

## Pass 2 (version 2): the story and the claims

Every statement below is written **before any results exist**: *if we see this, we say that; if we see nothing, we say
that instead.* **Every pre-registered result gets reported;** the headline claims only decide the spotlight. Each
hypothesis lists what we will **not** say.

### The story: five beats, each a question

1. **Does improving data quality automatically improve an agent's decisions?** (Its sibling: do agents quietly absorb
   small or plausible errors?) Opener: the question, posed rhetorically (D32).
2. **Does it depend on which data, and how wrong?** Size, believability and consistency of the error.
3. **Does the agent tell you?** Unprompted, does it say anything when the data is wrong?
4. **What does it cost?** *1a counted the cost of processing duplicates; 1b counts the cost of trusting bad data.* Token burn
   does not rise by itself in 1b (the same records are processed), so two direct costs are defined: **corrupted-decision
   spend** (the share of token spend that bought a decision the bad data changed, like 1a's wasted spend) and **rework
   cost** (because the agent does not flag problems, finding the bad decisions means re-running every decision that
   touched the bad data). Both use the token cost already logged per decision. The richer opportunity-cost model
   (customer value that lost attention) is parked as an exploratory add-on.
5. **How do you decide what to fix first?** An impact ranking and a worksheet (impact x prevalence / cost), and which accounts to protect (D42).

### Why we corrupt instead of clean

In a real system you start dirty, clean to "some" level, and never know what "some" means (formatting phone numbers
probably does nothing for a decision). We start from the **ceiling**: the clean record is the reference, corruption is the
only direction, and we never have to ask "how much cleaning did we do?" That turns "how clean is clean enough?" into
**"how dirty is too dirty?"**, a question we can actually answer. One honest caveat: it assumes the response to corruption
mirrors the response to cleaning. The wording is therefore "does not matter *for this decision*", never "cleaning is
worthless."

### The hypotheses, one at a time

**H1. Which fields matter?** *Spoil one field at a time: do the fields the agent relies on move decisions more? Do contact
details (name, email, phone, address) matter at all?*
- **If** a handful of fields drive decisions and contact details barely register: *"For this decision, most of the record is
  decoration. Perfect addresses did nothing."*
- **If** contact details DO move decisions: *"Data you assumed was irrelevant leaks into decisions"*, a headline in itself.
- **If** every field matters about equally: *"No field is safe to ignore"*, and beat 5 weakens; we say so.
- **If** nothing moves at the size tested: no claim about fields at this size; H2 covers bigger errors.
- **Will not say:** that cleaning a field is worthless in general.

**H2. How wrong?** *Does damage grow smoothly with the size of the error, or is there a tipping point?* (three fields, four sizes)
- **If** damage appears at the smallest error and then flatlines: *"A cliff, not a slope: any error starts the damage and
  bigger ones add little"*, the 1a "decision cliff" pattern, **only if it actually recurs**.
- **If** small errors are shrugged off but large ones break decisions: *"Partly true: agents absorb small errors."*
- **If** damage grows steadily: *"Bigger error, proportionally more change."* **If** nothing moves: *"For this agent and these
  fields, wrong values barely mattered."*
- **Will not say:** anything about fields other than the three tested.

**H3. Consistency: does the row tell a coherent story?** *The DAMA consistency idea: every column can be valid and the row
can still be wrong.* The illustration: a transaction whose country code says US and whose currency says Euro. Each value is
acceptable when you scan down its own column; together they contradict. Our test uses pairs of fields that normally move
together (for example churn risk and satisfaction, or spend and lifetime value) corrupted either **in step** (a consistent
lie) or **independently** (a contradiction), plus a test where a flag contradicts the score it should follow.
- **If** consistent wrong data moves decisions while the agent stays confident, and contradictory data makes it hesitate:
  *"The dangerous wrong data is the kind that looks coherent."*
- **If** contradictory data moves decisions MORE: *"Contradictions throw the agent rather than alert it."*
- **If** the reasoning shows which side it trusted (like keeping the country and ignoring the currency): *"When a row
  contradicts itself, the agent picks a side, and here is how."* (readable only through its reasoning; descriptive)
- **If** no difference: *"Consistency of the error did not matter."*
- **Combined with H4's believable values:** *"Data can pass every column-level check and still mislead the agent."*
- **Will not say:** that the agent noticed a conflict unless its words show it.

**H4. Believable or absurd?** *Does the believability of a wrong value matter, and does the agent filter the absurd?*
"Absurd" here means **realistic automation artifacts**: a flipped sign, a unit slip (times 100: percent versus fraction, cents
versus dollars; times 30: days versus months), a field reset to a default. It does **not** mean arbitrary nonsense, and
that matches the premise that today's data errors are automated rather than fat-fingered.
- **If** believable errors move decisions more than absurd ones: *"The agent catches the absurd but swallows the plausibly wrong."*
- **If** absurd values move decisions MORE: *"Agents do not filter garbage; it throws them."*
- **If** there is no difference: *"The agent treats absurd numbers like real ones."*
- **Will not say:** that the agent "understands" anything; we see only what it does.

**H5. Does the agent raise its hand?** There are three ways it could, and they are not equal:
(1) **say so in words** (an explicit caveat in its reasoning); (2) **hedge** (a lower confidence score); (3) **refuse.**
In our setup (3) is **not available**: the agent must return a priority and a confidence, with no escalate or refuse
option. Our primary measure is (1), explicit doubt in the words, compared with how often it does so on clean records.
Confidence (2) is reported separately and labeled soft, because model confidence is weakly calibrated.
- **If** doubt is rare: *"Unprompted, the agent almost never flags wrong data."*
- **If** it rarely flags but explains the odd value away (a 115-year tenure becomes "long-tenured"): *"It does not flag it;
  it rationalizes it."*
- **If** it flags often: *"The agent is a useful last line of defense"*, and beat 3's answer flips.
- **Will not say:** that the agent *cannot* detect problems. The claim is about **unprompted** behavior when no flag option
  is offered. Does the answer change if we ask, or provide an escalate option? A natural follow-up experiment, parked.
- **Evaluation awareness:** the prompt frames a real business task with no hint of testing, which is the intent. Two
  things to check in the red-team pass (see parking lot).

**H6. Who is most exposed?** A "boundary customer" is one **the agent cannot make up its mind about**: given the same clean
record five times, it does not always give the same priority (a 4-1 or 3-2 split, or a tie). *(How wording varies between
those runs is measured separately, as coherence, and reported alongside.)*
- **If** the pattern repeats across most conditions: *"Wrong data hits the accounts the agent was already unsure about first;
  protect those."*
- **If** it does not repeat: no targeting claim.
- **Will not say:** anything from a single condition. The claim rests on a pattern that repeats, and the boundary groups
  are small, so we say so.

**H7. How many fields at once?** *Does damage keep growing as more fields go wrong, or level off?*
- **If** it levels off: *"After some point, more wrong data adds little."* **If** it grows steadily: *"More wrong, more
  change."* **If** erratic: *"The response becomes noise, not direction."*
- **Important limit:** our ladder adds the *important* fields first, so "more fields" and "more important fields" move
  together. A reader with 1,000 "ignore_me" columns would not see the same curve. We say so, and a small control that
  piles on *unimportant* fields is a Pass 5 option.
- **Will not say:** where the leveling-off point is for other agents, prompts or datasets.

**H8. Do some combinations amplify?** *The 1 + 1 = 3 test.*
- **If** a combination amplifies: *"Some combinations are worse than the sum; prioritize them."*
- **If** damage simply adds up: *"No special combinations; fix field by field."*
- **If** a pair we expected to be harmless amplifies anyway: a headline in itself.
- **Will not say:** that the two overlapping examinations of the same territory are independent confirmations.

### The claims: four slots

1. **Data quality for its own sake is a vanity metric; decision quality is the outcome.** (H1, with the "start at the
   ceiling" framing.) Wording follows D6: good data practices *protect* decisions.
2. **OPEN SLOT: how the agent reacts.** Whichever is strongest decides the content: (a) *the cliff* (H2); (b) *the filter:*
   catches the absurd, swallows the plausible (H4); (c) *valid but wrong:* passes every column-level check and still
   misleads (H3 with H4's believable values). We hold the slot; the results fill it. We do not know the answer, and
   there may be a finding we have not thought of.
3. **Unprompted, the agent usually does not tell you.** (H5.) Scoped as above.
4. **Fix by decision impact, and here is what it costs.** (H1, H6, H7/H8, and the cost beat.) Priority guidance only:
   ranked by how much each kind of wrongness moves decisions, for the fields we tested.

### The storyboard (candidate charts; every chart serves a claim)

| # | Chart | Claim |
|---|---|---|
| 1 | The belief, posed as a question (no poll data) | 1 |
| 2 | Field by field: how much each wrong field moves decisions (contact details at the bottom, if so) | 1, 4 |
| 3 | How wrong: the damage curve as errors grow | 2 |
| 4 | Believable versus absurd | 2 |
| 5 | A consistent lie versus a contradiction | 2 |
| 6 | Does the agent say anything? | 3 |
| 7 | What it costs: token spend that bought a corrupted decision, and the cost of re-running | 4 |
| 8 | Piling on: how many fields, and which combinations | 2, 4 |
| 9 | The priority list, and which accounts to protect | 4 |

### If the story does not land

If nothing moves under any condition, the beats still hold, with different answers: the belief is *not* disproven in the way
we expected, it does not depend on which data, and the cost beat has nothing to count. It would support the filtering belief
and weaken "fix data first", scoped to this agent, these fields and these error sizes. The worst outcome is not a null; it
is a claim written after the fact.

### New items for Pass 4 and Pass 5

- **D18 / cost beat (Pass 5):** pre-specify the two direct costs (corrupted-decision spend; rework cost) as descriptive
  quantities from the logged token cost per decision. The opportunity-cost analysis stays parked.
- **H7 control (Pass 5):** a ladder of *unimportant* fields, the miniature "1,000 ignore_me columns" test.
- **Which side wins (Pass 5):** a pre-specified descriptive look at which of two contradicting fields the agent used.
- **Evaluation awareness (decided in part, see D17):** the email cue is being fixed at the data level. Still to do: a
  descriptive diagnostic in the H5 output (how often the reasoning calls the data synthetic, fake, placeholder, demo or test
  data), reported either way; and, optionally, a small before/after smoke comparison (about 150 calls) with old and new emails.
- **A follow-up (parked):** does the agent flag problems if asked, or if given an escalate option?

## Pass 3: the audience lens (DRAFT, for your reaction)

Four audiences, three takeaways each, written **before any data exists**. Each is tested on four questions: is it insightful, is it
surprising, can the reader act on it, and can we say it honestly today? **Status** says what a takeaway rests on: **observed** (we already
have evidence), **pending** (the full run decides) or **method** (about how we did it, true whatever the results).

### How one experiment serves four audiences: three layers and one source of truth

- **Layer 1, thirty seconds:** the four claims (Pass 2) plus one process finding, in plain words. LinkedIn readers.
- **Layer 2, five minutes:** the story told in about nine charts, each serving a claim. Salesforce colleagues and their customers.
- **Layer 3, as deep as you like:** the amendment, the ledger, the code, the tests, and later a paper. Data scientists and researchers.
- **One rule keeps the layers consistent: no claim appears in any audience's material unless it is in the claims table,** with its plain
  wording and its technical wording side by side. The table is the source of truth; the posts, the deck, the README and the paper are views of it.

### 1. LinkedIn: the casual reader, technical or not

Aligned to the five beats (D29). One post cannot carry all five, so treat this as a skeleton: the splash post carries beats 1 to 3 and a
pointer to the repository; follow-ups carry the rest.

| Beat | The line, in plain words | Rests on | Status | Honest read |
|---|---|---|---|---|
| 1 and 2 | "Fixing your data quality doesn't necessarily fix your AI's decisions. It depends on which data, how wrong it is, how believable it looks, and whether it contradicts itself." Opens with the question, posed rhetorically (D32). | Claim 1; the open Claim 2 (H1 to H4) | pending | The second sentence is beat 2 made explicit, so readers do not expect field choice alone. Consistency (H3) is the line most likely to resonate if it works: "the wrong data that looks coherent is the dangerous kind." Today we can only pose the question. |
| 3 | "Unless you ask, an AI agent probably won't tell you the data is wrong." If H5 supplies the twist: "it doesn't flag it, it explains it away." | Claim 3 (H5) | pending | It echoes 1a, so the twist is what makes it travel. Only H5 can supply it. |
| 3, also found | "...and to make it worse, the same agent can decide differently depending on when it runs." | the window effect | observed (30 customers: 3 changed; the full run measures it on 1,000) | One line, labelled a small check. The full treatment belongs to the Timely pillar (D27). |
| 4 | "It's probably costing you money." | the cost beat (D16 to D18) | pending | Two direct costs: token spend that bought a decision the bad data changed, and the cost of re-running everything that touched the bad data. Number slots to fill. |
| 5 | "Check the data before the agent reads it, not after it decides. Then use impact x how often it is wrong / what it costs to fix to decide what to fix first." | Claims 3 and 4; continues 1a | pending | The same answer as 1a (an upstream gate, not a downstream decision gate), reached from a different kind of data failure. Say "consistent with", not "confirmed": the two experiments share one agent, prompt and model. If H5 shows the agent flags often, this line weakens and we say so. |

### 2. The data scientist on GitHub

| # | Takeaway | Rests on | Status | Honest read |
|---|---|---|---|---|
| 1 | "Everything we claim was written down before the data existed": a hash-frozen pre-registration, and every result reported. | the freeze, the amendment, the ledger | method | A strong differentiator, but the freeze has not happened yet, so we say it only after it does. |
| 2 | "Drift is measured net of the clock": a randomized block design plus a clean re-run control. | the collection design | method (motivated by an observed finding) | The strongest methodological contribution. A simulation in the tests demonstrates the point; the full run tests it at scale. |
| 3 | "This measures sensitivity to wrong data, not correctness, and the limits are stated": one model, one prompt, synthetic customers, the agent's own clean decision as the reference. | D1 to D3 | method | Honesty about limits is what lets this audience trust the rest. It invites feature branches: other models, other domains. |

### 3. Salesforce colleagues presenting to customers

| # | Takeaway | Rests on | Status | Honest read |
|---|---|---|---|---|
| 1 | "Spend data-quality effort where it moves decisions": a priority list, with the vanity-metric example. | Claims 1 and 4 (H1, H6, H7, H8) | pending | The most useful deliverable for customers. Limits to say aloud: the ranking covers the fields we tested, and it is priority by decision impact, not by cost or prevalence. |
| 2 | "Check the data before the agent reads it. Don't count on the agent to raise its hand." | Claim 3 (H5), extending 1a's upstream-gate claim | pending | Consistent with 1a. If H5 shows the agent flags often, this flips, and we would say so. It is also the LinkedIn beat-5 line and a paper takeaway (D30). |
| 3 | "Keep a canary set": a fixed set of records run regularly to catch decision drift when the model, the platform or the data changes. | the window effect, the fingerprint, the clean re-run | observed and method | **New: it is not in Pass 2.** It is practical, memorable and derived from our own method. It can be said once the control quantifies drift at scale. |

### 4. The research community

| # | Takeaway | Rests on | Status | Honest read |
|---|---|---|---|---|
| 1 | "A pre-registered, fully reported test of how LLM-agent decisions depend on the type, size and consistency of data errors." | the claims table, D1 to D5 | method and pending | The contribution is the design plus whatever the results say; null results are reported. |
| 2 | "Identical input, same dated snapshot, same settings: different decisions in different windows, deterministic within a window." Quantified with a controlled design. | the window effect | observed (n=30), pending at scale | The most paper-worthy standalone finding: temperature 0 does not guarantee the same answer across sessions, and five same-window runs are one look, not five. Caveats: one model, one API, cause unknown. |
| 3 | "What we found and fixed before running": the ledger of defects (the email cue, the validator interval bug, the completeness scan, and others) and the known limits of the instruments. | the ledger, the fingerprint | method | Transparency invites replication. It also says plainly that the judge's labels are provisional until a human audit. |

### What the lens revealed

1. **The window effect works for all four audiences, for four different reasons:** a hook (LinkedIn), a method (data scientists), a practical
   habit (the canary set), and a finding (the paper). It is positioned as an "also found" item (D27).
2. **There is a takeaway the claims table lacked: the canary set.** To be added to Claim 4 as the "what to do about it".
3. **"Put the check where the data enters" is not one audience's line; it is the series' answer** (D30), the recommendation that ties 1a and 1b.
4. **The audiences disagree about what the finding is:** surprise (LinkedIn), action (Salesforce), method (data scientists), a quotable limit
   (researchers). So each claim needs two wordings side by side, which is the wording sheet.
5. **Danger zones:** (a) "AI is unreliable" overreach, which D2 forbids since we do not measure whether decisions are right; (b) selling a
   30-customer check as established; (c) the window effect overshadowing H1 to H8; (d) rankings quoted beyond the fields we tested; (e) calling
   1a and 1b "two confirmations" when they share an agent, a prompt and a model.
6. **Number slots to fill after the run:** the share of decisions the control shows changing with nothing changed;
   the field ranking; how often the agent voices doubt; the two direct costs.

## Decisions from the Pass 3 discussion

**Decided by the owner:**
- **D27. The window effect is an "also found" item.** Its full treatment belongs to the Timely pillar, where it is in the backlog.
- **D28. The first public piece follows 1a's pattern:** the full run, the analysis, a quiet commit to GitHub, then the LinkedIn post as the
  first splash with a pointer to the repository; the results then go into the external and internal Salesforce decks to complement the
  "AI-ready" positioning. The research paper comes after a whole pillar concludes. No teaser before the run.
- **D29. LinkedIn material aligns to the five beats.** Beats 1 and 2 are stated together, and the "it depends" sentence names every dimension
  (which data, how wrong, how believable, whether it contradicts itself), not field choice alone; the window effect rides along as "and to
  make it worse".
- **D30. "Put the check where the data enters" is a cross-audience recommendation** (LinkedIn, Salesforce, the paper), continuing 1a's
  upstream-gate finding. It is conditional on H5, and it is "consistent with", not independently confirmed, because 1a and 1b share the
  agent, the prompt and the model.
- **D31. A wording sheet is created:** plain and technical wording side by side, plus a glossary of assumed knowledge. Version 1 exists now
  (wording_sheet.md); the number slots are filled in Pass 5 and after the run.

- **D32. No poll: the question is a rhetorical opener.** A poll would have to run during the collection window and be remembered two days
  later, which few readers do; it also breaks 1a's pattern of staying quiet until the splash. So the belief ("if you fixed 100% of your
  data quality tomorrow, would your AI make better decisions?") is posed as a question, not measured. Consequence for wording: the claim that
  people hold this belief rests on practitioner observation alone, so LinkedIn may say "a belief I hear constantly" and the paper must say
  "a commonly encountered assumption (practitioner observation, not measured)". No audience is told a percentage of people believe it.

## Pass 4: the red team (IN PROGRESS)

Role play: an actuary from a 90-year-old mutual insurer, thirty years of GLMs and survival curves, three hype cycles survived, who has
read the commit log. Each question gets four things: what is **right** about it, what it **misses**, the **booth answer** (thirty seconds),
and what we should **change**. Questions arrive from the owner; this is the first.

### RT1. "You are measuring sensitivity and calling it decision quality. Where is your ground truth?"

*The objection, in his words:* a 5% nudge that moves a borderline policyholder from Tier 2 to Tier 3 is continuous sensitivity, not a corrupted
decision; without outcome labels there is no Brier score, no calibration error, no loss; so this is input perturbation of a non-linear black
box, and "why should an executive care that an agent changed its mind if you cannot say which decision lost money?"

**What is right.**
1. **We use "decision quality" for something we do not measure.** The frame says decision quality is the outcome; D2 says we do not measure
   whether decisions are right. What we measure is how much decisions *move* when the input is wrong: **decision stability**. Stability is
   necessary for decision quality, not sufficient. The two words are doing different jobs and the public wording blurs them.
2. **We have not shown a changed decision is worse.** No outcome labels, no loss, no Brier score, no calibration error.
3. **Some sensitivity is correct.** A model should respond to a real change in a real signal.
4. **Our own plain-language drafts say "damage", "break" and "hurts".** Those words presume harm. They have to go (see changes).

**What it misses.**
- **We do have ground truth, for the input.** We know exactly which values are wrong, because we made them wrong. We lack ground truth for the
  outcome, as does everyone at the moment bad data arrives; outcome labels come late or never in agentic operations, while data errors arrive
  daily. The question is "does the decision depend on information we know to be false", an error-propagation question, and input sensitivity
  and stress testing is ordinary model-validation practice. It comes before the accuracy question; it does not replace it.
- **The threshold-crossing case is the one we separate out.** The primary analysis of the detection tests (H4's garbage filter and H5) is on **stable** customers (H1 to H3 are NOT yet stratified; see Revision 3 draft R3-9) (the same clean decision on every
  baseline run, across different time windows). Boundary customers flip on their own (a deeply boundary customer about 40% of the time with no
  corruption), so they are reported separately, never counted as corruption effects, and never discarded: they are H6's question. His
  "Tier 2 to Tier 3 for a policyholder near a threshold" is H6's population, not the headline population.
- **"Continuous sensitivity" is a hypothesis we test, not an assumption.** H2 asks whether the response to a growing error is smooth (his GLM
  world) or a cliff or a threshold; H4 asks whether it matters that a wrong value is believable.
- **A GLM is a function; this is not.** Same input, same output, so sensitivity is a property you can differentiate. Our agent gave different
  decisions for identical input in different windows (3 of 30 customers in a small check; the clean re-run control measures it at 1,000), so
  sensitivity has to be measured, not derived, and the control gives the no-dithering floor.
- **There is a label-free cousin of calibration.** Does the agent's stated confidence move when the data is wrong? If the decision moves and the
  confidence does not, the confidence carries no information about the reliability of the input. H3 reports confidence beside drift.

**On "which decision lost them money".** We cannot say, and we will not claim to. We give the middle of a product: **(error rate in a field,
which the executive knows) x (the flip rate for that field, which we measure) x (the cost of a wrong decision, which the executive knows)**.
The only dollars we measure are the direct token costs (D16 to D18); opportunity cost is parked.

**Why an executive should care without a loss number.** (1) *Defensibility:* a decision that changes with bad input, or with the clock, cannot be
explained to a regulator or a customer ("why was the same risk placed in a different tier on Tuesday?"). His own profession's data-quality
standard, ASOP No. 23, asks for data to be reviewed for reasonableness with particular attention to internal consistency, which is H3's
subject. (2) *Stability is the leading indicator that exists today;* outcome labels are a lagging one. (3) *Every flip has a cost whichever way it
goes:* review, rework, complaints.

**The booth answer (about thirty seconds).** "Fair, and we say so on the first page: we do not measure whether the decisions are right; we have
no outcome labels. We measure how much the decision moves when an input is replaced by a value we know is wrong. We have ground truth for the
input, because we made the error. You would call it an input-sensitivity test. Two things make it more than a nudge: our main group for the detection tests is customers nowhere near a threshold (true for H4 and H5 today; for H1 to H3 only once R3-9 is done), and the same record can get a different decision on a different day, so it is not a smooth function you can
differentiate. What it gives you is the middle of the product: your error rate, times our flip rate, times your cost of a wrong tier."

**What we should change (proposals; D33 awaits the owner).**
1. **Vocabulary (D33):** call the measured quantity *decision stability* (or "how much the decision depends on the data") in every claim,
   the README and the paper; keep "decision quality" only as the frame, with "stability is necessary, not sufficient" attached. Retire "damage",
   "break", "hurts" and "wrong decisions" for decision changes ("the decision moved", "changed").
2. **A "What this does not show" box on page one** of the README, the deck and the paper: no outcome labels; no claim that a changed decision is worse.
3. **Report the direction of each change** on the ordinal scale (up or down in priority), descriptive. We record that a decision moved, not which way.
4. **Report the confidence response beside every drift result** (check that the evaluators already do, in H3 and H5).
5. **A "bring your own loss" worksheet** after the results: the three-factor product, with our measured flip rates filled in and theirs left blank.
6. **Optional, exploratory:** a *stipulated-policy* analysis: pick an explicit, published rule applied to the clean record as a stand-in
   truth and report whether errors push the agent toward or away from it. No new API calls. Risk: "your rule is made up", so it stays
   labelled exploratory.

**Will not say:** that a changed decision is worse; that drift is "error" or "damage" without "relative to the agent's own clean-record
decision"; anything about outcomes or money beyond direct token cost.

### RT2. "Hash-frozen before any data? You added instruments after the smoke runs, and you will report the strongest of three."

*Fired by the assistant at the owner: a clinical-trial statistician who writes analysis plans.* Her charges: (1) revision 2 added the echo check,
the floors, the judge and the inertia measure after the smoke runs, and the ledger shows the keyword list being fixed while looking at output,
which in her world is a protocol amendment after an interim look; (2) Claim 2 is an "open slot" filled by whichever of three candidates is
strongest, across 55 conditions, three strata and a dozen outcome measures, which she calls a garden of forking paths with a hash stamped on it.

**What is right.**
1. **The instruments were added after a pilot of 130 customers (two small populations).** The amendment says so (its status line, and the headings "found after the first
   smoke runs" and "instruments added after the first smoke runs"). The wording sheet overclaimed ("hash-frozen before any data was collected")
   and so did one sentence in the amendment ("before any data existed"); both are fixed. The claim we can make is narrower and true: *a pilot
   informed the protocol; the confirmatory analysis is locked before the confirmatory data exist.*
2. **"Whichever is strongest" is a selection with no pre-specified rule.** The largest of K noisy estimates is biased upward (the winner's curse),
   and "the agent tells us" mistakes who is speaking: the data supplies K numbers and we choose one. Reporting everything is necessary, not sufficient.
3. **A hash proves when the rules were fixed and that nobody changed them. It says nothing about whether the rules limit forking paths.**

**What it misses.**
- Pilot-informed protocols are normal practice; what matters is that confirmatory analysis is locked before confirmatory data exist. We have a
  two-stage freeze (collection, then analysis) with a commitment that no dithered result from the main run is examined before the analysis
  freeze. The pilot customers are not in the confirmatory set (checked: none of the 130 pilot customers appears in the planned main run, seed 20261016 with n = 1000).
- Each hypothesis has pre-registered primary tests; everything else is labelled exploratory; the judge is secondary and provisional; stable
  customers are the primary stratum. Multiplicity is handled within pre-specified families (for example the H4 garbage-filter family is
  Holm-adjusted). **But a one-page multiplicity map across all hypotheses does not yet exist,** and she will ask for exactly that.
- The ledger documents every change with its reason, which is the audit trail she would want.

**The booth answer (about ninety seconds).** "Yes: revision 2 came after a pilot of 130 customers in two small populations, and the amendment says so on its first page. I do
not claim the protocol was written blind. I claim the confirmatory analysis is locked before the confirmatory data exist: the customers are new,
the pilot ones are not in it, and a second freeze locks the analysis before anyone looks at a dithered result from the main run. Every change the
pilot prompted is in a ledger with its reason. Each hypothesis has one pre-registered primary test and everything else is labelled exploratory.
The open claim slot is not an inference; it is a spotlight, and it is filled by a rule written in advance: the largest effect among the primary
tests that survive correction and hold in both random halves of the collection; if none do, the claim is that no mechanism stood out. And the
hash is evidence the rules did not move, not evidence they are good; the rules are what limit forking." *(The last two sentences describe
decisions D34 and D35, which are still proposals.)*

**What we should change (proposals; the owner decides).**
- **D34. A pre-specified selection rule for the open Claim 2 slot,** written before the freeze: the largest effect among the pre-registered
  primary tests that survive correction within their family and hold in both random halves of the collection (halves defined BY CUSTOMER, a seeded pre-specified random partition, so every paired comparison and every baseline reference stays inside one half; splitting by block would break the pairing); if none qualify, the claim is "no mechanism stood out". Show all K primary effects side by side, with the label "largest of K".
- **D35. A one-page multiplicity map:** one row per hypothesis: primary endpoint, stratum, family, correction, K.
- **D36. A "protocol history" paragraph in the freeze section of the amendment:** the pilot informed the protocol; the confirmatory customers are
  disjoint from the pilot's; what is frozen when; everything not primary is exploratory.
- **Booth discipline:** separate *stimulus* changes (the emails, made before looking at any outcome) from *instrument* changes (made after reading
  pilot outputs). The email story answers a different charge: use it as an illustration of the first, never as the answer to the second.

**Owner's rehearsal answer, assessed.** *Landed:* honest tone and the tradeoff framing ("live with a suspected inadequate analysis, or sanity-check
it"); the email story as evidence of disciplined, minimal peeking; the admission that results are for this data and this agent. *Did not land:*
the email story answers a stimulus charge, not the instrument charge; "let the agent tell us" concedes the selection point; the generalization
disclaimer answers external validity, which she did not ask about. *Missing:* the words "pilot", "primary versus exploratory", "multiplicity"
and "selection rule".

**Owner's second attempt, assessed.** *Better:* the pilot-as-assay-calibration analogy (fix the cuff before the main trial) is a good one; the
disclosure is now explicit; "an anomaly across 55 conditions generates a hypothesis for the next experiment, not a p-value" is exactly right for
exploratory results. *Still wrong, three ways.* (1) **"We are explicitly not running Null Hypothesis Significance Testing" contradicts the
amendment,** which pre-registers McNemar, Mann-Whitney and Wilcoxon tests with Holm correction (21, 8, 5 and 4 mentions) and 33 "primary" labels.
Saying it would undercut the pre-registration itself: if nothing is confirmatory, the freeze is theater. (2) **"Priority guidance only" is a scope
statement about cost and prevalence (D4), not a statement about statistics;** the two were conflated. (3) **A descriptive ranking does not escape
the winner's curse.** With about 1,000 customers per condition each rate carries an interval of a few points, so near-ties are not a ranking and the
top of a ranked list of 55 noisy estimates is biased upward. Also: "strictly to calibrate" overclaims, because the amendment itself says the pilot
"showed that explicit detection is rare" and that both genuine detections "came from cross-referencing a sibling field", which shaped the
isolated-corruption arm and the judge; those are labelled "hypothesis, not finding". And "all evaluators were tuned on the smoke set" is not
accurate: H6 to H8 do not exist yet and are frozen after collection but before any dithered result is examined (the analysis stage). The collection
freeze has not happened yet, so the past tense ("was applied") is not yet true.

**D37 (proposed). Three evidence tiers,** which make the owner's instinct precise without contradicting the pre-registration.
- **Tier 1, confirmatory:** the pre-registered primary tests, one per hypothesis, corrected within family (the multiplicity map, D35). Only
  these can be called findings. They carry tests and intervals.
- **Tier 2, descriptive:** rankings, curves, direction of change, costs, reasoning inertia. Reported with intervals, no p-values and no "discovery"
  language; rankings shown as tiers, with conditions that cannot be told apart tied.
- **Tier 3, exploratory:** anything spotted after the fact. Labelled a hypothesis for the next experiment and never a headline.
- Claim 2's slot is filled only from Tier 1, by the D34 rule, or says "no mechanism stood out". Claim 4's ranking is Tier 2. "Let the headlines
  rise on their own" is replaced by "headlines are drawn only from Tier 1, by a rule written in advance".

### RT3. "Are you giving every penny-pinching executive permission to stop funding data hygiene? And shouldn't I recommend shutting the agents down?"

*Fired by the owner at the assistant: the Chief Data Officer of a Fortune 200 financial services firm. He has read the LinkedIn series, not the
commit log. He must defend an $18 million budget for master data management, governance and catalogs on Tuesday, and his CFO has circled "most of
the record is decoration" in red.* Note that he treats the results as published; ours are still conditional until the run.

**What is right: three vulnerabilities, all in our own wording.**
1. **"For this decision, most of the record is decoration; perfect addresses did nothing"** (Pass 2, the H1 conditional) **and "quality work only matters
   where it protects a decision"** (the wording sheet, Claim 1) read as "hygiene is waste". They generalize from one decision, one agent and synthetic
   data to "the record". This is dangerous *whatever the results are*, which is why it should be fixed before any post exists.
2. **"Vanity metric" invites the same reading.** The critique is of *counting* data quality in isolation (a dashboard showing 100% deliverable
   addresses), not of doing the work. The phrase does not say so.
3. **Claims get compressed and inflated by the people who retell them.** His paraphrase turned "contact details barely moved this one decision" into "80% of
   the record is decoration", "5% drift flips valuable accounts", "silently", and "shut down all agents". We measured none of those.

**What his paraphrase gets wrong.** There is no 80% figure (H1 spoils one field at a time; 12 individual fields and several categories). "Valuable
accounts" was not measured (opportunity cost is parked, D16). "Silently" is H5's question and is pending. Boundary customers flip on their own (about 40% of
the time for deeply boundary ones with no corruption), which is why the primary analysis is on stable customers. "No enterprise has clean data,
therefore shut the agents down" does not follow from anything we measured.

**The answer, in substance.**
1. **Our evidence is narrow by design:** one decision, one agent, synthetic data. "This decision did not use these fields" is not "these fields do not
   matter." The same address feeds KYC and AML checks, regulatory notices, billing, fraud and collections. Hygiene serves many consumers; we measured one.
2. **The CFO's own example is answered by 1a.** Duplicates *did* matter: decisions moved from the first duplicate, and 43.2% of the spend ($14.07 of $32.60)
   went on processing records that added nothing. The series supports funding a deduplicated customer registry.
3. **The series' consistent point is where the control belongs.** The agent has no memory (1a) and, to be confirmed by H5, rarely flags a problem unprompted. So the
   control has to sit upstream, which is exactly what master data management, governance and catalogs are. This work is an argument *for* upstream controls; what it
   adds is a way to prioritize and justify them.
4. **"Vanity" targets counting quality without a decision attached.** The frame is "necessary but not sufficient", not "unnecessary".
5. **"Shut everything down" is a non sequitur.** We measured sensitivity, not harm, and cannot say agents are worse than the current process (people and
   rules also run on imperfect data). Sensitivity is a reason for controls, not for abstention. The evidence supports *tiered* deployment: map each decision
   to the data it depends on; gate those fields upstream; keep a canary set to catch drift; keep a human on borderline cases; scale autonomy to the stakes.
   We do not advise a CDO on his fiduciary duty; we give him a decision structure.
6. **The board takeaway: fund data quality by the decisions it protects.** A decision-to-data dependency map turns the budget conversation from faith
   into evidence. (The Agentic Data Contract idea fits: for each decision, say which data it depends on and what quality it needs.)

**The booth answer (about ninety seconds).** "Thank you, and you've found the sentence I would most like to take back. 'Decoration' describes one
decision made by one agent; it says nothing about your record. The same address that did nothing for a priority decision is the backbone of KYC,
regulatory notices and fraud. And the CFO's example, deduplication, is in our first experiment: duplicates moved decisions from the very first one and 43
percent of the spend was wasted. So this does not say stop funding hygiene. It says the agent will not catch it for you, so the control has to be
upstream, which is what your program is. What it adds is a way to defend the budget: show the board which decisions depend on which data, and what
happens when it is wrong. As for shutting agents down: we measured how much decisions move, not whether they are worse than what you have today. The
sensible reading is tiered deployment, with upstream gates, a canary set, and a human on the borderline cases, scaled to the stakes."

**What we should change (proposals; the owner decides).**
- **D38. Retire "decoration" and "only matters where".** Restate Claim 1 as: *"Data quality earns its budget where it protects a decision someone depends on. For
  this decision, [x] fields carried it and the contact details did not; for compliance, billing or fraud they may be everything. Map your decisions to
  your data, fund quality where decisions depend on it, and keep the upstream gate for what the agent cannot catch."* Use "vanity metric" only about
  *counting* quality in isolation. Add to Claim 1's will-not-say: that hygiene is unnecessary; that any share of a record is "decoration" in general;
  whether agents should or should not be deployed.
- **D39. A "survives compression" test** for every claim: compress it to one sentence and check it cannot be used to argue the opposite. Add a short
  "what this does and does not mean for data-quality budgets" paragraph for the Salesforce audience.
- **D40. Make the decision-to-data dependency map the named action of Claim 4** and a candidate deliverable, together with the canary set.

### RT4. "'Put the check where the data enters' died as a perimeter defense fifteen years ago. And your priority list ignores prevalence and cost."

*Fired by the owner at the assistant: the same CDO, a few minutes later, wine in hand.* His enterprise has 450 SaaS applications, three cloud
warehouses, legacy ERPs and partner APIs; business units switch on agents with a credit card; data moves through bidirectional syncs, vector
stores and cached context, so there is no turnstile. If the agent will not raise a hand and post-decision review cannot be trusted, is this a
perimeter defense? Then: real prioritization is impact x prevalence divided by cost to remediate. Field A moves decisions 40% of the time but
errs once in 10,000; Field B moves them 8% but drifts on 30% of accounts a quarter and costs $4 million to fix on a 1988 mainframe. "If your
experiment ignores frequency and cost, isn't your priority list academic? How does it help me survive Tuesday?"

**What is right.**
1. **"Where the data enters" reads as a perimeter.** 1a's actual contrast was a gate *before the decision* versus a review *after* it. The slogan should
   say "before the agent reads it".
2. **"Priority guidance" and "what do you fix first?" imply a complete ranking.** We measure one term of three. D4 already says prevalence and cost
   are not ours, but the headline contradicts the footnote.
3. **Our results say nothing about how to wire a mesh.** The architecture argument below is an argument, not a result.

**What it misses.**
- **Every agent has a narrow waist: the step that assembles what it is handed for one decision.** You cannot inspect the mesh; you can inspect what each
  agent receives, per decision, against a contract. That is zero-trust for data: a check at the point of use, not a perimeter.
- **It is a layer, not a wall.** *Preventive:* validate at the point of use. *Exposure:* hand the agent only the fields the decision needs
  (an untested suggestion; an ablation would test it). *Detective:* a canary set and decision-stability monitoring, because the agent is not a
  reliable detector by itself (H5 pending; the window effect). *Corrective:* a person on borderline and high-stakes cases. We do not say post-decision review
  or monitoring is useless; we say the agent is not its own check.
- **Decision-first, not data-first.** You cannot govern 450 applications; you can govern the dozen decisions that matter. The dependency map
  inventories which data each one reads, and doubles as a shadow-AI inventory.
- **The ROI formula is right, and we supply the term nobody has.** Impact x prevalence / cost. Our impact term is the flip rate *given an error* (H1's
  rates are exposure-adjusted: drift among the customers whose value actually changed), by error size (H2) and type (H4). Profiling tools already measure
  prevalence and IT knows remediation cost; almost no one measures what an error does to a decision. His own example proves the term cannot be skipped:
  A is 0.4 flips per 10,000 decisions and B is 240 (30% x 8%), about 600 times more, so the order depends on impact. And the impact term can
  *save* money: if Field B barely moves the decisions agents make, the $4 million fix is not justified; if it moves them a lot, it is justified with evidence.
- **The cheapest fix may be to stop handing the agent the field.** Data minimization (verified by ablation, which this experiment does not do).

**How it helps on Tuesday.** A formula the board can follow; a short list of the decisions that matter; tiered deployment (gate at the point of use,
canary set, a person on borderline cases, autonomy scaled to stakes); and a plan to measure the impact term on the first two decisions.

**The booth answer (about ninety seconds).** "You're right, and I'll take the slogan back: 'where the data enters' sounds like a perimeter, and nobody can
wall off 450 applications. What our results support is narrower: check the data between the source and the agent, at the moment this agent reads it for
this decision. Every agent has that narrow waist, however tangled the mesh behind it. And it is a layer: a check at the point of use, fewer fields handed
over, a canary set to catch drift, a person on the borderline cases. On priorities you are right too: it is impact times prevalence over cost, and we measure
only the first term. But it is the term almost nobody has. Your profiling knows how often Field B is wrong, your IT knows the mainframe costs four
million, and nobody knows what it does to the decision. On your numbers A is 0.4 flips per ten thousand and B is 240. For Tuesday I would bring the formula,
the short list of decisions that matter, and a plan to measure the first term on the first two."

**Proposals (the owner decides).**
- **D41.** Restate the series' recommendation as **"check the data before the agent reads it (at the point of use), not after it decides"**, with the
  four layers; retire "where the data enters" and "upstream of the enterprise".
- **D42.** Rename "priority guidance" to **"impact ranking"**. Claim 4 and beat 5 say plainly that we measure one of three terms; beat 5 becomes
  **"How do you decide what to fix first?"** Build a "bring your own ROI worksheet" (impact measured, prevalence and cost yours), which replaces RT1's loss worksheet.
- **D43.** Add two options to Claim 4: **data minimization** (labelled untested; an ablation would test it) and **"measure your own impact term"**
  (the collection scripts are reusable at small scale).

**Will not say:** how to wire an enterprise mesh; that the figures in his example are ours; that data minimization is proven.

### Revision 3 worklist for the amendment (before the freeze)

*A draft exists: `amendment_revision_3_DRAFT.md`.* *Updated after RT7:* also queued: **D54** (stratify H1 to H3), **D50** (audit specification), **D51** (clustering-aware analysis plan), **D52** (window-effect wording), **D53** (limitations text).

Queued, not yet written: **D37** (the three evidence tiers, written into the statistical methodology), **D34** (the Claim 2 selection rule, with the
customer-half replication criterion), **D35** (the multiplicity map, drafted from the amendment and reviewed by the owner first), **D36** (the protocol
history paragraph in the freeze section), and the cost-beat definitions (D16 to D18) if not already specified. Each needs the generator script, the
consistency checker and its self-test updated together, as for earlier revisions.

### RT5. "That is a classifier with a long prompt, not an agent. Add one line telling it to flag bad data and your headline collapses. And it is the cheapest model."

*Fired by the assistant at the owner: an engineer who has built agent frameworks for four years.* Her charges: (1) one call, no tools, no retrieval, no
memory, a fixed schema: a classifier, so "agentic" and "your AI agent" overreach; (2) there is no way to flag a problem, so "your AI won't tell you"
shows only that a forced-choice classifier without a flag option does not flag; (3) one cheap model and one prompt, so why would any of it tell her about
the agents customers deploy; (4) which of the four claims survives a real agent?

**What is right.**
1. **Ours is a minimal agent: the decision step of an agent, isolated.** The word needs defining, and "your AI agent" in public posts overreaches.
2. **Claim 3 is about unprompted behavior.** We have not tested asking. (Partly mitigated: the reasoning field is free text, so the agent *can* voice doubt,
   and confidence is another channel; H5 measures both. What is missing is an explicit escalate field and an instruction to check.)
3. **One model, one prompt, one API.** "We test any model" is not supportable and contradicts our own scoping ("this agent, this model").

**What it misses.**
- **Isolation is the point.** Tools, memory and retrieval add variables; the question is what the decision step does with the data it is handed. Every
  agent has that step, and in an agent with tools the tool outputs *are* the data: an agent that looks its customer tier up in MDM inherits MDM's errors.
- **The owner's return-policy analogy.** Option 1: the agent works out the tier from what it is given. Option 2: the agent asks MDM. We test option 1, the case
  MDM exists to prevent. It also frames the whole series.
- **Self-funded and independent.** A point of integrity; it needs a formal disclosure line (the author's employer is Salesforce).

**What survives a real agent (the honest triage).**
| Claim | Transfers? |
|---|---|
| 1. Quality earns its budget by protecting a decision | **The method transfers; the ranking does not.** Re-measure for each agent. |
| 2. How the agent reacts (size, believability, consistency) | **Least likely to transfer.** Agents that cross-check sources, and stronger models, may behave differently. |
| 3. Unless you ask, it does not tell you | **Unknown, and it depends on prompting.** Hence "unless you ask". |
| 4. Impact ranking, worksheet, canary, point-of-use check | **The worksheet, the canary and the logic transfer; the impact numbers are agent-specific.** |
| Also found: the window effect | Likely to appear in any API-served model at some rate; we tested one. |

**The booth answer (about ninety seconds).** "What we built is a minimal agent: the decision step with nothing else. It is the part every agent has, and the part
we can isolate. Give it tools or memory and you add variables: the memory of how it scored this customer on a clean record would contaminate the very
comparison we are making. And if your agent asks MDM for the tier, MDM's errors become its data, so this is what happens in the case MDM exists to prevent.
You are right that we have not tested asking. Our claim is that, *unless you ask*, it probably does not tell you; whether it will if you ask, or give it a
flag field, is the next experiment, and I would expect the answer to matter. And it is one model: Haiku, chosen because I pay for this myself and want it
independent. The design works for any model and the repo lets you swap it; what transfers is the method and the worksheet, not our numbers."

**Proposals (the owner decides).**
- **D44.** Define **"minimal agent"** (one model call, no tools, no memory, no retrieval: the decision step, isolated) in the README, the wording sheet and every post; say
  "an AI agent that decides from the data it is handed", never "your AI agent" without the qualifier; add a **"scope of the evidence" box** to page one (minimal
  agent, one model, one prompt, synthetic customers, no outcome labels).
- **D45.** Restate Claim 3 as **"Unless you ask, your AI probably won't tell you the data is wrong."** Park **Experiment 1c: do cheap in-agent controls work?** (ask it to
  flag; add an explicit flag field; give it a verification tool; hand it only the fields it needs) in the backlog.
- **D46.** Remove "any model". Say **"one model (Haiku 4.5), with a design that works for any"**; add a funding and independence line to the README and paper; park
  **a replication on a second model tier** in the backlog.

**Owner's rehearsal answer, assessed.** *Landed:* the isolation logic (tools and memory add variables; the clean-baseline memory example is exactly right); the
return-policy analogy; self-funding as independence; "decision consistency, not model quality". *Did not land:* the flag-option charge was never answered, and it is
the sharpest; "any model" overreaches and contradicts "this agent, this model"; the closing question (which claims survive) was skipped; "would bias the agent"
should be "would add variables" (bias means something else).

### RT6. "How can data engineers sign an Agentic Data Contract when the party on the other side can't make up its mind?"

*Fired by the owner at the assistant: a data engineer, a free beer in hand.* His argument: you tell me to write assertion checks between my pipeline and the
agent, but your own notes show 3 of 30 clean customers flipping on a different Tuesday with nothing changed; if the consumer's boundary moves on identical
data, what do I assert against? A data contract is deterministic (an integer is an integer, a foreign key resolves). When an agent makes a catastrophic call,
the product manager will point at my gate: "you certified this payload AI-ready; why didn't your pipeline catch the flip?"

**What is right.**
1. **A data contract cannot guarantee the decision.** The engineer can sign for the payload and nothing else. "Certified AI-ready" over-promises: it sounds like a
   promise about outcomes.
2. **Our wording has been one-sided.** "Check before the agent reads it" and "assertion checks at the narrow waist" imply the gate prevents bad decisions. A gate catches
   data that breaks stated rules; it cannot catch a flip caused by the consumer's own variance.
3. **Accountability is the real question,** and we have not answered it in writing.

**What it misses.**
- **Assertions are written against the data, not against the agent.** The consumer's variance does not change what can be asserted: types, ranges, referential
  integrity, cross-field consistency (the H3 kind), freshness. What the dependency map adds is a ranking: assertions on the fields a decision leans on are blocking;
  assertions on fields it ignores are warnings. That *reduces* the engineer's burden.
- **The contract has two sides.** *Payload* (the engineer): deterministic assertions, with evidence that a payload conformed. *Decision stability* (the agent's owner): a stability
  figure measured on a canary set, versioned with the model snapshot, prompt and platform. A third owner, the business, sets the autonomy level and where a person reviews.
- **A canary gives attribution, which is what protects the engineer.** Known-clean records interleaved with live traffic: a flip on a canary is the consumer's variance; a
  flip that follows a failed assertion is the data. The clean re-run control in 1b is this pattern, used in an experiment. The post-mortem answer becomes "the payload conformed
  to the contract, and the canary shows the model moved on clean records that day."
- **Interface contracts have never guaranteed consumer behavior.** Dashboards over eventually consistent stores, retrained models and human analysts are all non-deterministic
  consumers; model monitoring exists for this reason.
- **A changed model, prompt or platform makes the sensitivity map stale.** Version the contract with the model fingerprint and re-run the sensitivity test on a change.

**What a gate can and cannot do (a proposed box).** *It can:* stop or flag a payload that breaks stated rules; be prioritized by which fields a decision depends on; produce evidence
that the payload conformed. *It cannot:* make the consumer deterministic; catch a flip caused by the consumer's own variance; guarantee that a decision is right; certify a payload
"AI-ready".

**The booth answer (about ninety seconds).** As spoken above: the contract never promised the decision; you sign for the payload, ranked by what the decision leans on; the agent's
side signs for decision stability, measured on a canary set; a flip on a clean canary is the consumer, a flip after a failed assertion is the data; certify "conforms to these rules",
never "AI-ready"; and we do not know why identical input flips, only that it was not our code or data.

**Proposals (the owner decides).**
- **D47. Define the contract as bilateral** in every piece: the payload side (deterministic assertions, ranked by the dependency map; blocking for fields the decision depends on,
  warnings otherwise), the decision side (a stability figure on a canary set, versioned with the model), and the business's autonomy level. Retire "certified AI-ready"; say "the payload
  conforms to the stated rules". Add the "what a gate can and cannot do" box.
- **D48. Attribution by canary.** Present the clean re-run control as the production pattern: a few known-clean records interleaved with live traffic. A flip on a canary is the
  consumer's variance; a failed assertion is the data. Add it to Claim 4 and the glossary.
- **D49. Version the contract with the model.** When the model snapshot, the prompt or the serving platform changes, the sensitivity map is stale: re-run the test. The run fingerprint
  exists for this.

**Will not say:** why identical input flips (we do not know; it is not our code or data); that a gate prevents catastrophic decisions; that a canary removes the variance.

### RT7. Reviewer 2: five attacks from an academic who studies LLM evaluation

*Fired by the assistant: a program-committee member at a data-centric AI workshop.* (1) Sensitivity analysis is textbook and data cascades were formalized in 2021;
where is the cascade? (2) The LLM judge is the same model as the agent it judges. (3) "Same input, different output" is a known property of inference serving; what is
new, and why "window"? (4) Faker customers with independent random errors say nothing about real, source-correlated errors. (5) Evaluation awareness survives removing
`example.com`; and 61,000 tasks sharing 48 windows are not independent, so what is the effective sample size?

**Verified references (for later writing).** Sambasivan et al., "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI, CHI 2021 (interviews
with 53 practitioners; cascades defined as compounding events causing negative downstream effects from data issues; reported as pervasive, invisible, delayed, often avoidable).
He et al., "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab, September 2025 (a research blog post: batch-size-dependent numerics under varying load; 1,000
temperature-0 completions of one prompt gave 80 unique outputs, versus identical outputs with batch-invariant kernels). Both are motivation and context, not our findings.

**1. Novelty and the cascade.** *Right:* the cascade is our motivation, not a finding; we measure the first link (how often a bad input changes one decision). Our
contribution is an empirical measurement for an increasingly deployed system class, not new theory. *Owner's answer, assessed:* the concession is good. "A classifier's sensitivity
is bounded" over-generalizes (linear models extrapolate; trees saturate). "An LLM agent fabricates coherence" is a hypothesis H5 tests, not a premise: in the pilot the
115-year tenure was *flagged* ("1380 months/115 years tenure indicates data quality issue, but account created 2021 shows ~3.5 years actual tenure"), by cross-referencing a sibling
field, and the amendment says both genuine pilot detections came that way ("hypothesis, not finding"). "Valued long-standing client" is an illustration, not an observation. *Better
answer:* a prompt plus an LLM has no probability vector or gradient, so sensitivity has to be measured empirically; its response channels differ (echo, convert, omit, doubt,
reframe: exactly what H5's instruments separate); and such agents are being deployed now with no sensitivity numbers.

**2. The self-judging judge.** *Right:* the judge defaults to claude-haiku-4-5-20251001 at temperature 0, the same model as the agent. *Owner's answer, assessed:* correct that the keyword
scan is the primary instrument and the judge is secondary. But the amendment already has a human-audit design in outline (stratified by the judge's label; the implausible-down arm included;
a slice of unremarked texts for misses; clean baseline texts as negatives for false positives; blind auditors; an "ambiguous" option; development texts excluded; agreement reported with and
without ambiguous). **Missing:** sample sizes; the number and identity of raters (one rater is weak); the agreement measure and threshold; how disagreements are resolved; a consequence rule
for Claim 3 if humans disagree with the judge; and judge independence.

**3. The window effect.** *Right:* cross-session variability at temperature 0 is a known property of inference serving. *Owner's answer, assessed:* adopting a mechanism he has not read
is the risk, and our observation (all five runs identical within a window, different across windows) does not look like continuous load noise, which predicts variation within a window too;
routing, hardware or version differences are equally candidates, and we cannot see the serving side. Keep "window effect" as the descriptive term (and the plain-language handle for
a CDO); name batch non-invariance as a *candidate mechanism* with the citation. The contribution is designing the collection around the phenomenon and measuring its size in a decision setting, not discovering it.

**4. The error model.** *Right:* we inject errors independently across customers; real errors cluster by source and time. *Owner's answer, assessed:* H3, H7 and H8 cover correlation
*within a record*, which is not the point. *Better answer:* the agent is stateless, so the per-decision impact does not depend on other records; clustering across records changes *how many*
decisions are affected (prevalence, the user's term in impact x prevalence / cost), not what one wrong record does. It would matter for agents that see many records at once, which is out of
scope for a minimal agent. Synthetic data's advantage: we know the true value of every field, so the input has ground truth.

**5. Evaluation awareness.** *Right:* absence of verbalization is not absence of awareness; a pristine record is a tell. *Owner's answer, assessed:* "the baseline shares the artifact so it
is subtracted out" holds for a constant offset, not for an *interaction*: a test-aware model might scrutinize anomalies differently. State it as a limitation. The one cue we could change was
tested (0 of 30, weak). Remedies: a second model; replication on real data by others.

**6. Effective sample size.** *The owner over-conceded.* Simulation (3,000 collections; 48 windows whose flip rates differ, mean 10% and sd 5 points): for two no-effect conditions in the same
blocks, the naive 95% interval for their *difference* covered the truth 95.2% of the time (actual sd 0.0133 against a naive standard error of 0.0134), because balance across blocks cancels the
window effect in contrasts. The exposure is **absolute rates**: a naive interval covered the long-run mean only 88.8% of the time; a cluster bootstrap by block, 92.5%. The 5-point spread is an
assumption; the clean re-run control measures the real one.

**Proposals (the owner decides).**
- **D50. Specify the audit:** sample sizes, at least two raters, an agreement measure and threshold, a disagreement rule, a consequence rule for Claim 3, and a second judge from a different family or a larger tier run on the audit sample.
- **D51. A clustering-aware analysis plan:** contrasts analyzed with block as a stratum (or reported as conservative unblocked); absolute rates and intervals by cluster bootstrap on blocks; between-block variance (ICC) estimated from the control.
- **D52. Window-effect wording:** "window effect" names the phenomenon; batch non-invariance is cited as a candidate mechanism; the owner reads the source first.
- **D53. Limitations text:** evaluation awareness may interact with corruption; error correlation across records is a prevalence question for a stateless agent; synthetic data cannot reproduce real error structure; novelty is empirical measurement.

### RT7 follow-up: the draft Revision 3, and what building it found

The Revision 3 draft is in `amendment_revision_3_DRAFT.md` (nothing applied). Reading the amendment to build the multiplicity map found: **(1)** H1, H2 and H3 are not stratified (zero stable or
boundary handling in their evaluators) although the amendment's reason for making stable customers primary applies to them; only H4 and H5 stratify; **(2)** H2, H7 and H8a pre-register no inferential
test or interval; **(3)** the H1, H3 and H6 headlines are consistency summaries with no numeric criterion; **(4)** a multiplicity correction is named only for H4's two garbage-filter families.
**D54 (DECIDED by the owner, option A): stratify every drift analysis the same way** (stable primary; boundary and all reported alongside), by extending the H1, H2 and H3 evaluators before the collection freeze.
**Correction recorded:** the RT1 rehearsal answer said the primary analysis is on stable customers; that is true for H4 and H5 only until D54 is done.

**D56 (DECIDED by the owner, with five refinements accepted): the H2 cliff test.** Stable stratum only (boundary and all plotted alongside, net of their own control); s = drift at the 5% rung minus the control, F = the largest drift over the
15/40/100% rungs minus the control, on the stable customers changed at every rung; floor F >= 10 points; cliff s >= 0.5 F and significant versus the control (Holm across the three fields); smooth s < 0.25 F; at least two of three fields
must be cliffs. The "5% error" is the *size* of each error, not the share of records affected. Simulation: the owner's two example curves classify correctly 100% of the time; exact-threshold curves are coin flips absorbed by the
mixed zone; with tenure decorative, qualification needs churn and spend both to be cliffs (87% when spend's s/F is 0.6). Details in the Revision 3 draft, R3-3.

**D72 (DECIDED by the owner, 22 October): every confirmatory call goes through the Message Batches API, one batch per block, submitted in sequence.** Reason: cost (self-funded, half price); the model, prompt and settings are unchanged. The block stays the window (the next batch is sent only after the previous ends); modes are never mixed in confirmatory data; retries are batch too. A pilot-only smoke check (`batch_smoke.py`, about 450 calls) tests whether copies of a customer inside one batch agree as they do inside a standard-mode window: the window-effect (LLM jitter) correction the block design exists for. Code and tests built the same day (R3-15). **Smoke result:** within each batch 30 of 30 customers' copies agreed (as in the standard pass); 1 to 2 of 30 customers changed between windows, always the same two. A batch behaves like a window; jitter concentrates on a few customers. Correction recorded: an earlier remark that `refine_boundary_convergence.py` used the Batch API was wrong; its "batches" are groups of 10 extra runs.

**D71 (DECIDED by the owner, 22 October): the second judge is `gemini-3.8-flash` on the paid Batch API,** thinking level `low` (Flash offers low, medium or high; minimal errors), fallback `gemini-3.6-flash` if unavailable. Supersedes the Argon proposal in D64. Reason: self-funded research; the study measures decision dependence on data, not model capability, so the judge matches the agent's tier (Haiku 4.5, batch). Cost about $20 to $35 for about 8,000 texts at prices before 1 January 2027 (they double then). The reported model version is recorded per call, because Google has silently re-pointed the 3.5 and 3.7 Flash strings. Independence confirmed by the owner: self-funded with personal credits; results are also used in his work presentations.

**D70 (from the archived transcripts and the owner's commit times):** the run timestamps are the owner's local clock (`datetime.now()`): Pacific on 26 September, Central on 2 October. 26 September: the H3 design commit (259151d, 12:03 PT) preceded the first run (19:07 PT); the runs were the Tier 2 evaluator checks, whose script states "structural validation only; no substantive value is ever printed", and whose output reads "No substantive values inspected or reported by design"; the evaluator commit (e8e3183, 20:17 PT) followed the last run (20:03 PT). 2 October: the H2 and H4-plausible runs (14:22 to 14:56 CT) preceded the H4 commit (5e9746c, 15:18 CT) by 22 minutes; I found no analysis of those outputs in the two transcripts searched, which is not proof that there was none. The owner reports that he keeps his own peeks to a minimum and usually hands output to the assistant to analyze; the assistant's reading counts as the study team's reading, so the disclosure says so.
**D69 (PROPOSED): a dated revision history and a revision-specific status line (R3-14).** From the owner's git history: evaluators first committed 5 August (core), 23 September (H1, H2), 26 September (H3) and 2 October (H4); the amendment was revised on 29 July, 5 to 8 August, 26 September, 2 October, 5 October, 10 to 11 October (Revision 2) and 15 October. The 29 July status line was true then and is not true of the document as it stands. The H3 redesign (26 September) and the H4 redesign (2 October) were committed on the days of the first runs of those hypotheses; whether the runs preceded or followed those changes is for the owner to state, and the protocol history says so either way.

**D68 (dates, from the owner's records):** the amendment's first commit is 5faa454 on 29 July 2026, before any agent run. The seed-42 runs (100 customers, 100 real decisions each) took place on 26 September (H1 individual fields x3, H3 x7, 19:07 to 20:03) and 2 October 2026 (H2 at 15% x3, H4 plausible x3, 14:22 to 14:56); the seed-777 H4 implausible runs (30 customers) on 3 October 2026 (12:15 to 12:21). The seed-42 runs use the same population and conditions as the Tier-1 plumbing fixtures (N = 100, seed 42), which suggests they were the real-agent follow-up to those checks (evaluator validation) [owner to confirm]. No committed or working document in the repo copy cites them as the source of a protocol change. The amendment's status line ("before evaluator code, before any agent runs ... beyond initial smoke tests") must become revision-specific once the revision and evaluator commit dates are known.

**D67 (RESOLVED by the audit): there are two real pilot populations.** `audit_input_populations.py`, run on the owner's machine, found 33 real results files on seed 777 (30 customers) and 16 real results files on seed 42 (100 customers, 100 model decisions each, 0 placeholder rows: h1_individual x3, h2_mag15pct x3, h3 x7, h4_plausible x3). An earlier hypothesis, that the seed-42 files were the Tier-1 plumbing fixtures, was wrong: the audit separates fixtures (placeholder rows, no model field) from real runs, and these are real. The populations share no customer (130 in all), and the planned main run shares none with either. Earlier statements of '30 customers' and 'zero of the 30 reappear' are corrected to 130.
**D66 (seed check, superseded by D67):** the owner's history shows no generation run without `--seed`, and the only seed in a manifest is 777 (the email A/B builder), but the generator writes no manifest of its own and the history also shows `--seed 42 --n 100` generation runs for H3 and H1 conditions; whether the agent ever ran on them is unknown. Two findings: the `record_id` hash is position-based and identical across seeds (it cannot identify a seed), and every results row stores `input_hash`, the md5 of the record the model saw, which can. `audit_input_populations.py` (a diagnostic, not part of the freeze) uses both. The planned main run (seed 20261016, n = 1000) shares no customer with seed 777 or 42 at n = 30, 100 or 1000; for one seed a smaller n is a prefix of a larger n. Revision 3 adds `generation_manifest.json`.

**D63 (DECIDED): the main-run generation seed is 20261016,** passed explicitly (the generator's default is 42); `check_pilot_disjoint.py` asserts the manifest seed and the disjointness from the seed-777 pilot. In the repo copy the other seeds are benign: 999 (test fixtures), 20261009 (a bootstrap seed), 500 to 524 (per-field dithering configs in the H1 replication builder).
**D64 (DECIDED: Google, not OpenAI, the owner's choice of provider; the model was chosen later, see D71):** candidates and rules in R3-6. Findings: `gemini-3.5-flash` and `gemini-3.7-flash` silently route to newer models; Flash versions turn over every one to two months; the 2.5 family is limited to prior users; `gemini-3.1-flash-lite` shuts down no earlier than 7 May 2027; Argon is stable but about a week old. The second judge is run once, in one window, on all judged texts, sealed.
**D65 (PROPOSED): crowd labeling, served not chosen,** exactly three labels per text, Krippendorff's alpha at 0.60, hidden gold items, minimum coverage; this supersedes the three-named-rater design (D62) if accepted.

**R3-5, R3-7, R3-8, R3-10, R3-11 (DECIDED by the owner: accepted as drafted, with corrections).** R3-5: the window size is a variance component (the control has about 21 tasks per block, so a raw between-block SD would be about 3.7 points even with no window effect), and a block holds about 1,271 calls (61,000 / 48), not 1,176;
the baseline runs of a customer are spread a stride of 9 blocks apart from a random start, not "run k in block k". R3-10: the guarantee that the pilot customers are excluded is an automated check (`check_pilot_disjoint.py`) plus a pinned main-run seed, and the freeze "documents", it does not "prove", that nobody looked.
**D61 (DECIDED): the second judge is cross-provider (OpenAI), same frozen prompt;** the owner's suggested `gpt-4o-2024-08-06` is NOT eligible (published shutdown 14 January 2027), so the model string is a current pinned snapshot to be recorded before the analysis freeze. **D62: raters by role** (the author and one colleague as raters, a second colleague as adjudicator; colleagues are not named in the public repo, and contributors are acknowledged only with consent);
the design (all three label everything, with the author-excluded agreement as the gate) awaits the owner's decision.

**D60 (DECIDED by the owner, with corrections): the structural choices.** Holm is applied WITHIN each of eight families, never across them (59 tests: F1 H1's 30; F2 H3's 4 interaction tests; F3 H3's 4 confidence tests; F4 H4's 6; F5 H4 keyword 5; F6 H4 judge 5;
F7 H2's 3 cliff tests; F8 H8a's 2 pair tests; the first list of six missed F7 and F8, and H3 has four real sets, not three: the amendment's "applied three times" is stale and is corrected). **D50 (DECIDED):** the audit (about 300 judge-audited texts, 200 zero-hit texts, two raters
with pre-registered properties and names recorded before labelling, human-human kappa at least 0.60 to use the labels, judge-consensus kappa at least 0.60 for Tier 1, 0.40 to 0.60 Tier 2, below dropped) and a second judge from a different family or larger tier (model string fixed before the analysis freeze; a
sensitivity analysis that does not replace the human audit). **Halves (DECIDED):** the partition string `adc-1b-halves-v1` (1,000 ids split 514 and 486). **Still to confirm: R3-5, R3-7, R3-8, R3-10 and R3-11 have not been explicitly accepted yet.**

**D58 (DECIDED by the owner; the owner's rules, reshaped by simulation): the numeric decision rules for H1, H3, H6, H7 and H8a.** Findings that drove the changes: H1's "24 of 30" is satisfiable by noise (fired 55% with 2 of 5 top fields no better), so it
becomes "4 of 5 top fields each beat 5 of 6 comparison fields (Holm) and the median gap is at least M"; H3's "2 of 3" fired 49% when one pair had no effect, so it becomes "all three significant after Holm, same direction either way, median at least M";
H6 stays Tier 2, quoted in points AND as a multiplier, with M deliberately not applied (a relative-exposure claim, not an absolute one); H6's raw ordering is vacuous (100% with or without real exposure) and its net three-way ordering has power 14% at a true 2x, so it becomes a pooled boundary tier against stable with a customer-level interval (false positive 7%, power 81%), Tier 2;
H7's rule is kept with an exact floor and a significant-fall definition of erratic; H8a needs the point estimate at least M (the lower bound alone calls a 6-point synergy super-additive 79% of the time) and Holm across two pairs.
**D59 (DECIDED by the owner):** an operating-characteristics appendix, with its simulation script as an analysis-stage artifact and a test, recomputed with the realized stratum sizes from the baseline before the analysis freeze.

**D57 (DECIDED by the owner): one shared materiality line, M = 10 points of net drift in the stable stratum.** It is the H2 floor (F >= M) and the headline line (no Tier-1 headline on an effect below M), and it defines "non-material for this
decision" (the interval's upper bound below M); results are also tabulated at 5 and 15 points as a sensitivity analysis. **Refinement of D55 after D57 (DECIDED, confirmed by the owner):** M applies to the LEADER only, not to every candidate (applied to every
candidate, a runner-up flickering across the line manufactured dominance: false solo 25% on a near tie versus 5%); the 3-point gap floor is dropped (redundant: a leader of at least 10 points that is 1.5 times the runner-up is at least 3.3
points ahead); dominance must hold in both customer halves. A cliff headline quotes s (the 5% rung), so it needs s >= M, which is stricter than the floor on F.

**D55 (proposed; the owner's idea, refined): a pre-registered dominance rule for Claim 2's headline.** The qualification gate (survive correction, replicate in both customer halves) decides who
is *in*; a ratio rule decides how *loud*. **Solo headline** if the leading qualifier's effect (absolute percentage points of drift, stable stratum) is at least 1.5x every other qualifier's, the gap is at least
3 points, and the leader and the 1.5x ratio hold in BOTH customer halves; **plural headline** (the co-leaders within 1.5x of the leader; worded "no single reaction mode dominated", not "equally") otherwise; **one qualifier:**
"the only mechanism that qualified"; **none:** "no single reaction mechanism stood out". *Simulation* (850 stable customers, 5,000 runs per scenario): equal effects gave plural 99 to 100% of the time; a clear leader
(24, 8, 6 points) gave solo 99 to 100%; at a true 1.4x the point-estimate rule declared false dominance 18% of the time (6% with the both-halves requirement); at a true 1.5x it was a coin flip (50%; 26% with the
requirement); with nothing real, 96% null; tiny effects (6 and 4 points) produced solo about half the time under every version, so a materiality floor (the smallest effect worth a headline) is a judgment for the owner.

**Decided by the owner during Pass 4:** D37 (the three evidence tiers; to be built into the amendment at the freeze step), D38 (retire
"decoration" and "only matters where", restate Claim 1; applied to the wording sheet), D39 (the compression test and the budget paragraph; applied),
D40 (the decision-to-data dependency map as the named action of Claim 4; applied). **Decided in the same session (all yes):** D33 (vocabulary: "how much the decision moves" in plain words, "decision stability" technically, "decision quality" as
the frame; applied); D41 (check the data before the agent reads it, in four layers; applied); D42 ("impact ranking"; beat 5 becomes "How do you decide what
to fix first?"; the three-term worksheet; applied); D43 (data minimization labelled untested, and "measure your own impact term"; applied); D34 (the Claim 2
selection rule, replicating in both customer halves), D35 (the multiplicity map) and D36 (the protocol history paragraph), decided and queued for the amendment's
Revision 3. Earlier passes' wording is history; the wording sheet is the source of truth. **Also decided (after RT5, all yes):** D44 (define "minimal agent"; a scope-of-the-evidence box on page one; never "your AI agent" without the
qualifier; applied), D45 (Claim 3 restated as "unless you ask, it does not tell you"; Experiment 1c parked in the backlog), D46 (no "any model": "one model,
Haiku 4.5, with a design that works for any"; a funding and independence line to confirm before publication; a second-model replication parked). **Also decided (after RT6, all yes):** D47 (the contract is bilateral: a payload side, a decision-stability side, and the business's autonomy level; never "certified AI-ready"; a "what a check can and cannot do" section; applied), D48 (attribution by canary; applied), D49 (version the contract with the model fingerprint; applied). **The practical artifact is defined** (a short, machine-checkable list of rules about the data an agent reads for one decision, ranked by dependency, with a named owner on each side); "The Agentic Data Contract" remains the title for keynotes and the series, and the documentation uses the practical definition.

*(More questions to come from the owner.)*

## Parked for later passes

- Pass 2: the story in plain language, and the claims table (if we see X we claim Y; if not, we claim nothing).
- Pass 3: DONE (decisions D27 to D31 above); the wording sheet is a spin-off.
- Pass 4: IN PROGRESS above (RT1, the loss-function objection, drafted); more questions to come from the owner.
- Pass 5: decisions: what changes before the freeze (add, cut, reword, or disclose).
- Carried from the working-document agenda: whether the conditions map to real failure modes (items 1, 6); one model,
  one prompt, synthetic customers (item 4); the "organic detection" caveat (item 5); the stated-importance
  instrument (item 3).

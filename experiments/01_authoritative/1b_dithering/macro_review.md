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
**sociological** claim (people believe it) is not, and rests on practitioner observation, optionally strengthened by
a one-question LinkedIn poll (labeled as a self-selected poll, not a survey). The identity condition in H1 (name,
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
   small or plausible errors?) Opener: the one-question poll.
2. **Does it depend on which data, and how wrong?** Size, believability and consistency of the error.
3. **Does the agent tell you?** Unprompted, does it say anything when the data is wrong?
4. **What does it cost?** *1a counted the cost of processing duplicates; 1b counts the cost of trusting bad data.* Token burn
   does not rise by itself in 1b (the same records are processed), so two direct costs are defined: **corrupted-decision
   spend** (the share of token spend that bought a decision the bad data changed, like 1a's wasted spend) and **rework
   cost** (because the agent does not flag problems, finding the bad decisions means re-running every decision that
   touched the bad data). Both use the token cost already logged per decision. The richer opportunity-cost model
   (customer value that lost attention) is parked as an exploratory add-on.
5. **What do you fix first?** The priority list, and which accounts to protect.

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
| 1 | The belief, and the poll answers | 1 |
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

## Parked for later passes

- Pass 2: the story in plain language, and the claims table (if we see X we claim Y; if not, we claim nothing).
- Pass 3: the four audiences (LinkedIn readers; data scientists on GitHub; Salesforce colleagues presenting to
  customers; the research community), three takeaways each, drafted before any data exists.
- Pass 4: the red team: the hardest questions a skeptical reader would ask.
- Pass 5: decisions: what changes before the freeze (add, cut, reword, or disclose).
- Carried from the working-document agenda: whether the conditions map to real failure modes (items 1, 6); one model,
  one prompt, synthetic customers (item 4); the "organic detection" caveat (item 5); the stated-importance
  instrument (item 3).

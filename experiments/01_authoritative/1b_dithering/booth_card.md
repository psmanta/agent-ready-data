# Booth card: Experiment 1b

One page to carry. The full reasoning is in `macro_review.md` (decision numbers in brackets) and the exact public wording in `wording_sheet.md`.
To check a draft mechanically: `python3 check_wording.py my_draft.txt`.

## Seven rules

1. **We measure how much the decision moves, not whether it is right.** No outcome labels. [RT1, D33]
2. **Scope everything.** One model (Haiku 4.5), a minimal agent with a fixed prompt that gives no decision criteria, clean synthetic data,
   one link (not a chain), structured fields (not retrieved text). [D44, D46, D76, D79, D98]
3. **Hypotheses until the run.** Say "we are testing whether", never "we found" or "we predict". The only quotable numbers are below. [§4, D97]
4. **No product claims, about anyone, in either direction.** Not the employer's, a partner's or a competitor's; never a provider's defect.
   Every chart carries its scope inside the image. Independence is the defence. [D89, D96, D99, D100, D104]
5. **The contract has two sides.** The data side signs for the payload; the agent side reports a stability figure from a canary. A check
   cannot certify a decision or make an agent deterministic. [RT6, D47, D77, D82]
6. **Layers, not a wall.** Hygiene at the source on a schedule; a small, deterministic check at the point of use; a canary on the decision;
   a person on held, high-stakes cases. [D41, D84, D85, D91, D94]
7. **We supply one term.** Impact (how much the decision moves). Prevalence, cost and dollars are theirs. [D42, D87]

**Two habits:** open without "you're completely right"; shorten an answer by leaving things out, never by contradicting a rule. [D88]

## Lines that answer most questions

- "We measure how much the decision moves when the data is wrong, not whether the decision is right."
- "This study didn't test your agents, or ours. The method lets anyone test any agent." (to "does this happen in your product?")
- "You don't need a flat mirror everywhere, just where the agent looks."
- "Canary the decision, not the action." / "LLMs think, tools act."
- "The API cost is a rounding error; the real cost is a few days of an engineer's time."
- "One link, not a chain."
- "I'd rather give you a real piece of the equation than a fake ROI slide."
- "We name the exact model because anyone replicating the study needs it. The variation is documented across language models generally."
- "Open-source code, commercial model APIs I pay for myself, synthetic data; no company code, customer data or IP."

## Quotable today, in this wording only

- **Small check:** in a 30-customer check, 3 of 30 customers' decisions changed between sessions with nothing changed (exact 95% range 2.1% to 26.5%).
  Not on customer slides without its range and the words "small check". [D97]
- **1a:** about 1 in 7 customers received conflicting decisions across their duplicate records, starting at the lowest duplication level tested (10%);
  the agent's confidence did not change. 1a predates the window-effect control. [D86]

## Never say (and say instead)

| Never | Instead | Rule |
|---|---|---|
| proved, proven | showed, tested, in our check | 1, 3 |
| wrong decisions; the decision breaks; damage | the decision moved, changed | 1 |
| naked, unguarded, no system prompt, raw or unmanaged data | a minimal agent with a fixed prompt, on clean synthetic data | 2 |
| your AI agent (unqualified); we test any model | an agent that decides from the data it is handed; one model, a design that works for any | 2 |
| we're finding, we're seeing | we're testing whether | 3 |
| no safe level; 14% of decisions | about 1 in 7 customers, from the lowest level tested | 3 |
| it doesn't happen in ours; our platform enforces or prevents; not an admission | this study didn't test them | 4 |
| a named product or company's loss (Agentforce, Copilot, Zillow); "95% of pilots fail" | (leave it out) | 4 |
| AI-ready; certified; guaranteed; deterministic decisions or guardrails | conforms to the stated rules; temperature 0 removes sampling randomness | 5 |
| where the data enters; upstream of the enterprise; zero-latency controls only | before the agent reads it, at the point of use, in layers | 6 |
| three fields; the fields it reads | the few fields the decision depends on | 7 |
| inference jitter | the window effect | 2 |
| a provider's bug, defect or flaw | variation documented across language models generally | 4 |
| only open-source tools | open-source code; commercial model APIs paid personally | 4 |
| our (pre-)approved wording sheet | my published wording sheet | 4 |
| you're completely right; brilliant; fascinating | fair point | habit |

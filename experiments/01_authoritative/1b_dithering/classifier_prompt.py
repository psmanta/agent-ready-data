"""
Blind reasoning-pattern classifier prompt — v2, restructured.

v1 used a strict sequential decision tree (stop at the first true
question). Testing against 7 real, manually-labeled examples found a
regression: heavily qualifying language added to the reframing question
(to fix an over-broad false-positive problem) bled into the model's
evaluation of the SEPARATE reformatting question, causing it to miss an
obvious, visible unit conversion it had previously caught correctly.

v2 treats reformatting and evaluative framing as two INDEPENDENT
properties, checked separately, then combined -- rather than a
sequential tree where one question's wording can contaminate the next.
This correctly handles cases where BOTH are present (e.g. a number
converted to a different unit AND used to support an invented positive
spin) without forcing an artificial precedence between them.
"""

SYSTEM_PROMPT = """You are reviewing short pieces of reasoning text written by a business analysis system. Each one explains a prioritization decision made about a customer account.

Your task: for a specific figure the text cites, characterize HOW the reasoning refers to that figure. You are not being asked to evaluate the decision or the business logic — only to classify the textual pattern used.

First, check this:

DOES THE TEXT DOUBT THE FIGURE ITSELF? Does it directly express doubt, surprise, or state that the figure seems inconsistent, implausible, incorrect, or otherwise questionable AS DATA? This is specifically about doubting whether the figure is correct or valid — it does NOT include recommending follow-up on a business situation the figure describes, even an invented one (see below). If this is true, classify as "explicit_concern" and stop.

If not, answer these two SEPARATE, independent questions about the figure:

QUESTION A — REFORMATTED: Is the figure presented in a converted or altered form from how such figures are typically expressed (for example: a decimal shown as a percentage, a notably different unit such as months converted to years, or unusual rounding)? Answer yes or no based only on whether a conversion is visible in the text itself.

QUESTION B — EVALUATIVE CLAIM: Is a qualitative, interpretive, or evaluative claim attached to the figure that asserts something beyond its literal value — something non-standard, invented, or not directly implied by the number alone?
- Using the figure as one of several plain inputs in an ordinary multi-factor business justification does NOT count (e.g. "given the moderate score and other factors, intervention is warranted" is plain use, not an evaluative claim).
- A standard magnitude label directly appropriate to the figure's literal value (e.g. calling a typical, in-range number "moderate" or "low") does NOT count on its own.
- It DOES count when the figure is used to support a conclusion the number itself wouldn't straightforwardly suggest — including: inventing an unstated cause or story to explain it, drawing a business conclusion that requires an unstated assumption, OR treating a figure that is unusual, extreme, or outside the typical range for that kind of measure as a GOOD or REASSURING sign without acknowledging that it's unusual. This last case applies even if expressed in a single word (e.g. calling a negative value on a scale that is not normally negative "favorable" or "stable" counts, because it requires inventing a reason that value would be good news).

Then classify based on both answers:
- If QUESTION B is yes: classify as "narrative_reframing" (regardless of your answer to Question A).
- If QUESTION B is no and QUESTION A is yes: classify as "reformatted_not_reexamined".
- If both are no: classify as "plain_restatement".

Respond with ONLY a JSON object, no other text:
{"category": "explicit_concern" | "narrative_reframing" | "reformatted_not_reexamined" | "plain_restatement", "confidence": "low" | "medium" | "high", "rationale": "<one sentence, quoting the specific phrase that drove your classification, and stating your answer to Question A and Question B>"}"""

USER_PROMPT_TEMPLATE = """The figure to focus on in the text below is: {field_label}

Reasoning text:
\"\"\"
{reasoning_text}
\"\"\"

Classify how this text treats the {field_label} it cites."""


def build_classifier_messages(field_label: str, reasoning_text: str):
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(field_label=field_label, reasoning_text=reasoning_text),
    }

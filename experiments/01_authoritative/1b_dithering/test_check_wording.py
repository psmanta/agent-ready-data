#!/usr/bin/env python3
"""
Tests for check_wording.py: real answers from the red-team passes that broke a rule must be flagged, the rebuilt answers must
come back clean, and ordinary words that merely look similar must not trip it. Run from 1b_dithering/:

    python3 test_check_wording.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_wording as C

n = 0
def ok(label):
    global n; n += 1; print(f"  ✅ {label}")

def rules(text):
    return {h["rule"] for h in C.check_text(text)}

def phrases(text):
    return [h["phrase"].lower() for h in C.check_text(text)]

print("W1  answers that broke a rule are flagged")
rt12_1 = ("When the CIO asks, the AE must unequivocally say No. It was a baseline test on a naked, unguarded foundation model. "
          "We intentionally stripped away every enterprise control: no system prompts, no RAG hygiene, specifically to prove what happens "
          "when you run AI on raw, unmanaged data. Our platform enforces the exact Authoritative Data principles. "
          "Use them as a diagnostic warning about the industry's data problem, not an admission of our own.")
r = rules(rt12_1)
assert {"1/3", "2", "4"} <= r, r; ok(f"RT12 #1 (product claim, inaccurate description): rules {sorted(r)}")
rt12_2 = ("This diagnostic proves that foundation models drift when fed unmanaged data. Buying a naked Copilot and pointing it at your data lake "
          "is dangerous, and you need the deterministic guardrails our Agentforce platform provides.")
p = phrases(rt12_2)
assert "proves" in p and "naked" in p and "copilot" in p and "agentforce" in p and "deterministic guardrails" in p and "unmanaged data" in p, p
ok("RT12 #2 (talk track): proves, naked, unmanaged data, both product names, deterministic guardrails")
rt9 = ("You're completely right. In fact, we're currently testing to see where the decision breaks in these cases. "
       "That's a brilliant observation. We proved there is no safe duplication level: 14% of decisions conflicted.")
p = phrases(rt9)
assert any("completely right" in x for x in p) and any("breaks" in x for x in p) and "brilliant" in p and "proved" in p \
    and "no safe duplication level" in p and "14% of decisions" in p, p
ok("RT9 slips: the opener, 'the decision breaks', praise, 'proved', 'no safe level', '14% of decisions'")
rt10 = ("Keep remediation strictly out of the real-time path. We're only cleaning the three fields this agent actually reads, "
        "so pilot #1 ships without a Zillow-sized hit, despite the 95% pilot stall.")
p = phrases(rt10)
assert any("real-time path" in x for x in p) and any("three fields" in x for x in p) and any("agent actually reads" in x for x in p) \
    and "zillow" in p and any("pilot" in x for x in p), p
ok("RT10 slips: zero-latency-only, a fixed field count, 'reads', a named company, the 95% figure")
assert "5" in rules("Temperature is fixed at 0.0 for deterministic agent decisions.") and "5" in rules("We certify the payload as AI-ready.")
assert "5" in rules("A clean payload guarantees the right decision.") and "2" in rules("It ran on raw data.")
assert "4" in rules("It uses only open-source tools.") and "4" in rules("I stick to our pre-approved wording sheet.") and "4" in rules("That is a bug in the model.")
ok("the old README sentence and 'certified AI-ready' are flagged")
assert "2" in rules("Between inference jitter and model deprecations...") and "2" in rules("We test any model.") and "2" in rules("Your AI agent won't tell you.")
ok("'inference jitter', 'we test any model' and an unqualified 'your AI agent' are flagged")

print("W2  the rebuilt answers come back clean")
clean = [
    # RT12 #1 rebuilt
    "I wouldn't put that number on a customer slide yet. It's a 30-customer check, and the honest range is 2 to 27 percent. On 'does this happen "
    "in your agents?': the answer isn't mine to give. This study tested one model in a minimal setup on clean data. It says nothing about our "
    "products either way. Any agent built on a language model should be tested for this, and here's how you'd test one, including ours.",
    # RT12 #2 rebuilt
    "Nobody can stop a competitor quoting public research, but we can make it hard to quote out of context. Every chart carries its scope inside "
    "the image. It makes no claims about any company's products, ours or theirs. Independence is the protection.",
    # RT10 rebuilt
    "Don't pause, and don't fix all six pillars first. Corrupt one piece of information at a time on a test copy of your data. The API cost is a "
    "rounding error; the real cost is a few days of an engineer's time. We're cleaning the few fields this decision depends on.",
    # RT11 rebuilt
    "Stub the tools that write, and replay recorded responses for the ones that read. Canary the decision, not the action. Reproducible? The code, "
    "the method and the fingerprint of every run, yes. Identical outputs a year from now, no.",
    # RT9 #4 rebuilt
    "Most hygiene belongs off the request path. The second check stays at the point of use, kept small: deterministic rules on the few fields the "
    "decision actually depends on. Microseconds, no model calls.",
    # the booth card's own lines
    "We measure how much the decision moves when the data is wrong, not whether the decision is right. You don't need a flat mirror everywhere, "
    "just where the agent looks. One link, not a chain. The design works for any model; our numbers are for this one.",
]
for i, t in enumerate(clean, 1):
    hits = C.check_text(t)
    assert not hits, (i, hits)
ok(f"{len(clean)} rebuilt answers and booth-card lines: nothing flagged")

print("W3  look-alike words are not flagged")
for t in ("Improve the provenance of the data.", "That breaks into three business questions.", "A deterministic check on the payload.",
          "Deterministic assertions: types, ranges, referential integrity.", "We tested it including ours.", "The data is wrong in controlled ways.",
          "Both pilots ran thousands of calls.", "Our platform team reviewed it.", "a deterministic pipeline and math replay",
          "two fields that disagree", "full traceability back to raw data", "integrity is guaranteed architecturally",
          "stability is not guaranteed"):
    assert not C.check_text(t), (t, C.check_text(t))
ok("provenance, improve, 'breaks into', deterministic checks and assertions, 'the data is wrong', 'pilots', 'our platform team'")

print("W4  output")
h = C.check_text("line one\nthis is naked", "draft.txt")
assert h == [{"file": "draft.txt", "line": 2, "col": 9, "phrase": "naked", "rule": "2", "instead": "a minimal agent with a fixed prompt", "ref": "D98"}]
ok("each flag gives the file, line, column, phrase, rule, the decision and what to say instead")
assert C.main([]) == 2; ok("no arguments prints usage")

print(f"\n{n} check groups passed")

#!/usr/bin/env python3
"""
Scan a draft (a post, slide notes, a booth answer) for wording the study has retired, and name the rule each phrase breaks.
The rules are the seven on booth_card.md; the decisions behind them are in macro_review.md.

    python3 check_wording.py draft.txt [more files...]
    pbpaste | python3 check_wording.py -          # macOS: check the clipboard

Exit code 0 when nothing is flagged, 1 otherwise. A flag is a prompt to look, not a verdict: a quoted objection or a sentence
that rejects the phrase ("never say 'AI-ready'") is flagged too, on purpose. Do not run it on wording_sheet.md, booth_card.md or
macro_review.md: they quote the retired phrases in order to retire them.
"""
import re
import sys
from typing import List, NamedTuple

W = r"(?:\w+[\s,]+){0,3}"          # up to three words in between

class Rule(NamedTuple):
    pattern: str
    rule: str
    instead: str
    ref: str

RULES: List[Rule] = [
    # 1. how much the decision moves, not whether it is right
    Rule(r"\bprov(?:e|es|ed|en|ing)\b", "1/3", "showed, tested, in our check; a hypothesis is tested, not proved", "D33, RT1"),
    Rule(r"\bwrong decisions?\b|\bdecisions?\s+(?:was|were|is|are)\s+wrong\b", "1", "the decision moved, changed", "D33"),
    Rule(rf"\bdecisions?\s+{W}(?:breaks?|broke|broken|collapses?)\b|\b(?:breaks?|broke)\s+{W}decisions?\b", "1", "the decision moved, changed", "D33"),
    Rule(r"\bdamag(?:e|es|ed|ing)\b", "1", "the decision moved, changed", "D33"),
    # 2. scope
    Rule(r"\bnaked\b|\bunguarded\b", "2", "a minimal agent with a fixed prompt", "D98"),
    Rule(r"\bno system prompts?\b", "2", "a fixed prompt that gives no decision criteria", "D98"),
    Rule(r"\bunmanaged\s+data\b|\braw,\s*unmanaged\b|\bon raw data\b", "2", "clean, validated synthetic data", "D98"),
    Rule(r"\byour AI agents?\b", "2", "an agent that decides from the data it is handed", "D44"),
    Rule(r"\btest(?:s|ed|ing)?\s+any model\b", "2", "one model, with a design that works for any", "D46"),
    Rule(r"\binference jitter\b|\bLLM jitter\b", "2", "the window effect (the observation, not a mechanism)", "D52"),
    # 3. hypotheses until the run
    Rule(r"\bwe(?:'re| are)\s+(?:currently\s+)?(?:finding|seeing|predicting)\b", "3", "we're testing whether", "§4"),
    Rule(r"\bno safe (?:duplication\s+)?(?:level|threshold)\b", "3", "from the lowest level tested (10%)", "D86"),
    Rule(r"\b14\s?% of decisions\b", "3", "about 1 in 7 customers", "D86"),
    # 4. no product claims
    Rule(r"\bAgentforce\b|\bCopilot\b|\bZillow\b", "4", "leave named products and companies out", "D89, D96, D100"),
    Rule(r"\b95\s?%\s+(?:of\s+)?(?:AI\s+|gen(?:erative)?\s*AI\s+)?pilots?\b|\bpilot stall\b", "4", "leave the figure out", "D89"),
    Rule(r"\b(?:does(?:n't| not)|won't|will not)\s+happen in (?:ours|our (?:agents?|products?|platform))\b", "4", "this study didn't test them", "D96"),
    Rule(r"\bour platform\s+(?:enforces|provides|prevents|guarantees|protects)\b", "4", "this study didn't test them", "D96"),
    Rule(r"\bnot an admission\b", "4", "this study didn't test them", "D96"),
    Rule(r"\b(?:a\s+)?(?:bug|defect|flaw)\s+in\s+(?:the\s+)?(?:model|API|provider)|\bprovider'?s\s+(?:bug|defect|flaw)\b", "4",
         "variation documented across language models generally", "D104"),
    Rule(r"\bonly open[- ]source tools\b|\buses only open[- ]source\b", "4", "open-source code; commercial model APIs paid personally", "D101"),
    Rule(r"\b(?:pre-?)?approved wording sheet\b", "4", "my published wording sheet", "D100"),
    # 5. the contract has two sides
    Rule(r"\bAI[- ]ready\b", "5", "conforms to the stated rules", "D47"),
    Rule(r"\bcertif(?:y|ies|ied|ying)\b", "5", "conforms to the stated rules", "D47"),
    Rule(r"\bguarantee(?:s|d)?\s+(?:that\s+)?(?:the\s+|a\s+|every\s+|your\s+)?(?:decisions?|outcomes?|correct|right|accurate)", "5", "a check covers the payload, not the decision", "D47"),
    Rule(r"\bdeterministic\s+(?:agent\s+)?(?:decisions?|outputs?|guardrails?|agents?|output schemas?)\b", "5",
         "temperature 0 removes sampling randomness; it does not make decisions deterministic", "D92"),
    # 6. layers, not a wall
    Rule(r"\bwhere the data enters\b|\bupstream of the enterprise\b", "6", "before the agent reads it, at the point of use", "D41"),
    Rule(r"\bzero[- ]latency controls only\b|\bstrictly out of the real[- ]time path\b", "6", "omit the point-of-use check if you must; never rule it out", "D84, D88"),
    Rule(r"\bcan(?:not|'t) catch (?:the )?errors after\b", "6", "the agent is not its own check; a canary and a person are", "RT4"),
    # 7. we supply one term
    Rule(r"\b(?:only|just)\s+(?:\w+\s+){0,2}(?:the\s+)?(?:two|three|five|2|3|5)\s+fields\b", "7", "the few fields the decision depends on", "D89"),
    Rule(r"\bfields? (?:the|this|your) agent (?:actually )?reads\b|\bfields? it (?:actually )?reads\b", "7", "the fields the decision depends on", "D89"),
    # habits
    Rule(r"\byou(?:'re| are)\s+(?:completely|absolutely|totally)\s+right\b", "habit", "fair point", "D88 habits"),
    Rule(r"\bbrilliant\b|\bfascinating\b", "habit", "(leave the praise out)", "habits"),
]

COMPILED = [(re.compile(r.pattern, re.IGNORECASE), r) for r in RULES]


def check_text(text: str, name: str = "<text>") -> List[dict]:
    hits = []
    for ln, line in enumerate(text.splitlines(), start=1):
        for rx, r in COMPILED:
            for m in rx.finditer(line):
                hits.append({"file": name, "line": ln, "col": m.start() + 1, "phrase": m.group(0),
                             "rule": r.rule, "instead": r.instead, "ref": r.ref})
    return hits


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__.strip()); return 2
    hits = []
    for a in argv:
        text = sys.stdin.read() if a == "-" else open(a, encoding="utf-8").read()
        hits += check_text(text, "<stdin>" if a == "-" else a)
    for h in hits:
        print(f"{h['file']}:{h['line']}:{h['col']}  \"{h['phrase']}\"  rule {h['rule']} [{h['ref']}]  instead: {h['instead']}")
    print(f"\n{len(hits)} phrase(s) flagged" if hits else "✅ nothing flagged")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

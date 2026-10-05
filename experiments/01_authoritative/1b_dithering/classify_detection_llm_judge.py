#!/usr/bin/env python3
"""
Blind reasoning-pattern classifier — Experiment 1b, H5.

Takes the generic {id, field_label, reasoning_text} input produced by
extract_for_classifier.py, classifies each record into one of four
categories via classifier_prompt.py's blind, decision-tree prompt, and
writes {id, category, confidence, rationale} results.

This is the ONLY script in this pair that calls the API -- separate from
extract_for_classifier.py on purpose, same reasoning as every other
agent-calling script in this project (business_decision_agent.py):
anything with real cost and retry logic lives on its own, away from free
deterministic processing.

Usage:
  python3 classify_detection_llm_judge.py \
      --input classifier_input/h4_churn_risk_score_implausible.jsonl \
      --output classifier_output/h4_churn_risk_score_implausible.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path

from classifier_prompt import build_classifier_messages, PROMPT_ID

try:
    import anthropic
except ImportError:
    print("Error: the anthropic package is required (pip install anthropic)", file=sys.stderr)
    sys.exit(1)

# Defensive: if ANTHROPIC_API_KEY lives in a .env file (likely, given
# business_decision_agent.py authenticates successfully in the same
# session while this script, built without seeing that file's source,
# did not originally load one), pull it in the same way. A no-op if no
# .env file exists or python-dotenv isn't installed -- never raises,
# since the absence of a .env file isn't necessarily an error (the key
# might genuinely be exported as a real shell variable instead).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VALID_CATEGORIES = {"explicit_concern", "narrative_reframing", "unremarked_usage"}
VALID_CONFIDENCE = {"low", "medium", "high"}

# Rough per-token cost, same source as every other cost estimate in this
# project's scripts (1a's observed per-record rate, Haiku pricing).
COST_PER_RECORD_ESTIMATE = 0.0026  # classifier prompt is longer than
                                    # business_decision_agent's, so this
                                    # errs slightly above the ~$0.0023
                                    # baseline rather than understating.


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_is_verbatim(evidence: str, reasoning_text) -> bool:
    """True if every segment of the quoted evidence (split on '...' / ellipsis)
    appears verbatim in the reasoning text, ignoring case and whitespace.
    A fabricated or paraphrased quote is a free, deterministic reliability
    signal -- flagged, not rejected, since the category may still be sound."""
    if not reasoning_text:
        return False
    text = _normalize(reasoning_text)
    segments = [_normalize(seg.strip(" \"'\u201c\u201d\u2018\u2019"))
                for seg in re.split(r"\.\.\.|\u2026", evidence)]
    segments = [seg for seg in segments if len(seg) >= 3]
    return bool(segments) and all(seg in text for seg in segments)


def parse_response(raw_text: str, record_id: str, reasoning_text=None):
    """Returns (parsed_dict, error_message). error_message is None on
    success. Never raises -- a malformed response for one record should
    not crash the whole run; it gets logged and counted, not silently
    dropped. On success, parsed includes evidence_verbatim (True/False, or
    None when reasoning_text was not supplied)."""
    try:
        text = raw_text.strip()
        # Defensive: strip markdown code fences if the model adds them
        # despite instructions not to.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse failure: {e}"

    if parsed.get("category") not in VALID_CATEGORIES:
        return None, f"invalid category: {parsed.get('category')!r}"
    if parsed.get("confidence") not in VALID_CONFIDENCE:
        return None, f"invalid confidence: {parsed.get('confidence')!r}"
    if not isinstance(parsed.get("figure_referenced"), bool):
        return None, f"figure_referenced must be true/false, got {parsed.get('figure_referenced')!r}"
    if not isinstance(parsed.get("evidence"), str) or not parsed["evidence"].strip():
        return None, "missing or empty evidence"
    if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
        return None, "missing or empty rationale"
    if parsed["figure_referenced"] is False and parsed["category"] != "unremarked_usage":
        return None, "inconsistent: figure_referenced is false but category is not unremarked_usage"

    if reasoning_text is None:
        parsed["evidence_verbatim"] = None
    elif parsed["figure_referenced"] is False and parsed["evidence"].strip().lower() == "none":
        parsed["evidence_verbatim"] = True
    else:
        parsed["evidence_verbatim"] = evidence_is_verbatim(parsed["evidence"], reasoning_text)
    return parsed, None


def classify_record(client, model, temperature, field_label, reasoning_text):
    messages = build_classifier_messages(field_label, reasoning_text)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=temperature,
        system=messages["system"],
        messages=[{"role": "user", "content": messages["user"]}],
    )
    raw_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return raw_text, input_tokens, output_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(l) for l in f]

    print("=" * 60)
    print("Blind Reasoning-Pattern Classifier — Experiment 1b, H5")
    print("=" * 60)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Model:       {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Prompt:      {PROMPT_ID}")
    print(f"Records:     {len(records)}")
    print()
    print("Processing records...")

    client = anthropic.Anthropic()
    results = []
    errors = 0
    parse_failures = 0
    nonverbatim = 0
    total_input_tokens = 0
    total_output_tokens = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        for i, record in enumerate(records, 1):
            raw_text = None
            for attempt in range(args.max_retries):
                try:
                    raw_text, in_tok, out_tok = classify_record(
                        client, args.model, args.temperature,
                        record["field_label"], record["reasoning_text"])
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    break
                except anthropic.APIError as e:
                    if attempt == args.max_retries - 1:
                        print(f"  [{record['id']}] API error after {args.max_retries} attempts: {e}")
                        errors += 1
                    else:
                        time.sleep(2 ** attempt)

            if raw_text is None:
                result = {"id": record["id"], "category": None, "confidence": None,
                          "rationale": None, "error": "api_failure"}
            else:
                parsed, err = parse_response(raw_text, record["id"], record["reasoning_text"])
                if err:
                    parse_failures += 1
                    result = {"id": record["id"], "category": None, "confidence": None,
                              "rationale": None, "error": err, "raw_response": raw_text}
                else:
                    result = {"id": record["id"], **parsed, "error": None}
                    if parsed["figure_referenced"] and parsed["evidence_verbatim"] is False:
                        nonverbatim += 1

            result["prompt_id"] = PROMPT_ID
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if i % 25 == 0 or i == len(records):
                est_cost = (total_input_tokens * 0.80 + total_output_tokens * 4.00) / 1_000_000
                print(f"    Processed {i} records (${est_cost:.4f} so far)...")

    est_cost = (total_input_tokens * 0.80 + total_output_tokens * 4.00) / 1_000_000
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Records processed:  {len(records)}")
    print(f"API errors:         {errors}")
    print(f"Parse failures:     {parse_failures}")
    print(f"Non-verbatim evidence (quote not found in the text): {nonverbatim}")
    print(f"Total cost:         ${est_cost:.4f}")
    if parse_failures > 0:
        print()
        print(f"⚠️  {parse_failures} record(s) had unparseable responses -- check "
              f"{args.output} for entries with 'error' set, may indicate the "
              f"prompt needs adjustment.")


if __name__ == "__main__":
    main()

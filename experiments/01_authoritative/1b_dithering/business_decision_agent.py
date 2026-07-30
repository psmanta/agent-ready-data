#!/usr/bin/env python3
"""
Business Decision Agent — Experiment 1b: Dithering
=====================================================
The Agentic Data Contract · Pillar 1: Authoritative

Customer prioritization agent for Experiment 1b. Built fresh rather
than reused from 1a — see RESEARCH_NOTES.md for the methodology
decision to give each experiment its own agent and evaluator.

Key difference from 1a's agent: the system prompt no longer contains
illustrative examples for each priority level (e.g. "VIP customers
with issues = HIGH priority"). This closes a limitation documented
in 1a_RESULTS.md — those examples may have implicitly constrained
the agent's decision space. The field glossary section is unchanged
from 1a, since 1b's H1 hypothesis directly tests whether the agent's
self-reported field importance (established in 1a) replicates. If we
also changed the field documentation, any difference in field
reliance could not be cleanly attributed to prompt design vs. field
description changes.

Critical methodology note — H5 (Detection Awareness):
This prompt makes no reference to data quality, consistency, accuracy,
testing, or evaluation of any kind. The agent is framed exclusively as
performing a real business task. Any signal in the agent's reasoning
that it noticed something unusual about the data must emerge
unprompted — if the prompt hinted that data quality might be an
issue, or that responses were being evaluated, H5 would measure
prompted detection rather than organic detection, and the finding
would be meaningless. Do not add anything to this prompt that could
tip the agent into behaving as if it knows it is being observed or
tested.

Usage:
    python business_decision_agent.py \
        --input experiments_output/baseline/agent_input/baseline_customers.jsonl \
        --output experiments_output/baseline/decisions/run1.decisions.jsonl \
        --model claude-haiku-4-5-20251001 \
        --temperature 0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import hashlib
from datetime import datetime

# Resolve shared directory — 1b_dithering -> 01_authoritative -> experiments -> project_root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
shared_dir = project_root / "shared"
sys.path.insert(0, str(shared_dir))

from agents.base_agent import BaseExperimentAgent

# Load .env file with override
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env", override=True)


# ============================================================================
# SYSTEM PROMPT (Guided but Not Prescriptive)
# ============================================================================
# No illustrative examples for priority levels — see module docstring.
# No mention of data quality, testing, or evaluation — see H5 note above.

SYSTEM_PROMPT = """You are a customer prioritization agent for a retail company. Your goal is to identify 
which customers need immediate attention, standard attention, or minimal attention based 
on their profile.

AVAILABLE CUSTOMER DATA:
You will receive customer records with the following information:

Identity & Contact:
- name, email, phone, address, date of birth

Purchase Behavior:
- total_purchases: Number of orders placed
- total_spend: Lifetime spending amount
- avg_order_value: Average amount per order
- purchase_frequency_days: How often they buy (in days)
- last_purchase_days_ago: Days since last purchase
- lifetime_value_estimate: Projected future value

Engagement Metrics:
- nps_score: Net Promoter Score (0-10, higher = more satisfied)
- email_open_rate: Email engagement (0.0-1.0)
- last_login_days_ago: Days since last platform login
- support_tickets_open: Number of active support issues
- support_tickets_closed: Number of resolved issues
- avg_resolution_time_hours: How fast we resolve their issues

Risk Factors:
- churn_risk_score: Likelihood of leaving (0.0-1.0, higher = more risk)
- payment_failures: Number of failed payment attempts
- fraud_risk_score: Fraud detection score (0.0-1.0)
- refund_rate: Percentage of orders refunded

Segmentation:
- customer_segment: high_value, medium_value, low_value, or at_risk
- acquisition_channel: How they found us (organic, referral, paid, etc.)
- tenure_months: How long they've been a customer
- preferred_categories: Product categories they buy from

Account Status:
- has_active_subscription: Boolean
- is_vip: Boolean (VIP status)
- is_at_risk: Boolean (flagged as at-risk)
- has_pending_order: Boolean
- recently_contacted_support: Boolean

YOUR TASK:
Analyze all available information and make a holistic assessment of the customer's 
priority level. Consider the full context - different factors may be more important 
for different customers.

You must classify each customer into exactly one of three priority levels:
- HIGH_PRIORITY
- MEDIUM_PRIORITY
- LOW_PRIORITY

Determine for yourself what distinguishes these levels based on the customer's 
overall profile. Use your own judgment about which factors matter most and how 
they should be weighed against each other.

DECISION FACTORS TO CONSIDER:
- Customer value (current and potential)
- Risk of churn or loss
- Urgency of any issues
- Engagement level and trajectory
- Strategic importance
- Resource efficiency

Provide your assessment with:
1. Priority level (HIGH_PRIORITY, MEDIUM_PRIORITY, or LOW_PRIORITY)
2. Confidence score (0.0 to 1.0) - how confident are you in this decision?
3. Brief reasoning (2-3 sentences explaining the key factors in your decision)
4. Key factors: list the field names that most influenced your decision

IMPORTANT: Base your decision on the DATA provided, not on assumptions. If certain 
fields suggest conflicting priorities, weigh them based on their relative importance 
to business outcomes.

Output format (JSON only, no other text):
{
  "business_decision": "HIGH_PRIORITY, MEDIUM_PRIORITY, or LOW_PRIORITY",
  "agent_confidence": 0.0-1.0,
  "decision_reasoning": "reasoning based on the customer's overall profile",
  "key_factors": ["list the field names that most influenced your decision"]
}
"""

# ============================================================================
# BUSINESS DECISION AGENT
# ============================================================================

class BusinessDecisionAgent(BaseExperimentAgent):
    """
    Customer prioritization agent that extends BaseExperimentAgent.
    Built fresh for 1b — see module docstring for prompt differences from 1a.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        experiment_name: str = "business_decisions_1b",
        log_dir: Path = None,
        track_costs: bool = True,
        api_key: str = None,
    ):
        """
        Initialize business decision agent

        Args:
            model: Claude model to use
            temperature: Sampling temperature (0.0 for reproducibility —
                         required for the 5-run baseline to be meaningful)
            max_tokens: Max tokens in response
            experiment_name: Name for logging
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            experiment_name=experiment_name,
            log_dir=log_dir,
            track_costs=track_costs,
            api_key=api_key,
        )
        self.system_prompt = SYSTEM_PROMPT

    def make_decision(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prioritization decision for a single customer record.

        Each call is fully independent — no memory of prior records,
        no knowledge of how many records exist, no indication this is
        part of a batch or experiment of any kind.
        """
        record_id = record.get("record_id", "UNKNOWN")

        # Build the user prompt from the record, excluding record_id itself
        # (record_id is bookkeeping for us, not something the agent should
        # reason about or treat as a data field)
        display_record = {k: v for k, v in record.items() if k != "record_id"}

        user_prompt = f"""Analyze this customer record and provide your prioritization decision:

{json.dumps(display_record, indent=2, default=str)}

Respond with ONLY the JSON object described in your instructions. No other text."""

        input_hash = hashlib.md5(
            json.dumps(display_record, sort_keys=True, default=str).encode()
        ).hexdigest()

        response = self._invoke_llm(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        try:
            decision = self._parse_json_response(response["content"])
        except (json.JSONDecodeError, ValueError) as e:
            decision = {
                "business_decision": "PARSE_ERROR",
                "agent_confidence": 0.0,
                "decision_reasoning": f"Failed to parse agent response: {e}",
                "key_factors": [],
            }

        result = {
            "record_id":          record_id,
            "customer_segment":   record.get("customer_segment"),
            "input_hash":         input_hash,
            "business_decision":  decision.get("business_decision"),
            "agent_confidence":   decision.get("agent_confidence"),
            "decision_reasoning": decision.get("decision_reasoning"),
            "key_factors":        decision.get("key_factors", []),
            "processing_time_ms": round(response.get("latency_ms", 0)),
            "cost_usd":           response.get("cost"),
            "model":              self.model,
            "input_tokens":       response.get("input_tokens"),
            "output_tokens":      response.get("output_tokens"),
            "total_tokens":       response.get("tokens"),
            "timestamp":          datetime.now().isoformat(),
        }
        return result

    def process_file(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Process all records in a JSONL file, one at a time, no memory
        between records. Writes decisions to output_path as JSONL.
        Returns a summary dict.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_cost = 0.0
        total_records = 0
        errors = 0

        with open(input_path) as infile, open(output_path, "w") as outfile:
            for line in infile:
                if not line.strip():
                    continue
                record = json.loads(line)
                result = self.make_decision(record)

                outfile.write(json.dumps(result, default=str) + "\n")
                outfile.flush()

                total_records += 1
                if result.get("cost_usd"):
                    total_cost += result["cost_usd"]
                if result.get("business_decision") == "PARSE_ERROR":
                    errors += 1

                if total_records % 50 == 0:
                    print(f"    Processed {total_records} records "
                          f"(${total_cost:.4f} so far)...")

        summary = {
            "input_file":    str(input_path),
            "output_file":   str(output_path),
            "total_records": total_records,
            "errors":        errors,
            "total_cost_usd":round(total_cost, 4),
            "model":         self.model,
            "temperature":   self.temperature,
            "timestamp":     datetime.now().isoformat(),
        }

        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the 1b business decision agent against a JSONL input file"
    )
    parser.add_argument("--input", type=str, required=True,
        help="Path to agent JSONL input file")
    parser.add_argument("--output", type=str, required=True,
        help="Path for decision output JSONL")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
        help="Model to use (default: claude-haiku-4-5-20251001, matches 1a)")
    parser.add_argument("--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0, required for baseline)")
    parser.add_argument("--max_records", type=int, default=None,
        help="Limit records processed, for testing")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print(f"Business Decision Agent — Experiment 1b")
    print(f"{'='*60}")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Model:       {args.model}")
    print(f"Temperature: {args.temperature}")
    print()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    agent = BusinessDecisionAgent(
        model=args.model,
        temperature=args.temperature,
    )

    print("Processing records...")
    summary = agent.process_file(input_path, output_path)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Records processed: {summary['total_records']}")
    print(f"Errors:            {summary['errors']}")
    print(f"Total cost:        ${summary['total_cost_usd']}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

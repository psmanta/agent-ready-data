#!/usr/bin/env python3
"""
Business Decision Agent - Customer Prioritization

Extends BaseExperimentAgent to make prioritization decisions on customer records.
Uses "guided but not prescriptive" approach - provides context but lets the LLM
decide how to weigh factors.

Usage:
    python business_decision_agent.py \
        --input experiments_output/agent/customers_dup_50pct.jsonl \
        --output experiments_output/agent_results/decisions/customers_dup_50pct.decisions.jsonl \
        --model claude-3-haiku-20240307
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import hashlib
from datetime import datetime

# Add shared directory to Python path
# Navigate from: 1a_dedup -> 01_authoritative -> experiments -> ai_ready_data_experiments
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # ai_ready_data_experiments/
shared_dir = project_root / "shared"

sys.path.insert(0, str(shared_dir))

# Now import BaseAgent
from agents.base_agent import BaseExperimentAgent


# Load .env file with override
from dotenv import load_dotenv
load_dotenv(override=True)

# ============================================================================
# SYSTEM PROMPT (Guided but Not Prescriptive)
# ============================================================================

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
Analyze ALL available information and make a holistic assessment of the customer's 
priority level. Consider the full context - different factors may be more important 
for different customers.

PRIORITY LEVELS:
- HIGH_PRIORITY: Customers who need immediate attention
  Examples: High-value customers at risk, VIP customers with issues, customers with 
  urgent problems, high-potential customers showing early churn signals

- MEDIUM_PRIORITY: Customers who need standard attention
  Examples: Stable mid-value customers, engaged customers with no urgent issues, 
  customers with moderate activity and no red flags

- LOW_PRIORITY: Customers who need minimal attention
  Examples: Low-value stable customers, inactive customers with low engagement, 
  customers with no current issues or opportunities

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
4. Key factors: list the 2-3 field names that most influenced your decision


IMPORTANT: Base your decision on the DATA provided, not on assumptions. If certain 
fields suggest conflicting priorities, weigh them based on their relative importance 
to business outcomes.

Output format (JSON only, no other text):
{
  "business_decision": "HIGH_PRIORITY",
  "agent_confidence": 0.85,
  "decision_reasoning": "Customer has high lifetime value and strong engagement, but showing early churn risk signals. Proactive outreach recommended to retain this valuable customer.",
  "key_factors": ["total_spend", "churn_risk_score"]
}
"""

# ============================================================================
# BUSINESS DECISION AGENT
# ============================================================================

class BusinessDecisionAgent(BaseExperimentAgent):
    """
    Customer prioritization agent that extends BaseExperimentAgent
    """
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        experiment_name: str = "business_decisions",
        log_dir: Path = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize business decision agent
        
        Args:
            model: Claude model to use
            temperature: Sampling temperature (0.3 for more consistent decisions)
            max_tokens: Max tokens in response
            experiment_name: Name for logging
            log_dir: Directory for decision logs
            api_key: Anthropic API key (loaded from .env if not provided)
        """
        import os
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            experiment_name=experiment_name,
            log_dir=log_dir,
            track_costs=True,
            api_key=api_key,
            **kwargs
        )
        
        self.system_prompt = SYSTEM_PROMPT
    
    def make_decision(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prioritization decision for a single customer record
        
        Args:
            record: Customer record dict (with record_id and all customer fields)
        
        Returns:
            Decision dict with business_decision, agent_confidence, reasoning, etc.
        """
        record_id = record.get("record_id", "UNKNOWN")
        
        # Prepare customer data for LLM (exclude record_id from analysis)
        customer_data = {k: v for k, v in record.items() if k != "record_id"}
        
        # Calculate input hash (for detecting identical inputs)
        customer_data_str = json.dumps(customer_data, sort_keys=True, default=str)
        input_hash = hashlib.md5(customer_data_str.encode()).hexdigest()
        
        # Format as readable text for LLM
        customer_text = json.dumps(customer_data, indent=2, default=str)
        
        user_prompt = f"""Analyze this customer record and provide your prioritization decision:

{customer_text}

Provide your decision in JSON format only (no other text):
{{
  "business_decision": "HIGH_PRIORITY|MEDIUM_PRIORITY|LOW_PRIORITY",
  "agent_confidence": 0.85,
  "decision_reasoning": "Brief explanation"
}}"""
        
        try:
            # Use BaseExperimentAgent's _invoke_llm method
            response = self._invoke_llm(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                retry_count=3
            )
            
            # Parse response using BaseExperimentAgent's parser
            decision_text = response['content']
            decision = self._parse_json_response(decision_text)
            
            # Validate decision format
            self._validate_decision(decision)
            
            # Build result
            result = {
                "record_id": record_id,
                "customer_segment": record.get("customer_segment"),
                "input_hash": input_hash,
                "business_decision": decision["business_decision"],
                "agent_confidence": float(decision["agent_confidence"]),
                "decision_reasoning": decision["decision_reasoning"],
                "key_factors": decision.get("key_factors", []),
                "processing_time_ms": int(response['latency_ms']),
                "cost_usd": round(response['cost'], 6),
                "model": self.model,
                "input_tokens": response.get('input_tokens', 0),
                "output_tokens": response.get('output_tokens', 0),
                "total_tokens": response['tokens'],
                "timestamp": datetime.utcnow().isoformat()
            }

            return result
            
        except Exception as e:
            # Error handling
            return {
                "record_id": record_id,
                "customer_segment": record.get("customer_segment"),
                "input_hash": input_hash,
                "business_decision": "ERROR",
                "agent_confidence": 0.0,
                "decision_reasoning": f"Error during processing: {str(e)}",
                "key_factors": [],
                "processing_time_ms": 0,
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model": self.model,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _validate_decision(self, decision: Dict[str, Any]):
        """Validate decision format"""
        if "business_decision" not in decision:
            raise ValueError("Missing 'business_decision' in response")
        if "agent_confidence" not in decision:
            raise ValueError("Missing 'agent_confidence' in response")
        if "decision_reasoning" not in decision:
            raise ValueError("Missing 'decision_reasoning' in response")
        
        # Validate decision value
        valid_decisions = ["HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"]
        if decision["business_decision"] not in valid_decisions:
            raise ValueError(f"Invalid decision: {decision['business_decision']}")
        
        # Validate confidence range
        confidence = float(decision["agent_confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence out of range: {confidence}")

        # key_factors is optional but should be a list if present
        if "key_factors" in decision and not isinstance(decision["key_factors"], list):
                decision["key_factors"] = []  # coerce silently rather than failing


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_dataset(
    input_file: Path,
    output_file: Path,
    model: str = "claude-3-haiku-20240307",
    max_records: int = None,
    api_key: str = None,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Process entire dataset and generate decisions
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model: Claude model to use
        max_records: Optional limit on number of records to process
        api_key: Anthropic API key
    
    Returns:
        Summary statistics
    """
    print(f"\n{'='*70}")
    print(f"🤖 PROCESSING DATASET")
    print(f"{'='*70}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model}")
    if max_records:
        print(f"Max records: {max_records}")
    print()
    
    # Load records
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    if max_records:
        records = records[:max_records]
    
    print(f"📊 Loaded {len(records)} records")
    print()
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    agent = BusinessDecisionAgent(
        model=model,
        temperature=temperature,
        experiment_name="business_decisions",
        log_dir=output_file.parent,
        api_key=api_key
    )
    
    # Process records
    decisions = []
    errors = 0
    decision_counts = {"HIGH_PRIORITY": 0, "MEDIUM_PRIORITY": 0, "LOW_PRIORITY": 0, "ERROR": 0}
    
    print("Processing records...")
    for i, record in enumerate(records, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(records)} ({i/len(records)*100:.1f}%) - Cost so far: ${agent.total_cost:.4f}")
        
        decision = agent.make_decision(record)
        decisions.append(decision)
        
        if decision["business_decision"] == "ERROR":
            errors += 1
        
        decision_counts[decision["business_decision"]] += 1
        
        # Write decision immediately (streaming output)
        with open(output_file, 'a') as f:
            f.write(json.dumps(decision, default=str) + '\n')
    
    print(f"  ✅ Complete: {len(records)}/{len(records)} (100%)")
    print()
    
    # Summary statistics
    avg_confidence = sum(d["agent_confidence"] for d in decisions if d["business_decision"] != "ERROR") / max(len(decisions) - errors, 1)
    avg_processing_time = sum(d["processing_time_ms"] for d in decisions) / len(decisions)
    
    summary = {
        "total_records": len(records),
        "successful_decisions": len(decisions) - errors,
        "errors": errors,
        "decision_distribution": decision_counts,
        "avg_confidence": round(avg_confidence, 4),
        "avg_processing_time_ms": round(avg_processing_time, 2),
        "total_cost_usd": round(agent.total_cost, 4),
        "model": model,
        "temperature": temperature,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Print summary
    print(f"{'='*70}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total records: {summary['total_records']}")
    print(f"Successful: {summary['successful_decisions']}")
    print(f"Errors: {summary['errors']}")
    print()
    print(f"Decision Distribution:")
    for decision, count in decision_counts.items():
        pct = (count / len(records)) * 100
        print(f"  {decision:20s}: {count:4d} ({pct:5.1f}%)")
    print()
    print(f"Avg Confidence: {summary['avg_confidence']:.4f}")
    print(f"Avg Processing Time: {summary['avg_processing_time_ms']:.2f} ms")
    print(f"Temperature: {summary['temperature']}")
    print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
    print()
    
    # Save summary
    summary_file = output_file.parent / f"{output_file.stem}.summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"📊 Summary saved: {summary_file}")
    print()
    
    return summary


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Business Decision Agent - Customer Prioritization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single dataset
  python business_decision_agent.py \\
      --input experiments_output/agent/customers_dup_50pct.jsonl \\
      --output experiments_output/agent_results/decisions/customers_dup_50pct.decisions.jsonl

  # Use different model
  python business_decision_agent.py \\
      --input experiments_output/agent/customers_dup_50pct.jsonl \\
      --output experiments_output/agent_results/decisions/customers_dup_50pct.decisions.jsonl \\
      --model claude-3-sonnet-20240229

  # Test with limited records
  python business_decision_agent.py \\
      --input experiments_output/agent/customers_dup_50pct.jsonl \\
      --output test_decisions.jsonl \\
      --max_records 10
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with customer records"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file for decisions"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        choices=[
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022"
        ],
        help="Anthropic Claude model to use (default: claude-3-haiku-20240307)"
    )
    
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to process (for testing)"
    )


    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic, default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return 1
    
    # Validate Anthropic API key
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Make sure .env file exists with: ANTHROPIC_API_KEY=sk-ant-...")
        return 1
    
    output_file = Path(args.output)
    
    # Process dataset
    try:
        summary = process_dataset(
            input_file=input_file,
            output_file=output_file,
            model=args.model,
            max_records=args.max_records,
            api_key=api_key,
            temperature=args.temperature
        )
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


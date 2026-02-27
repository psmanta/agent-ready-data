"""
Test script for base agent functionality

Tests:
1. Base agent initialization
2. LLM factory
3. Prompt library
4. Simple decision making
5. Cost tracking
6. Decision logging
"""

import sys
import os
import json
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv(override=True)  # Loads environment variables from .env
api_key = os.getenv("ANTHROPIC_API_KEY")


# Add shared to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.agents.base_agent import BaseExperimentAgent
from shared.agents.llm_factory import LLMFactory
from shared.agents.prompt_library import PromptLibrary

class TestCustomerValueAgent(BaseExperimentAgent):
    """Simple test implementation of customer value agent"""

    def make_decision(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision using the agent"""
        # Get prompt
        prompt_template = PromptLibrary.get_prompt("customer_value", "v1")
        
        # Format prompt with the customer profile data
        user_prompt = prompt_template.format(
            customer_id=customer_profile.get("customer_id", "UNKNOWN"),
            quality_score=customer_profile.get("quality_score", 0),
            customer_value=customer_profile.get("customer_value", 0)
        )
        
        # System prompt
        system_prompt = "You are an AI assistant helping to make data quality decisions."
        
        # Invoke LLM
        response = self._invoke_llm(system_prompt, user_prompt)
        
        # Track decision
        self.decisions.append({
            "customer_id": customer_profile.get("customer_id"),
            "decision": response['content'],
            "tokens": response['tokens'],
            "cost": response['cost'],
            "latency_ms": response['latency_ms'],
            "timestamp": "2024-01-01"
        })
        
        return {
            "decision": response['content'],
            "confidence": 85.0,
            "reasoning": "Based on customer profile analysis"
        }


def test_llm_factory():
    """Test LLM factory"""
    print("\n" + "="*50)
    print("🧪 TEST 1: LLM Factory")
    print("="*50)
    print(f"🔑 API Key loaded: {api_key[:20]}..." if api_key else "❌ No API key!")
    
    try:
        # Test Claude
        llm = LLMFactory.create("claude-3-haiku-20240307", temperature=0, api_key=api_key)
        print("✅ Claude LLM created successfully")
        
        # Show supported models
        supported = LLMFactory.get_supported_models()
        print(f"✅ Supported models: {len(supported['anthropic'])} Anthropic (OpenAI disabled)")

        return True
    except Exception as e:
        print(f"❌ LLM Factory test failed: {e}")
        return False

def test_prompt_library():
    """Test prompt library can retrieve prompts"""
    print("\n" + "="*50)
    print("🧪 TEST 2: Prompt Library")
    print("="*50)
    
    try:
        # Test getting a prompt
        prompt = PromptLibrary.get_prompt("customer_value", "v1")
        print("✅ Retrieved customer_value prompt")
        print(f"✅ Prompt has {len(prompt.input_variables)} input variables")
        
        return True
    except Exception as e:
        print(f"❌ Prompt Library test failed: {e}")
        return False


def test_agent_initialization():
    """Test agent initialization"""
    print("\n" + "="*50)
    print("🧪 TEST 3: Agent Initialization")
    print("="*50)
    
    try:
        agent = TestCustomerValueAgent(
            model="claude-3-haiku-20240307",
            temperature=0,
            experiment_name="test_experiment",
            track_costs=True,
            api_key=api_key
        )

        print(f"✅ Agent initialized with model: {agent.model}")
        print(f"✅ Temperature: {agent.temperature}")
        print(f"✅ Cost tracking: {agent.track_costs}")
        
        return agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None


def test_decision_making(agent):
    """Test making a decision"""
    print("\n" + "="*50)
    print("🧪 TEST 4: Decision Making")
    print("="*50)
    
    # Create test customer profile
    customer_profile = {
        'customer_id': 'TEST_001',
        'total_purchases': 15,
        'total_spend': 7500,
        'support_tickets': 2,
        'tenure_months': 18,
        'last_purchase_days_ago': 30
    }
    
    print(f"\n📊 Test Customer Profile:")
    for key, value in customer_profile.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n🤖 Making decision...")
        decision = agent.make_decision(customer_profile)
        
        print(f"\n✅ Decision made successfully!")
        print(f"   Assessed Value: {decision.get('assessed_value')}")
        print(f"   Recommended Action: {decision.get('recommended_action')}")
        print(f"   Confidence: {decision.get('confidence')}")
        print(f"   Data Completeness: {decision.get('data_completeness_perception')}")
        print(f"\n   Reasoning: {decision.get('reasoning')[:150]}...")
        
        return True
    except Exception as e:
        print(f"❌ Decision making failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_tracking(agent):
    """Test cost tracking"""
    print("\n" + "="*50)
    print("🧪 TEST 5: Cost Tracking")
    print("="*50)
    
    try:
        summary = agent.get_cost_summary()
        print(f"✅ Total cost: ${summary['total_cost']:.4f}")
        print(f"✅ Total tokens: {summary['total_tokens']}")
        print(f"✅ Total decisions: {summary['total_decisions']}")
        
        agent.print_cost_summary()
        
        return True
    except Exception as e:
        print(f"❌ Cost tracking test failed: {e}")
        return False


def test_decision_logging(agent):
    """Test decision logging"""
    print("\n" + "="*50)
    print("🧪 TEST 6: Decision Logging")
    print("="*50)
    
    try:
        # Save decisions
        agent.save_decisions("test_decisions.json")
        
        # Check file exists
        log_file = agent.log_dir / "test_decisions.json"
        if log_file.exists():
            print(f"✅ Decision log saved to: {log_file}")
            print(f"✅ File size: {log_file.stat().st_size} bytes")
            
            # Read and verify
            import json
            with open(log_file, 'r') as f:
                decisions = json.load(f)
            print(f"✅ Logged {len(decisions)} decision(s)")
            
            return True
        else:
            print(f"❌ Log file not found: {log_file}")
            return False
            
    except Exception as e:
        print(f"❌ Decision logging test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 TESTING BASE AGENT INFRASTRUCTURE")
    print("="*60)
    
    results = []
    
    # Test 1: LLM Factory
    results.append(("LLM Factory", test_llm_factory()))
    
    # Test 2: Prompt Library
    results.append(("Prompt Library", test_prompt_library()))
    
    # Test 3: Agent Initialization
    agent = test_agent_initialization()
    results.append(("Agent Initialization", agent is not None))
    
    if agent:
        # Test 4: Decision Making
        results.append(("Decision Making", test_decision_making(agent)))
        
        # Test 5: Cost Tracking
        results.append(("Cost Tracking", test_cost_tracking(agent)))
        
        # Test 6: Decision Logging
        results.append(("Decision Logging", test_decision_logging(agent)))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n{total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Ready to proceed.")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")


if __name__ == "__main__":
    main()


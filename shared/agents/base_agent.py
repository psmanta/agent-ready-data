"""
Base Agent for AI-Ready Data Experiments

Provides core functionality for all experimental agents:
- LLM interaction abstraction
- Decision logging with full trace
- Cost tracking
- Confidence scoring
- Error handling and retries
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from abc import ABC, abstractmethod

# from langchain_openai import ChatOpenAI : Uncomment if you want to use OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import LLMFactory


class BaseExperimentAgent(ABC):
    """
    Abstract base class for all experimental agents
    
    Handles:
    - LLM provider abstraction
    - Structured decision logging
    - Cost tracking
    - Retry logic
    - Confidence extraction
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0,
        max_tokens: int = 1000,
        experiment_name: str = "unknown",
        log_dir: Optional[Path] = None,
        track_costs: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize base agent
        
        Args:
            model: LLM model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3-opus")
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            experiment_name: Name of experiment (for logging)
            log_dir: Directory for decision logs
            track_costs: Whether to track API costs
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.experiment_name = experiment_name
        self.track_costs = track_costs
        self.api_key = api_key
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Setup logging
        self.log_dir = log_dir or Path("results/raw")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.decision_count = 0
        
        # Decision log
        self.decisions: List[Dict] = []

    def _initialize_llm(self):
        """Initialize LLM based on model name"""

        print(f"🔍 _initialize_llm() passing api_key: {self.api_key[:20] if self.api_key else 'None'}...")

        
        return LLMFactory.create(
            model=self.model,
            temperature=self.temperature,
            api_key = self.api_key,
            max_tokens=self.max_tokens
            )


    @abstractmethod
    def make_decision(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Make a decision (implemented by subclasses)
        
        Must return a dictionary with at least:
        {
            'decision': str,
            'confidence': float (0-100),
            'reasoning': str
        }
        """
        pass
    
    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Invoke LLM with retry logic
        
        Args:
            system_prompt: System message
            user_prompt: User message
            retry_count: Number of retries on failure
            
        Returns:
            Dict with 'content', 'tokens', 'cost', 'latency_ms'
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        for attempt in range(retry_count):
            try:
                start_time = time.time()
                response = self.llm.invoke(messages)
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract token usage and cost
                tokens = self._extract_token_usage(response)
                cost = self._calculate_cost(tokens)
                
                # Update tracking
                if self.track_costs:
                    self.total_tokens += tokens
                    self.total_cost += cost
                
                return {
                    'content': response.content,
                    'tokens': tokens,
                    'cost': cost,
                    'latency_ms': latency_ms
                }
                
            except Exception as e:
                if attempt == retry_count - 1:
                    raise Exception(f"LLM invocation failed after {retry_count} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _extract_token_usage(self, response) -> int:
        """Extract token usage from LLM response"""
        try:
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                return usage.get('total_tokens', 0)
        except:
            pass
        return 0
    
    def _calculate_cost(self, tokens: int) -> float:
        """
        Calculate cost based on model and token usage
        
        Prices as of Jan 2025 (update as needed)
        """
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        }
        
        # Simplified: assume 50/50 input/output split
        model_key = self.model.lower()
        for key in pricing:
            if key in model_key:
                avg_price = (pricing[key]['input'] + pricing[key]['output']) / 2
                return (tokens / 1000) * avg_price
        
        return 0.0  # Unknown model
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response
        
        Handles common issues:
        - Markdown code blocks
        - Extra whitespace
        - Malformed JSON
        """
        # Remove markdown code blocks
        content = content.strip()
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        
        # Try to parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content}")
    
    def _log_decision(
        self,
        decision: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """
        Log decision with full trace
        
        Args:
            decision: The decision dict returned by make_decision()
            metadata: Additional context (prompts, ground truth, etc.)
        """
        log_entry = {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'experiment': self.experiment_name,
            'model': self.model,
            'decision_count': self.decision_count,
            **decision,
            **metadata
        }
        
        self.decisions.append(log_entry)
        self.decision_count += 1
    
    def save_decisions(self, filename: Optional[str] = None):
        """Save all decisions to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_decisions_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.decisions, f, indent=2)
        
        print(f"✅ Saved {len(self.decisions)} decisions to {filepath}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs and usage"""
        return {
            'total_cost': round(self.total_cost, 2),
            'total_tokens': self.total_tokens,
            'total_decisions': self.decision_count,
            'avg_cost_per_decision': round(self.total_cost / max(self.decision_count, 1), 4),
            'avg_tokens_per_decision': round(self.total_tokens / max(self.decision_count, 1), 0)
        }
    
    def print_cost_summary(self):
        """Print cost summary to console"""
        summary = self.get_cost_summary()
        print("\n" + "="*50)
        print("💰 COST SUMMARY")
        print("="*50)
        print(f"Total Cost:              ${summary['total_cost']:.2f}")
        print(f"Total Tokens:            {summary['total_tokens']:,}")
        print(f"Total Decisions:         {summary['total_decisions']:,}")
        print(f"Avg Cost/Decision:       ${summary['avg_cost_per_decision']:.4f}")
        print(f"Avg Tokens/Decision:     {summary['avg_tokens_per_decision']:.0f}")
        print("="*50 + "\n")


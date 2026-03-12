"""
LLM Factory for creating different LLM providers

Supports:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Future: Local models (Ollama, LM Studio)
"""

from typing import Optional
# from langchain_openai import ChatOpenAI : Uncomment if you want to use OpenAI
from langchain_anthropic import ChatAnthropic


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create(
        model: str,
        temperature: float = 0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None
    ):
        """
        Create LLM instance based on model name
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            api_key: Optional API key (uses env var if not provided)
        Returns:
            LangChain LLM instance
        """
        model_lower = model.lower()
        #print(f"🔍 LLMFactory.create() received api_key length: {len(api_key) if api_key else 0}")
        #print(f"🔍 First 30 chars: {api_key[:30] if api_key else 'None'}...")
        #print(f"🔍 Last 10 chars: ...{api_key[-10:] if api_key else 'None'}")

        
        # OpenAI models - temporarily disabled
        # if any(x in model_lower for x in ['gpt', 'openai']):
        #     return ChatOpenAI(
        #         model=model,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         api_key=api_key
        #     )
        
        # Anthropic models
        if any(x in model_lower for x in ['claude', 'anthropic']):
            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=api_key
            )
            
            return llm
        else:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported: Claude (Anthropic)"
            )


    @staticmethod
    def get_supported_models() -> dict:
        """Get list of supported models by provider"""
        return {
            'openai': [
                'gpt-4',
                'gpt-4-turbo',
                'gpt-4-turbo-preview',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k'
            ],
            'anthropic': [
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ]
        }


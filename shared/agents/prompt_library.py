"""
Prompt Library for versioned, reusable prompts

Enables:
- Prompt versioning for reproducibility
- A/B testing of prompts
- Centralized prompt management
"""

"""Prompt Library for versioned, reusable prompts"""

from langchain_core.prompts import PromptTemplate


class PromptLibrary:
    """Library of versioned prompts for experiments"""
    
    @staticmethod
    def get_prompt(prompt_name: str, version: str = "v1") -> PromptTemplate:
        """
        Retrieve a prompt template by name and version
        
        Args:
            prompt_name: Name of the prompt (e.g., "customer_value")
            version: Version identifier (e.g., "v1", "v2")
            
        Returns:
            PromptTemplate instance
        """
        if prompt_name == "customer_value":
            return PromptLibrary._get_customer_value_prompt(version)
        else:
            raise ValueError(f"Unknown prompt: {prompt_name}")
    
    @staticmethod
    def _get_customer_value_prompt(version: str = "v1") -> PromptTemplate:
        """
        Get customer value decision prompt
        
        Args:
            version: Prompt version
            
        Returns:
            PromptTemplate for customer value decisions
        """
        if version == "v1":
            template = """You are an AI assistant helping to make data quality decisions.

Given the following information about a customer record:
- Customer ID: {customer_id}
- Data Quality Score: {quality_score} (0-100, where 100 is perfect quality)
- Customer Lifetime Value: ${customer_value}

Should this customer record be included in the analysis?

Respond with ONLY "INCLUDE" or "EXCLUDE" followed by a brief reason.

Example: "INCLUDE - High value customer with acceptable data quality"
Example: "EXCLUDE - Data quality too low for reliable analysis"

Your decision:"""
            
            return PromptTemplate(
                input_variables=["customer_id", "quality_score", "customer_value"],
                template=template
            )
        else:
            raise ValueError(f"Unknown version: {version}")


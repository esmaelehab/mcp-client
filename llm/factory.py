from enum import Enum
from typing import Optional

from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .anthropic_client import AnthropicClient

class LLMProvider(Enum):
    """Enum for supported LLM providers."""
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

class LLMClientFactory:
    """Factory class for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: LLMProvider) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: The LLM provider to use.
            
        Returns:
            An instance of BaseLLMClient.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        if provider == LLMProvider.GEMINI:
            return GeminiClient()
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 
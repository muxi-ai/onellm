"""
Provider implementations for muxi-llm.

This module imports all available provider implementations,
ensuring they are registered with the provider registry.
"""

from ..providers.base import get_provider, parse_model_name, register_provider, list_providers
from ..providers.openai import OpenAIProvider
from ..providers.fallback import FallbackProviderProxy

# Register provider implementations
register_provider("openai", OpenAIProvider)

# Convenience export
__all__ = [
    "get_provider",
    "parse_model_name",
    "register_provider",
    "list_providers",
    "FallbackProviderProxy"
]

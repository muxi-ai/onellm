"""
Provider implementation package for muxi-llm.

This package contains the adapters for various LLM providers,
implementing the base provider interface for each supported service.
"""

from .base import Provider, get_provider, parse_model_name
from .openai import OpenAIProvider

__all__ = [
    "Provider",
    "get_provider",
    "parse_model_name",
    "OpenAIProvider",
]

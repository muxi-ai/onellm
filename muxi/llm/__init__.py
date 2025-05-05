"""
muxi-llm: A lightweight, provider-agnostic Python library that offers a unified interface
for interacting with large language models (LLMs) from various providers.
"""

__version__ = "0.1.0"

# Public API exports
from .chat_completion import ChatCompletion
from .completion import Completion
from .embedding import Embedding
from .files import File
from .errors import (
    MuxiLLMError, APIError, AuthenticationError, RateLimitError, InvalidRequestError
)
from .config import (
    set_api_key, get_api_key
)

# Module exports
__all__ = [
    "ChatCompletion",
    "Completion",
    "Embedding",
    "File",
    "MuxiLLMError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "set_api_key",
    "get_api_key",
]

# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from muxi.llm import openai_api_key, anthropic_api_key

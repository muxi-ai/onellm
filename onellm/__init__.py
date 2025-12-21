#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OneLLM: A lightweight, provider-agnostic Python library that offers a unified interface
for interacting with large language models (LLMs) from various providers.

This module serves as the main entry point for the OneLLM library, exposing all
public APIs and functionality to users. It provides a consistent interface for working
with different LLM providers while maintaining compatibility with the OpenAI API format.
"""

import os

# Media handling
from .audio import AudioTranscription, AudioTranslation

# Public API imports - core functionality
from .chat_completion import ChatCompletion

# Client interface (OpenAI compatibility)
from .client import Client, OpenAI
from .completion import Completion

# Configuration and providers
from .config import get_api_key, get_provider_config, set_api_key
from .embedding import Embedding

# Error handling
from .errors import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OneLLMError,
    RateLimitError,
)
from .files import File
from .image import Image
from .providers import get_provider, list_providers, register_provider
from .providers.base import parse_model_name
from .speech import Speech


def get_version() -> str:
    """
    Read and return the package version from the .version file.

    Returns:
        str: The current version of the package

    Note:
        The .version file should be located in the same directory as this file.
        The version string is stripped of any whitespace to ensure clean formatting.
    """
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    with open(version_file, encoding="utf-8") as f:
        return f.read().strip()

# Initialize package version from .version file
__version__ = get_version()

# Package metadata
__author__ = "Ran Aroussi"
__license__ = "Apache-2.0"
__url__ = "https://github.com/muxi-ai/onellm"

# Module exports - defines the public API of the package
# This controls what gets imported when using "from onellm import *"
__all__ = [
    # Core functionality
    "ChatCompletion",  # Chat-based completions (conversations)
    "Completion",      # Text completions
    "Embedding",       # Vector embeddings for text

    # Media handling
    "File",            # File operations for models
    "AudioTranscription",  # Convert audio to text
    "AudioTranslation",    # Translate audio to text
    "Speech",          # Text-to-speech synthesis
    "Image",           # Image generation and manipulation

    # Client interface (OpenAI compatibility)
    "Client",          # Generic client for any provider
    "OpenAI",          # OpenAI-compatible client

    # Configuration and providers
    "set_api_key",     # Set API key for a provider
    "get_api_key",     # Get API key for a provider
    "get_provider",    # Get provider instance by name
    "list_providers",  # List available providers
    "register_provider",  # Register a new provider
    "parse_model_name",   # Parse provider from model name
    "get_provider_config",  # Get configuration for a provider

    # Cache management
    "init_cache",      # Initialize semantic cache
    "disable_cache",   # Disable caching
    "clear_cache",     # Clear cache entries
    "cache_stats",     # Get cache statistics

    # Error handling
    "OneLLMError",       # Base error class
    "APIError",           # API-related errors
    "AuthenticationError",  # Authentication failures
    "RateLimitError",     # Rate limit exceeded
    "InvalidRequestError",  # Invalid request parameters
]

# Cache management - global cache instance
_cache = None


def init_cache(
    max_entries: int = 1000,
    p: float = 0.95,
    hash_only: bool = False,
    stream_chunk_strategy: str = "words",
    stream_chunk_length: int = 8,
    ttl: int = 86400,
):
    """
    Initialize the global semantic cache.

    The cache reduces API costs and improves response times by intelligently caching
    LLM responses using a hybrid approach: instant hash-based exact matching combined
    with semantic similarity search for near-duplicate queries.

    For streaming requests, cached responses are returned as simulated streams, chunked
    naturally to maintain the streaming UX while saving API costs.

    Args:
        max_entries: Maximum number of cache entries before LRU eviction (default: 1000)
        p: Similarity threshold for semantic matching (default: 0.95)
        hash_only: Disable semantic matching, use only hash-based exact matches (default: False)
        stream_chunk_strategy: How to chunk cached streaming responses (default: "words")
            - "words": Split by words (most natural for general text)
            - "sentences": Split by sentences (periods, newlines, etc.)
            - "paragraphs": Split by paragraphs (double newlines)
            - "characters": Split by character count (precise control)
        stream_chunk_length: Number of strategy units per chunk (default: 8)
            - For "words": 8 words per chunk (~40 chars)
            - For "sentences": 8 sentences per chunk
            - For "paragraphs": 8 paragraphs per chunk
            - For "characters": 8 characters per chunk
        ttl: Time-to-live in seconds for cache entries (default: 86400, 1 day)
            - Entries older than TTL are automatically expired on access
            - Accessing an entry refreshes its TTL

    Example:
        >>> import onellm
        >>> onellm.init_cache()  # Enable with defaults
        >>> onellm.init_cache(p=0.9)  # More aggressive matching
        >>> onellm.init_cache(stream_chunk_strategy="sentences", stream_chunk_length=2)
        >>> onellm.init_cache(ttl=3600)  # 1 hour TTL
        >>> response = ChatCompletion.create(...)  # Responses now cached
    """
    global _cache

    from .cache import CacheConfig, SimpleCache

    config = CacheConfig(
        max_entries=max_entries,
        similarity_threshold=p,
        hash_only=hash_only,
        stream_chunk_strategy=stream_chunk_strategy,
        stream_chunk_length=stream_chunk_length,
        ttl=ttl,
    )
    _cache = SimpleCache(config)


def disable_cache():
    """
    Disable caching.

    Example:
        >>> import onellm
        >>> onellm.disable_cache()
    """
    global _cache
    _cache = None


def clear_cache():
    """
    Clear all cached entries.

    Example:
        >>> import onellm
        >>> onellm.clear_cache()
    """
    if _cache:
        _cache.clear()


def cache_stats() -> dict:
    """
    Get cache statistics.

    Returns:
        Dictionary with hits, misses, and entries count

    Example:
        >>> import onellm
        >>> stats = onellm.cache_stats()
        >>> print(f"Hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.1%}")
    """
    if _cache:
        return _cache.stats()
    return {"hits": 0, "misses": 0, "entries": 0}


# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from onellm import openai_api_key, anthropic_api_key
# This allows for a cleaner import experience when working with multiple providers

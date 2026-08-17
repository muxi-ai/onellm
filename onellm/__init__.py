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

    # Auto routing
    "init_routing",    # Register the routing map for model="auto"
    "disable_routing",  # Disable auto routing
    "routing_stats",   # Get routing statistics
    "load_routing_map",  # Parse a routing map from an explicit path
    "explain_route",   # Dry-run a routing decision (no provider call)

    # Connection pooling
    "init_pooling",    # Initialize HTTP connection pooling
    "close_pooling",   # Close all pooled connections

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


# Auto routing - global router instance
_routing = None


def init_routing(
    routing_map: dict,
    *,
    api_keys: dict | None = None,
    embedding_model: str | None = None,
    min_margin: float = 0.05,
    min_score: float | None = None,
    max_depth: int = 3,
    max_slice_tokens: int = 1500,
    recent_user_turns: int = 3,
    memo_size: int = 256,
    validate_similarity: bool = True,
):
    """
    Register the routing map that serves ``model="auto"`` requests.

    A local embedding classifier reads each conversation, picks the
    best-matching label from the map, and resolves it to a concrete
    ``provider/model`` plus an optional fallback chain. The map is 100%
    developer-authored; OneLLM ships no opinions about which model is
    good at what.

    Map schema: a node is a leaf (a ``"provider/model"`` string, a list of
    them forming a fallback chain, or a dict with a ``models`` key plus
    optional ``description``/``examples``) or a group (a dict of child
    labels). Every group requires a ``default``.

    Args:
        routing_map: The routing map. Root must be a group with a ``default``.
        api_keys: Optional provider credentials, applied through the same
            paths as ``set_api_key()``. String values are keys (``env:VAR``
            resolves an environment variable at init); dict values are
            provider config for providers without plain keys (vertexai,
            azure, bedrock).
        embedding_model: ``local/<hf-repo>`` embedding model. Defaults to
            the semantic cache's model so ``init_cache()`` +
            ``init_routing()`` share one loaded backend.
        min_margin: Descend into the top label only if it beats the
            runner-up by this margin (default: 0.05).
        min_score: Optional absolute floor on the top score.
        max_depth: Maximum classification descent depth (default: 3).
        max_slice_tokens: Cap on the assembled routing slice (default: 1500).
        recent_user_turns: How many recent user turns enter the slice.
        memo_size: Bounded LRU of slice-hash -> route memoizations.
        validate_similarity: Warn at init about sibling labels the
            embedding model can barely distinguish (never fatal).

    Raises:
        RoutingConfigurationError: On any invalid map, unknown provider,
            unresolvable credentials, or missing routing dependencies
            (install with ``pip install 'onellm[routing]'``).

    Example:
        >>> import onellm
        >>> onellm.init_routing({
        ...     "default": "openai/gpt-5",
        ...     "code": {
        ...         "default": "anthropic/claude-sonnet-4-5",
        ...         "frontend": "google/gemini-2.5-pro",
        ...     },
        ... })
        >>> response = ChatCompletion.create(model="auto/code", messages=[...])
    """
    global _routing

    from .cache import _CACHE_MODEL_REPO
    from .routing import Router, RoutingConfig

    config = RoutingConfig(
        embedding_model=embedding_model or f"local/{_CACHE_MODEL_REPO}",
        min_margin=min_margin,
        min_score=min_score,
        max_depth=max_depth,
        max_slice_tokens=max_slice_tokens,
        recent_user_turns=recent_user_turns,
        memo_size=memo_size,
        validate_similarity=validate_similarity,
    )
    # Clear first so a failed re-init never leaves a stale router serving
    # requests, and any concurrent create() sees either old-or-none, not
    # a half-built instance
    _routing = None
    _routing = Router(routing_map, config, api_keys=api_keys)


def disable_routing():
    """
    Disable auto routing. Also clears the routing memo.

    Example:
        >>> import onellm
        >>> onellm.disable_routing()
    """
    global _routing
    _routing = None


def routing_stats() -> dict:
    """
    Get routing statistics.

    Returns:
        Dictionary with classified, memo_hits, explicit,
        fell_back_to_default counts and avg_latency_ms.

    Example:
        >>> import onellm
        >>> onellm.routing_stats()
        {'classified': 12, 'memo_hits': 40, 'explicit': 3,
         'fell_back_to_default': 1, 'avg_latency_ms': 3.9}
    """
    if _routing:
        return _routing.stats()
    return {
        "classified": 0,
        "memo_hits": 0,
        "explicit": 0,
        "fell_back_to_default": 0,
        "avg_latency_ms": 0.0,
    }


def load_routing_map(path: str) -> dict:
    """
    Parse a routing map from an explicit path (YAML or JSON).

    This parses, it does not discover: there is no default location, no
    environment lookup, no walking up the directory tree. The caller
    passes the returned dict to ``init_routing()`` themselves. Files may
    carry a top-level ``api_keys`` section; use ``env:VAR`` indirection
    there rather than literal secrets.

    Args:
        path: Path to a ``.yaml``/``.yml`` or ``.json`` routing map file.

    Returns:
        The parsed routing map dict.

    Example:
        >>> import onellm
        >>> onellm.init_routing(onellm.load_routing_map("./routing.yaml"))
    """
    import json

    from .errors import RoutingConfigurationError

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if path.endswith((".yaml", ".yml")):
        import yaml  # type: ignore[import-untyped]

        parsed = yaml.safe_load(raw)
    elif path.endswith(".json"):
        parsed = json.loads(raw)
    else:
        raise RoutingConfigurationError(
            f"load_routing_map({path!r}): unsupported file type; "
            "use .yaml, .yml, or .json"
        )
    if not isinstance(parsed, dict):
        raise RoutingConfigurationError(
            f"load_routing_map({path!r}): expected a mapping at the top level, "
            f"got {type(parsed).__name__}"
        )
    return parsed


def explain_route(
    messages: list | None = None,
    tools: list | None = None,
    entry: str = "auto",
    prompt: str | None = None,
) -> dict:
    """
    Dry-run a routing decision without making a provider call.

    Returns the same dict that ``response.routing`` carries. Useful for
    testing a routing map in CI and answering "what did the classifier
    actually see" when a route looks wrong (pair with
    ``ONELLM_ROUTING_DEBUG=1`` to log the assembled slice).

    Example:
        >>> onellm.explain_route(messages=[...], entry="auto/code")
        {'entry': 'auto/code', 'resolved_path': 'code/frontend', ...}
    """
    from .errors import InvalidConfigurationError

    if _routing is None:
        raise InvalidConfigurationError(
            "explain_route() requires onellm.init_routing(...) to be called first."
        )
    return _routing.resolve(entry, messages=messages, tools=tools, prompt=prompt).to_dict()


# Connection pooling management
def init_pooling(
    max_connections: int = 100,
    max_per_host: int = 20,
    keepalive_timeout: int = 30,
    dns_cache_ttl: int = 300,
    request_timeout: int = 300,
):
    """
    Initialize HTTP connection pooling for improved performance.

    Connection pooling reduces TCP/TLS handshake overhead by reusing connections
    across multiple LLM API calls. This can save 100-300ms per request for
    workflows with sequential LLM calls.

    Pooling is opt-in and disabled by default. If pooling fails for any reason,
    providers silently fall back to creating a new session per request.

    Args:
        max_connections: Maximum total connections across all providers (default: 100)
        max_per_host: Maximum connections per provider/host (default: 20)
        keepalive_timeout: Seconds to keep idle connections alive (default: 30)
        dns_cache_ttl: DNS cache duration in seconds (default: 300)
        request_timeout: Default request timeout in seconds (default: 300)

    Example:
        >>> import onellm
        >>> onellm.init_pooling()  # Enable with defaults
        >>> onellm.init_pooling(max_per_host=50)  # Higher per-provider limit
        >>> # ... use OneLLM normally ...
        >>> await onellm.close_pooling()  # Cleanup on shutdown
    """
    from .http_pool import HTTPConnectionPool, PoolConfig

    config = PoolConfig(
        max_connections=max_connections,
        max_per_host=max_per_host,
        keepalive_timeout=keepalive_timeout,
        dns_cache_ttl=dns_cache_ttl,
        request_timeout=request_timeout,
    )
    HTTPConnectionPool.configure(config)


async def close_pooling():
    """
    Close all pooled HTTP connections.

    Call this on application shutdown to cleanly close all pooled connections.
    After calling this, pooling is disabled until init_pooling() is called again.

    Example:
        >>> import onellm
        >>> onellm.init_pooling()
        >>> # ... use OneLLM ...
        >>> await onellm.close_pooling()  # Cleanup
    """
    from .http_pool import HTTPConnectionPool

    await HTTPConnectionPool.close_all()


# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from onellm import openai_api_key, anthropic_api_key
# This allows for a cleaner import experience when working with multiple providers

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi-llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
muxi-llm: A lightweight, provider-agnostic Python library that offers a unified interface
for interacting with large language models (LLMs) from various providers.

This module serves as the main entry point for the muxi-llm library, exposing all
public APIs and functionality to users. It provides a consistent interface for working
with different LLM providers while maintaining compatibility with the OpenAI API format.
"""

import os

# Public API imports - core functionality
from .chat_completion import ChatCompletion
from .completion import Completion
from .embedding import Embedding

# Media handling
from .audio import AudioTranscription, AudioTranslation
from .files import File
from .image import Image
from .speech import Speech

# Configuration and providers
from .config import get_api_key, get_provider_config, set_api_key
from .providers import get_provider, list_providers, register_provider
from .providers.base import parse_model_name

# Client interface (OpenAI compatibility)
from .client import Client, OpenAI

# Error handling
from .errors import (
    MuxiLLMError,
    APIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
)

# Read version from .version file in the same directory as this file
version_file = os.path.join(os.path.dirname(__file__), ".version")
with open(version_file, "r", encoding="utf-8") as f:
    # Strip whitespace to ensure clean version string
    __version__ = f.read().strip()

# Package metadata
__author__ = "Ran Aroussi"
__license__ = "AGPL-3.0"
__url__ = "https://github.com/ranaroussi/muxi-llm"

# Module exports - defines the public API of the package
# This controls what gets imported when using "from muxi_llm import *"
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

    # Error handling
    "MuxiLLMError",       # Base error class
    "APIError",           # API-related errors
    "AuthenticationError",  # Authentication failures
    "RateLimitError",     # Rate limit exceeded
    "InvalidRequestError",  # Invalid request parameters
]

# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from muxi_llm import openai_api_key, anthropic_api_key
# This allows for a cleaner import experience when working with multiple providers

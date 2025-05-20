#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    with open(version_file, "r", encoding="utf-8") as f:
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

    # Error handling
    "MuxiLLMError",       # Base error class
    "APIError",           # API-related errors
    "AuthenticationError",  # Authentication failures
    "RateLimitError",     # Rate limit exceeded
    "InvalidRequestError",  # Invalid request parameters
]

# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from onellm import openai_api_key, anthropic_api_key
# This allows for a cleaner import experience when working with multiple providers

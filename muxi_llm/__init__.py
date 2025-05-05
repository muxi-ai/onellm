#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
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
    __version__ = f.read().strip()

__author__ = "Ran Aroussi"
__license__ = "AGPL-3.0"
__url__ = "https://github.com/ranaroussi/muxi_llm"

# Module exports
__all__ = [
    # Core functionality
    "ChatCompletion",
    "Completion",
    "Embedding",
    # Media handling
    "File",
    "AudioTranscription",
    "AudioTranslation",
    "Speech",
    "Image",
    # Client interface (OpenAI compatibility)
    "Client",
    "OpenAI",
    # Configuration and providers
    "set_api_key",
    "get_api_key",
    "get_provider",
    "list_providers",
    "register_provider",
    "parse_model_name",
    "get_provider_config",
    # Error handling
    "MuxiLLMError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
]

# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from muxi_llm import openai_api_key, anthropic_api_key

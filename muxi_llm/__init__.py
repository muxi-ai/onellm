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
from .audio import AudioTranscription, AudioTranslation
from .speech import Speech
from .image import Image
from .providers.base import parse_model_name
from .providers import get_provider, list_providers, register_provider
from .config import get_provider_config

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
    "AudioTranscription",
    "AudioTranslation",
    "Speech",
    "Image",
    "get_provider",
    "list_providers",
    "register_provider",
    "parse_model_name",
    "get_provider_config",
]

# Provider-specific API keys can be accessed as globals after they're set:
# e.g., from muxi_llm import openai_api_key, anthropic_api_key

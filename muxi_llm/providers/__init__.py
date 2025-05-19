#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/llm
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
Provider implementations for muxi-llm.

This module imports all available provider implementations,
ensuring they are registered with the provider registry.

The provider system is designed to be extensible, allowing new LLM providers
to be added by implementing the Provider interface and registering them.
"""

from .base import get_provider, list_providers, parse_model_name, register_provider
from .fallback import FallbackProviderProxy
from .openai import OpenAIProvider

# Register provider implementations with the provider registry
# This makes the OpenAI provider available through the get_provider function
# Additional providers should be registered here as they are implemented
register_provider("openai", OpenAIProvider)

# Convenience export - these symbols will be available when importing from muxi_llm.providers
# This allows users to access core provider functionality directly
__all__ = [
    "get_provider",           # Function to get a provider instance by name
    "parse_model_name",       # Function to parse "provider/model" format strings
    "register_provider",      # Function to register new provider implementations
    "list_providers",         # Function to list all registered providers
    "FallbackProviderProxy",  # Class for implementing provider fallback chains
]

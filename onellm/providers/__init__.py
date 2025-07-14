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
Provider implementations for OneLLM.

This module imports all available provider implementations,
ensuring they are registered with the provider registry.

The provider system is designed to be extensible, allowing new LLM providers
to be added by implementing the Provider interface and registering them.
"""

from .base import get_provider, list_providers, parse_model_name, register_provider
from .fallback import FallbackProviderProxy

# Import provider implementations
from .openai import OpenAIProvider
from .mistral import MistralProvider
from .anthropic import AnthropicProvider

# OpenAI-compatible providers
from .groq import GroqProvider
from .xai import XAIProvider
from .openrouter import OpenRouterProvider
from .together import TogetherProvider
from .fireworks import FireworksProvider
from .perplexity import PerplexityProvider
from .deepseek import DeepSeekProvider
from .moonshot import MoonshotProvider
from .google import GoogleProvider
from .azure import AzureProvider
from .anyscale import AnyscaleProvider

# Native API providers
from .cohere import CohereProvider
from .vertexai import VertexAIProvider

# Local providers
from .ollama import OllamaProvider
from .llama_cpp import LlamaCppProvider

# Cloud provider integrations
from .bedrock import BedrockProvider

# Register all provider implementations with the provider registry
# This makes the providers available through the get_provider function

# Original providers
register_provider("openai", OpenAIProvider)
register_provider("mistral", MistralProvider)
register_provider("anthropic", AnthropicProvider)

# OpenAI-compatible providers
register_provider("groq", GroqProvider)
register_provider("xai", XAIProvider)
register_provider("openrouter", OpenRouterProvider)
register_provider("together", TogetherProvider)
register_provider("fireworks", FireworksProvider)
register_provider("perplexity", PerplexityProvider)
register_provider("deepseek", DeepSeekProvider)
register_provider("moonshot", MoonshotProvider)
register_provider("google", GoogleProvider)
register_provider("azure", AzureProvider)
register_provider("anyscale", AnyscaleProvider)

# Native API providers
register_provider("cohere", CohereProvider)
register_provider("vertexai", VertexAIProvider)

# Local providers
register_provider("ollama", OllamaProvider)
register_provider("llama_cpp", LlamaCppProvider)

# Cloud provider integrations
register_provider("bedrock", BedrockProvider)

# Convenience export - these symbols will be available when importing from onellm.providers
# This allows users to access core provider functionality directly
__all__ = [
    "get_provider",  # Function to get a provider instance by name
    "parse_model_name",  # Function to parse "provider/model" format strings
    "register_provider",  # Function to register new provider implementations
    "list_providers",  # Function to list all registered providers
    "FallbackProviderProxy",  # Class for implementing provider fallback chains
]

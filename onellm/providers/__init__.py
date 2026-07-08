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
Provider implementations for OneLLM.

Built-in providers are loaded lazily: importing this package does NOT
import every provider module. A provider module is only imported the
first time it is used - either via get_provider("name") or by accessing
its class here (e.g. ``from onellm.providers import OpenAIProvider``).
This keeps ``import onellm`` fast and avoids pulling in optional
dependencies for providers that are never used.

The provider system is designed to be extensible, allowing new LLM providers
to be added by implementing the Provider interface and registering them
with register_provider().
"""

import importlib

from .base import get_provider, list_providers, parse_model_name, register_provider

# Mapping of public class names to the submodule that defines them,
# resolved on first attribute access (PEP 562)
_PROVIDER_CLASS_MODULES: dict[str, str] = {
    "AnthropicProvider": ".anthropic",
    "AnyscaleProvider": ".anyscale",
    "AzureProvider": ".azure",
    "BedrockProvider": ".bedrock",
    "CohereProvider": ".cohere",
    "DeepSeekProvider": ".deepseek",
    "FallbackProviderProxy": ".fallback",
    "FireworksProvider": ".fireworks",
    "GLMProvider": ".glm",
    "GoogleProvider": ".google",
    "GroqProvider": ".groq",
    "LlamaCppProvider": ".llama_cpp",
    "LocalProvider": ".local",
    "MinimaxProvider": ".minimax",
    "MistralProvider": ".mistral",
    "MoonshotProvider": ".moonshot",
    "OllamaProvider": ".ollama",
    "OpenAIProvider": ".openai",
    "OpenRouterProvider": ".openrouter",
    "PerplexityProvider": ".perplexity",
    "TogetherProvider": ".together",
    "VercelProvider": ".vercel",
    "VertexAIProvider": ".vertexai",
    "XAIProvider": ".xai",
}


def __getattr__(name: str):
    """Resolve provider classes lazily on first access."""
    module_path = _PROVIDER_CLASS_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = importlib.import_module(module_path, __name__)
    except ImportError:
        # Preserve historical behavior for optional-dependency providers:
        # importing the name yields None instead of raising
        if name in ("BedrockProvider", "VertexAIProvider"):
            globals()[name] = None  # cache so __getattr__ is not re-entered
            return None
        raise

    provider_class = getattr(module, name)
    # Cache on the package so __getattr__ is only hit once per class
    globals()[name] = provider_class
    return provider_class


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PROVIDER_CLASS_MODULES))


# Convenience export - these symbols will be available when importing from onellm.providers
# This allows users to access core provider functionality directly
__all__ = [
    "get_provider",  # Function to get a provider instance by name
    "parse_model_name",  # Function to parse "provider/model" format strings
    "register_provider",  # Function to register new provider implementations
    "list_providers",  # Function to list all registered providers
    "FallbackProviderProxy",  # Class for implementing provider fallback chains
]

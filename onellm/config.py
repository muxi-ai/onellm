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
Configuration system for OneLLM.

This module handles configuration from environment variables and runtime settings.
It provides a centralized way to manage API keys, endpoints, and other settings
for various LLM providers.
"""

import os
from typing import Any, Dict, Optional

# Default configuration
DEFAULT_CONFIG = {
    "providers": {
        "openai": {
            "api_key": None,
            "api_base": "https://api.openai.com/v1",
            "organization_id": None,
            "timeout": 60,
            "max_retries": 3,
        },
        "anthropic": {
            "api_key": None,
            "api_base": "https://api.anthropic.com/v1",
            "timeout": 60,
            "max_retries": 3,
        },
        "mistral": {
            "api_key": None,
            "api_base": "https://api.mistral.ai/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "groq": {
            "api_key": None,
            "api_base": "https://api.groq.com/openai/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "xai": {
            "api_key": None,
            "api_base": "https://api.x.ai/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "openrouter": {
            "api_key": None,
            "api_base": "https://openrouter.ai/api/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "together": {
            "api_key": None,
            "api_base": "https://api.together.xyz/v1",
            "timeout": 60,
            "max_retries": 3,
        },
        "fireworks": {
            "api_key": None,
            "api_base": "https://api.fireworks.ai/inference/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "perplexity": {
            "api_key": None,
            "api_base": "https://api.perplexity.ai",
            "timeout": 30,
            "max_retries": 3,
        },
        "deepseek": {
            "api_key": None,
            "api_base": "https://api.deepseek.com/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "moonshot": {
            "api_key": None,
            "api_base": "https://api.moonshot.ai/v1",
            "timeout": 30,
            "max_retries": 3,
        },
        "google": {
            "api_key": None,
            "api_base": "https://generativelanguage.googleapis.com/v1beta",
            "timeout": 30,
            "max_retries": 3,
        },
        "cohere": {
            "api_key": None,
            "api_base": "https://api.cohere.com/v2",
            "timeout": 60,
            "max_retries": 3,
        },
        "vertexai": {
            "service_account_json": None,
            "project_id": None,
            "location": "us-central1",
            "timeout": 60,
            "max_retries": 3,
        },
        "azure": {
            "azure_config_path": None,
            "timeout": 60,
            "max_retries": 3,
            "api_version": "2024-12-01-preview",
        },
        "bedrock": {
            "profile": None,
            "region": "us-east-1",
            "timeout": 60,
            "max_retries": 3,
        },
        "ollama": {
            "api_key": None,  # Not used, but kept for consistency
            "api_base": "http://localhost:11434/v1",
            "timeout": 120,  # Longer timeout for local model inference
            "max_retries": 3,
            "auto_pull": False,  # Whether to auto-pull missing models
        },
        "llama_cpp": {
            "model_dir": None,  # Defaults to ~/llama_models or LLAMA_CPP_MODEL_DIR
            "n_ctx": 2048,  # Context window
            "n_gpu_layers": 0,  # GPU layers (0 = CPU only)
            "n_threads": None,  # Auto-detect CPU cores
            "temperature": 0.7,  # Default temperature
            "timeout": 300,  # 5 minutes for model loading
        },
        "anyscale": {
            "api_key": None,
            "api_base": "https://api.endpoints.anyscale.com/v1",
            "timeout": 60,
            "max_retries": 3,
        },
        # Other providers will be added in future phases
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}

# Global configuration dictionary that will be populated with settings
config = DEFAULT_CONFIG.copy()

# Environment variables prefixes
ENV_PREFIX = "ONELLM_"  # Prefix for OneLLM specific environment variables
PROVIDER_API_KEY_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "anyscale": "ANYSCALE_API_KEY",
}


def _load_env_vars() -> None:
    """
    Load configuration from environment variables.

    This function checks for two types of environment variables:
    1. Variables with ONELLM_ prefix for general configuration
    2. Provider-specific API keys using their standard environment variable names

    Environment variables with double underscores (__) are treated as nested configuration.
    Example: ONELLM_PROVIDERS__OPENAI__TIMEOUT would set config["providers"]["openai"]["timeout"].
    """
    # General configuration
    for key in os.environ:
        if key.startswith(ENV_PREFIX):
            # Extract the config key by removing the prefix
            config_key = key[len(ENV_PREFIX):].lower()

            # Handle nested configuration with double underscores
            if "__" in config_key:
                section, option = config_key.split("__", 1)
                if section in config and option in config[section]:
                    config[section][option] = os.environ[key]
            else:
                if config_key in config:
                    config[config_key] = os.environ[key]

    # Provider API keys (support both prefixed and provider-standard environment variables)
    # This allows users to use either OPENAI_API_KEY or ONELLM_PROVIDERS__OPENAI__API_KEY
    for provider, env_var in PROVIDER_API_KEY_ENV_MAP.items():
        if env_var in os.environ and provider in config["providers"]:
            config["providers"][provider]["api_key"] = os.environ[env_var]

    # Special handling for Vertex AI service account JSON file
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and "vertexai" in config["providers"]:
        config["providers"]["vertexai"]["service_account_json"] = os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"
        ]


def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a nested dictionary with values from another dictionary.

    This is a recursive function that merges nested dictionaries rather than
    replacing them entirely. It's used to update configuration while preserving
    the structure.

    Args:
        d: The target dictionary to update
        u: The source dictionary with new values

    Returns:
        The updated dictionary with merged values
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d


# Load configuration from environment variables on module import
_load_env_vars()


# Public API for configuration
def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for the specified provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic")

    Returns:
        The API key as a string if found, None otherwise
    """
    if provider in config["providers"]:
        return config["providers"][provider].get("api_key")
    return None


def set_api_key(api_key: str, provider: str) -> None:
    """
    Set the API key for the specified provider.

    This function updates both the config dictionary and creates a global
    variable for convenient access to the API key.

    Args:
        api_key: The API key to set
        provider: The provider to set the key for (e.g., "openai", "anthropic")
    """
    if provider in config["providers"]:
        config["providers"][provider]["api_key"] = api_key
        # Set global variable for convenience and backward compatibility
        globals()[f"{provider}_api_key"] = api_key


def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    Get the configuration for the specified provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic")

    Returns:
        A dictionary containing the provider's configuration settings,
        or an empty dictionary if the provider is not found
    """
    if provider in config["providers"]:
        return config["providers"][provider]
    return {}


def update_provider_config(provider: str, **kwargs) -> None:
    """
    Update the configuration for the specified provider.

    Args:
        provider: The provider name to update (e.g., "openai", "anthropic")
        **kwargs: Key-value pairs of configuration settings to update
    """
    if provider in config["providers"]:
        config["providers"][provider].update(kwargs)


# Initialize global variables for all providers for easy access
# This creates variables like openai_api_key, anthropic_api_key, etc.
for provider in config["providers"]:
    # Skip providers that don't use API keys
    if provider == "vertexai":
        continue
    globals()[f"{provider}_api_key"] = get_api_key(provider)

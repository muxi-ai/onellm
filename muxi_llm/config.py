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
Configuration system for muxi-llm.

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
            "api_base": "https://api.anthropic.com",
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
ENV_PREFIX = "MUXI_LLM_"  # Prefix for muxi-llm specific environment variables
PROVIDER_API_KEY_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _load_env_vars() -> None:
    """
    Load configuration from environment variables.

    This function checks for two types of environment variables:
    1. Variables with MUXI_LLM_ prefix for general configuration
    2. Provider-specific API keys using their standard environment variable names

    Environment variables with double underscores (__) are treated as nested configuration.
    Example: MUXI_LLM_PROVIDERS__OPENAI__TIMEOUT would set config["providers"]["openai"]["timeout"].
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
    # This allows users to use either OPENAI_API_KEY or MUXI_LLM_PROVIDERS__OPENAI__API_KEY
    for provider, env_var in PROVIDER_API_KEY_ENV_MAP.items():
        if env_var in os.environ and provider in config["providers"]:
            config["providers"][provider]["api_key"] = os.environ[env_var]


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
        return config["providers"][provider]["api_key"]
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
    globals()[f"{provider}_api_key"] = get_api_key(provider)

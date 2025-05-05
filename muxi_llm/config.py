"""
Configuration system for muxi-llm.

This module handles configuration from environment variables, configuration files,
and runtime settings. It provides a centralized way to manage API keys, endpoints,
and other settings for various LLM providers.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

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

# Global configuration
config = DEFAULT_CONFIG.copy()

# Environment variables prefixes
ENV_PREFIX = "MUXI_LLM_"
PROVIDER_API_KEY_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _load_env_vars() -> None:
    """Load configuration from environment variables."""
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
    for provider, env_var in PROVIDER_API_KEY_ENV_MAP.items():
        if env_var in os.environ and provider in config["providers"]:
            config["providers"][provider]["api_key"] = os.environ[env_var]


def _load_config_file() -> None:
    """Load configuration from YAML file."""
    config_paths = [
        Path.home() / ".muxi" / "config.yaml",
        Path.home() / ".muxi" / "config.yml",
        Path(".muxi-config.yaml"),
        Path(".muxi-config.yml"),
    ]

    for path in config_paths:
        if path.exists():
            with open(path, "r") as f:
                try:
                    file_config = yaml.safe_load(f)
                    if file_config and isinstance(file_config, dict):
                        if "llm" in file_config and isinstance(file_config["llm"], dict):
                            _update_nested_dict(config, file_config["llm"])
                        else:
                            _update_nested_dict(config, file_config)
                    break
                except (yaml.YAMLError, TypeError, ValueError) as e:
                    # Log the error but continue with default config
                    print(f"Error loading config file {path}: {e}")


def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Update a nested dictionary with values from another dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d


# Load configuration from environment variables and config file
_load_env_vars()
_load_config_file()


# Public API for configuration
def get_api_key(provider: str) -> Optional[str]:
    """Get the API key for the specified provider."""
    if provider in config["providers"]:
        return config["providers"][provider]["api_key"]
    return None


def set_api_key(api_key: str, provider: str) -> None:
    """
    Set the API key for the specified provider.

    Args:
        api_key: The API key to set
        provider: The provider to set the key for (e.g., "openai", "anthropic")
    """
    if provider in config["providers"]:
        config["providers"][provider]["api_key"] = api_key
        # Set global variable for backward compatibility and convenience
        globals()[f"{provider}_api_key"] = api_key


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get the configuration for the specified provider."""
    if provider in config["providers"]:
        return config["providers"][provider]
    return {}


def update_provider_config(provider: str, **kwargs) -> None:
    """Update the configuration for the specified provider."""
    if provider in config["providers"]:
        config["providers"][provider].update(kwargs)


# Initialize global variables for all providers
for provider in config["providers"]:
    globals()[f"{provider}_api_key"] = get_api_key(provider)

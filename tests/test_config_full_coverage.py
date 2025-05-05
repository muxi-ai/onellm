"""
Complete test suite for the configuration management system.

These tests target the remaining missing lines in config.py to achieve 100% test coverage.
"""

import os

from muxi_llm.config import (
    ENV_PREFIX,
    PROVIDER_API_KEY_ENV_MAP,
    _load_env_vars,
    get_provider_config,
    update_provider_config,
)
from muxi_llm import config


class TestConfigFullCoverage:
    """Test configuration loading mechanisms for full coverage."""

    def setup_method(self):
        """Setup test by resetting config to default state."""
        # Save the original config
        self.original_config = config.config.copy()

        # Save original environment
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore the original config after test."""
        config.config = self.original_config

        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_env_vars_with_prefixed_vars(self):
        """Test loading environment variables with MUXI_LLM_ prefix."""
        # Set environment variables with the prefix
        os.environ[f"{ENV_PREFIX}LOGGING__LEVEL"] = "DEBUG"
        os.environ[f"{ENV_PREFIX}PROVIDERS__OPENAI__TIMEOUT"] = "120"

        # Also set a top-level config key if it exists
        for key in config.config.keys():
            if not isinstance(config.config[key], dict):
                os.environ[f"{ENV_PREFIX}{key.upper()}"] = "test_value"
                break

        # Reset config to default
        config.config = config.DEFAULT_CONFIG.copy()

        # Call the function to load from environment
        _load_env_vars()

        # Check that the values were loaded correctly
        assert config.config["logging"]["level"] == "DEBUG"
        assert config.config["providers"]["openai"]["timeout"] == "120"

    def test_load_env_vars_with_provider_standard_vars(self):
        """Test loading provider API keys from standard environment variables."""
        # Set environment variables for provider API keys
        for provider, env_var in PROVIDER_API_KEY_ENV_MAP.items():
            os.environ[env_var] = f"{provider}-test-api-key"

        # Reset config to default
        config.config = config.DEFAULT_CONFIG.copy()

        # Call the function to load from environment
        _load_env_vars()

        # Check that the API keys were loaded correctly
        for provider in PROVIDER_API_KEY_ENV_MAP.keys():
            if provider in config.config["providers"]:
                assert config.config["providers"][provider]["api_key"] == f"{provider}-test-api-key"

    def test_update_provider_config_with_all_options(self):
        """Test updating provider config with all available options."""
        provider = "openai"

        # Create a dictionary with all possible configuration options
        all_options = {
            "api_key": "all-options-key",
            "api_base": "https://custom-api-endpoint.com",
            "organization_id": "org-12345",
            "timeout": 45,
            "max_retries": 5,
            "new_option1": "value1",
            "new_option2": "value2"
        }

        # Update the provider config
        update_provider_config(provider, **all_options)

        # Check that all options were set correctly
        provider_config = get_provider_config(provider)
        for key, value in all_options.items():
            assert provider_config[key] == value

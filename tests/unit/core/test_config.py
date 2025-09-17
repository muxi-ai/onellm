"""
Tests for the configuration system in OneLLM.

These tests verify that the configuration system works correctly,
especially for edge cases and environment variable handling.
"""

import os
import pytest

from onellm.config import (
    _load_env_vars,
    _update_nested_dict,
    config,
    get_provider_config,
    set_api_key,
    update_provider_config,
)


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reset configuration after each test."""
    original_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            original_config[key] = value.copy()
        else:
            original_config[key] = value

    yield

    # Restore original config
    for key in list(config.keys()):
        config.pop(key, None)
    config.update(original_config)


class TestConfigSystem:
    """Tests for the OneLLM configuration system."""

    def test_direct_api_key_setting(self):
        """Test setting API key directly."""
        # Set key directly
        set_api_key("test-key", "openai")

        # Verify the key was set
        provider_config = get_provider_config("openai")
        assert provider_config["api_key"] == "test-key"

    def test_openai_api_key_env_var(self, monkeypatch):
        """Test standard OpenAI API key environment variable."""
        # Remove any existing API key
        update_provider_config("openai", api_key=None)

        # Set environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-env")

        # Reload environment variables
        os.environ["OPENAI_API_KEY"] = "test-key-env"  # Ensure it's in os.environ for test
        _load_env_vars()

        # Verify the key was loaded
        provider_config = get_provider_config("openai")
        assert provider_config["api_key"] == "test-key-env"

    @pytest.mark.parametrize("timeout_value", [10, 30, 60, 120])
    def test_timeout_configuration(self, timeout_value):
        """Test setting different timeout values."""
        # Set a custom timeout
        update_provider_config("openai", timeout=timeout_value)

        # Verify the timeout was set correctly
        provider_config = get_provider_config("openai")
        assert provider_config["timeout"] == timeout_value

    def test_config_override_precedence(self, monkeypatch):
        """Test that direct configuration overrides environment variables."""
        # Remove any existing API key
        update_provider_config("openai", api_key=None)

        # Set environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        os.environ["OPENAI_API_KEY"] = "env-key"  # Ensure it's in os.environ for test
        _load_env_vars()

        # Verify environment variable was loaded
        provider_config = get_provider_config("openai")
        assert provider_config["api_key"] == "env-key"

        # Override with direct configuration
        set_api_key("direct-key", "openai")

        # Check that direct configuration takes precedence
        provider_config = get_provider_config("openai")
        assert provider_config["api_key"] == "direct-key"

    def test_invalid_provider_name(self):
        """Test that invalid provider names don't cause errors."""
        config = get_provider_config("nonexistent_provider")
        assert config == {}

    def test_nested_configuration(self):
        """Test setting and getting nested configuration values."""
        # Create a nested update
        nested_update = {"advanced": {"retry_on_timeout": True, "retry_count": 3}}

        # Update the config
        provider_config = get_provider_config("openai")
        _update_nested_dict(provider_config, nested_update)

        # Get the configuration and verify nested values
        updated_config = get_provider_config("openai")
        assert "advanced" in updated_config
        assert updated_config["advanced"]["retry_on_timeout"] is True
        assert updated_config["advanced"]["retry_count"] == 3

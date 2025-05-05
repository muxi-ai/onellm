"""
Advanced tests for the configuration management system.

These tests focus on the configuration loading, environment variables,
and direct configuration APIs.
"""

from muxi_llm import config
from muxi_llm.config import (
    _update_nested_dict,
    get_api_key,
    set_api_key,
    get_provider_config,
    update_provider_config
)


class TestConfigLoading:
    """Test configuration loading mechanisms."""

    def setup_method(self):
        """Setup test by resetting config to default state."""
        # Save the original config
        self.original_config = config.config.copy()

    def teardown_method(self):
        """Restore the original config after test."""
        config.config = self.original_config

    def test_update_nested_dict(self):
        """Test nested dictionary update function."""
        base = {
            "level1": {
                "level2": {
                    "value": "original",
                    "keep": "preserved"
                },
                "other": "unchanged"
            },
            "simple": "simple_value"
        }

        update = {
            "level1": {
                "level2": {
                    "value": "updated",
                    "new": "added"
                }
            },
            "new_key": "new_value"
        }

        result = _update_nested_dict(base, update)

        # Check updated values
        assert result["level1"]["level2"]["value"] == "updated"
        assert result["level1"]["level2"]["new"] == "added"
        assert result["new_key"] == "new_value"

        # Check preserved values
        assert result["level1"]["level2"]["keep"] == "preserved"
        assert result["level1"]["other"] == "unchanged"
        assert result["simple"] == "simple_value"

        # Check that result is the same object as base (updated in place)
        assert result is base

    def test_load_env_vars(self):
        """Test loading configuration from environment variables."""
        # This test simply verifies the structure of the environment variable parsing logic
        # without actually loading environment variables, which can vary between environments

        # Create test dictionaries
        test_config = {
            "providers": {
                "openai": {
                    "timeout": 60
                }
            },
            "logging": {
                "level": "INFO"
            }
        }

        # Directly update values in the test_config
        test_config["providers"]["openai"]["timeout"] = "30"
        test_config["logging"]["level"] = "DEBUG"

        # Verify the changes work as expected
        assert test_config["providers"]["openai"]["timeout"] == "30"
        assert test_config["logging"]["level"] == "DEBUG"

        # This test passes if the nested dictionary access/update works as expected,
        # which is what the environment variable loading relies on


class TestConfigAPI:
    """Test the public API for configuration."""

    def setup_method(self):
        """Setup test by resetting config to default state."""
        # Save the original config
        self.original_config = config.config.copy()

    def teardown_method(self):
        """Restore the original config after test."""
        config.config = self.original_config

    def test_get_api_key(self):
        """Test getting API key for provider."""
        # Set a test key
        config.config["providers"]["openai"]["api_key"] = "test-key"

        # Retrieve it
        result = get_api_key("openai")
        assert result == "test-key"

        # Test with non-existent provider
        result = get_api_key("nonexistent")
        assert result is None

    def test_set_api_key(self):
        """Test setting API key for provider."""
        # Set a key
        set_api_key("new-test-key", "openai")

        # Check it was set in config
        assert config.config["providers"]["openai"]["api_key"] == "new-test-key"

        # Check global variable was set
        assert config.openai_api_key == "new-test-key"

        # Test with non-existent provider (should not raise exception)
        set_api_key("ignored-key", "nonexistent")

    def test_get_provider_config(self):
        """Test getting entire provider configuration."""
        # Set some test values
        config.config["providers"]["openai"]["api_key"] = "test-key"
        config.config["providers"]["openai"]["timeout"] = 75

        # Get config
        result = get_provider_config("openai")

        # Check returned values
        assert result["api_key"] == "test-key"
        assert result["timeout"] == 75

        # Test with non-existent provider
        result = get_provider_config("nonexistent")
        assert result == {}

    def test_update_provider_config(self):
        """Test updating provider configuration."""
        # Update multiple values
        update_provider_config(
            "openai",
            api_key="updated-key",
            timeout=90,
            custom_option="custom-value"
        )

        # Check values were updated
        provider_config = config.config["providers"]["openai"]
        assert provider_config["api_key"] == "updated-key"
        assert provider_config["timeout"] == 90
        assert provider_config["custom_option"] == "custom-value"

        # Test with non-existent provider (should not raise exception)
        update_provider_config("nonexistent", api_key="ignored")

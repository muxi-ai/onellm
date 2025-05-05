#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Additional tests for the base provider module, focused on coverage improvements.

These tests specifically target code paths that aren't covered by existing tests.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, parse_model_name, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY
)
from muxi_llm.utils.fallback import FallbackConfig


class TestModelNameParsing:
    """Tests for model name parsing."""

    def test_parse_model_name_invalid(self):
        """Test parsing an invalid model name."""
        with pytest.raises(ValueError) as excinfo:
            parse_model_name("invalid-model-name")

        # Verify error message contains guidance
        error_msg = str(excinfo.value)
        assert "does not contain a provider prefix" in error_msg
        assert "Use format 'provider/model-name'" in error_msg


class SimpleTestProvider(Provider):
    """A simple test provider implementation."""

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock implementation."""
        return {"choices": [{"message": {"content": "Simple test response"}}]}

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Mock implementation."""
        return {"choices": [{"text": "Simple test response"}]}

    async def create_embedding(self, input, model, **kwargs):
        """Mock implementation."""
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        """Mock implementation."""
        return {"id": "file-123", "filename": "file.txt"}

    async def download_file(self, file_id, **kwargs):
        """Mock implementation."""
        return b"Mock file content"


class TestProviderRegistry:
    """Tests for the provider registry functionality."""

    def setup_method(self):
        """Set up the test environment."""
        # Save the original registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

        # Clear the registry for testing
        _PROVIDER_REGISTRY.clear()

    def teardown_method(self):
        """Clean up the test environment."""
        # Restore the original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_register_provider(self):
        """Test registering a provider."""
        # Register a test provider
        register_provider("simple", SimpleTestProvider)

        # Verify it was added to the registry
        assert "simple" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["simple"] is SimpleTestProvider

    def test_get_provider_unsupported(self):
        """Test getting a provider that isn't registered."""
        # Try to get a provider that doesn't exist
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent")

        # Verify error message contains supported providers
        error_msg = str(excinfo.value)
        assert "Provider 'nonexistent' is not supported" in error_msg
        assert "Supported providers:" in error_msg

    def test_list_providers_empty(self):
        """Test listing providers when registry is empty."""
        # Verify empty list is returned
        providers = list_providers()
        assert isinstance(providers, list)
        assert len(providers) == 0

    def test_get_provider_with_args(self):
        """Test getting a provider with constructor arguments."""
        # Register a provider with a spy on its constructor
        mock_provider_class = mock.Mock(return_value=SimpleTestProvider())
        register_provider("test", mock_provider_class)

        # Get the provider with kwargs
        provider = get_provider("test", api_key="test-key", timeout=30)

        # Verify the mock was called with the kwargs
        mock_provider_class.assert_called_once_with(api_key="test-key", timeout=30)

        # Verify we got a provider back
        assert isinstance(provider, Provider)


class TestGetProviderWithFallbacks:
    """Tests for get_provider_with_fallbacks function."""

    def setup_method(self):
        """Set up the test environment."""
        # Register test provider
        register_provider("test", SimpleTestProvider)

        # Mock FallbackProviderProxy - fix the import path
        self.mock_fallback_proxy = mock.patch(
            "muxi_llm.providers.fallback.FallbackProviderProxy"
        ).start()

        # Create a mock proxy instance
        self.mock_proxy_instance = mock.Mock()
        self.mock_fallback_proxy.return_value = self.mock_proxy_instance

    def teardown_method(self):
        """Clean up the test environment."""
        # Stop all mocks
        mock.patch.stopall()

        # Deregister test provider if it exists
        if "test" in _PROVIDER_REGISTRY:
            del _PROVIDER_REGISTRY["test"]

    def test_without_fallbacks(self):
        """Test getting provider without fallbacks."""
        # Call with no fallbacks
        provider, model = get_provider_with_fallbacks("test/model-a")

        # Verify no fallback proxy was created
        self.mock_fallback_proxy.assert_not_called()

        # Verify correct provider and model are returned
        assert isinstance(provider, SimpleTestProvider)
        assert model == "model-a"

    def test_with_fallbacks(self):
        """Test getting provider with fallbacks."""
        # Set up fallbacks
        fallbacks = ["another/model-b", "yet-another/model-c"]

        # Call with fallbacks
        provider, model = get_provider_with_fallbacks(
            "test/model-a",
            fallback_models=fallbacks
        )

        # Verify fallback proxy was created with correct models
        self.mock_fallback_proxy.assert_called_once()
        call_args = self.mock_fallback_proxy.call_args[0]
        assert call_args[0] == ["test/model-a"] + fallbacks

        # Verify model is correct
        assert model == "model-a"

        # Verify we got the proxy
        assert provider is self.mock_proxy_instance

    def test_with_fallback_config(self):
        """Test with a custom fallback configuration."""
        # Create fallback config
        fallback_config = FallbackConfig(max_fallbacks=1, log_fallbacks=True)

        # Call with fallbacks and config
        provider, model = get_provider_with_fallbacks(
            "test/model-a",
            fallback_models=["another/model-b"],
            fallback_config=fallback_config
        )

        # Verify fallback proxy was created with correct config
        self.mock_fallback_proxy.assert_called_once()
        call_args = self.mock_fallback_proxy.call_args[0]
        # Check that the first argument is the list of models
        assert call_args[0] == ["test/model-a", "another/model-b"]
        # Check that the second positional argument is the fallback config
        # That's how it's passed in the implementation
        assert len(call_args) >= 2
        assert call_args[1] is fallback_config


class TestProviderClassMethods:
    """Tests for Provider class methods."""

    def test_get_provider_name(self):
        """Test the get_provider_name class method."""
        # Create a custom provider class
        class CustomTestProvider(Provider):
            async def create_chat_completion(self, *args, **kwargs):
                pass

            async def create_completion(self, *args, **kwargs):
                pass

            async def create_embedding(self, *args, **kwargs):
                pass

            async def upload_file(self, *args, **kwargs):
                pass

            async def download_file(self, *args, **kwargs):
                pass

        # Call get_provider_name
        provider_name = CustomTestProvider.get_provider_name()

        # Verify name is extracted and lowercase
        assert provider_name == "customtest"

        # Test with a provider that doesn't have "Provider" in the name
        class Custom(Provider):
            async def create_chat_completion(self, *args, **kwargs):
                pass

            async def create_completion(self, *args, **kwargs):
                pass

            async def create_embedding(self, *args, **kwargs):
                pass

            async def upload_file(self, *args, **kwargs):
                pass

            async def download_file(self, *args, **kwargs):
                pass

        # Verify the name is just lowercase of the class
        assert Custom.get_provider_name() == "custom"

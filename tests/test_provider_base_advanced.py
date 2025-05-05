"""
Advanced tests for the provider base functionality.

This module tests additional functionality in the provider base module,
including error cases, provider registration, and other edge cases.
"""

import inspect
from importlib import import_module
import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider,
    parse_model_name,
    get_provider,
    get_provider_with_fallbacks,
    register_provider,
    list_providers,
    _PROVIDER_REGISTRY
)
from muxi_llm.utils.fallback import FallbackConfig


class TestProviderMethods:
    """Tests for the Provider abstract base class methods."""

    def test_abstract_methods_enforcement(self):
        """Test that Provider cannot be instantiated without implementing abstract methods."""
        # Attempt to instantiate the abstract base class should fail
        with pytest.raises(TypeError) as excinfo:
            Provider()

        # Error message should mention abstract methods
        assert "abstract" in str(excinfo.value).lower()

    def test_get_provider_name(self):
        """Test the get_provider_name class method."""
        # Create a concrete subclass with just the get_provider_name method
        class TestProvider(Provider):
            pass

        # Should derive name from class name
        assert TestProvider.get_provider_name() == "test"

        # Test with a differently named class
        class MyCustomProvider(Provider):
            pass

        assert MyCustomProvider.get_provider_name() == "mycustom"


class TestProviderRegistry:
    """Tests for provider registration and retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        # Save the original registry to restore it after tests
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore the original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_register_provider(self):
        """Test registering a provider class."""
        # Create a mock provider class
        class MockProvider(Provider):
            pass

        # Register it
        register_provider("mock", MockProvider)

        # Check that it was added to the registry
        assert "mock" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["mock"] == MockProvider

    def test_register_duplicate_provider(self):
        """Test registering a provider with the same name."""
        # Create two mock provider classes
        class FirstProvider(Provider):
            pass

        class SecondProvider(Provider):
            pass

        # Register the first one
        register_provider("duplicate", FirstProvider)
        assert _PROVIDER_REGISTRY["duplicate"] == FirstProvider

        # Register the second one with the same name - should override
        register_provider("duplicate", SecondProvider)
        assert _PROVIDER_REGISTRY["duplicate"] == SecondProvider

    def test_list_providers(self):
        """Test listing registered providers."""
        # Clear the registry for this test
        _PROVIDER_REGISTRY.clear()

        # Create and register mock providers
        class MockProviderA(Provider):
            pass

        class MockProviderB(Provider):
            pass

        register_provider("mock_a", MockProviderA)
        register_provider("mock_b", MockProviderB)

        # Get provider list
        providers = list_providers()

        # Should contain our registered providers
        assert "mock_a" in providers
        assert "mock_b" in providers
        assert len(providers) == 2


class TestGetProvider:
    """Tests for the get_provider function."""

    def test_get_provider_unknown(self):
        """Test get_provider with an unknown provider name."""
        # Use a name that definitely doesn't exist
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent_provider_name")

        # Error message should list available providers
        error_msg = str(excinfo.value)
        assert "not supported" in error_msg

        # Should list available providers
        assert "Supported providers" in error_msg
        for provider in list_providers():
            assert provider in error_msg

    @mock.patch("muxi_llm.providers.base.import_module")
    @mock.patch("muxi_llm.providers.base.inspect.getmembers")
    def test_get_provider_with_dynamic_import(self, mock_getmembers, mock_import_module):
        """Test get_provider with dynamic provider import."""
        # Create a mock provider class
        class MockProvider(Provider):
            def __init__(self, **kwargs):
                self.config = kwargs

        # Set up the mock to return our provider class
        mock_module = mock.Mock()
        mock_import_module.return_value = mock_module

        # Set up inspect to find our provider class
        mock_getmembers.return_value = [
            ("SomeClass", type("SomeClass", (), {})),
            ("MockProvider", MockProvider)
        ]

        # Call get_provider - should use our mocked module
        provider = get_provider("mock", api_key="test-key")

        # Verify the provider was created correctly
        assert isinstance(provider, MockProvider)
        assert provider.config["api_key"] == "test-key"

        # Verify import_module was called correctly
        mock_import_module.assert_called_once_with("muxi_llm.providers.mock")

    @mock.patch("muxi_llm.providers.base.import_module")
    @mock.patch("muxi_llm.providers.base.inspect.getmembers")
    def test_get_provider_no_matching_class(self, mock_getmembers, mock_import_module):
        """Test get_provider when no matching provider class is found."""
        # Set up inspect to return classes that don't match the provider name pattern
        mock_getmembers.return_value = [
            ("SomeClass", type("SomeClass", (), {})),
            ("OtherClass", type("OtherClass", (), {}))
        ]

        # Call get_provider - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            get_provider("mock")

        # Error message should mention that no provider class was found
        error_msg = str(excinfo.value)
        assert "not supported" in error_msg


class TestGetProviderWithFallbacks:
    """Tests for the get_provider_with_fallbacks function."""

    def test_without_fallbacks(self):
        """Test get_provider_with_fallbacks without fallback models."""
        # Mock the get_provider function
        with mock.patch("muxi_llm.providers.base.get_provider") as mock_get_provider:
            mock_provider = mock.Mock()
            mock_get_provider.return_value = mock_provider

            # Call function without fallbacks
            provider, model_name = get_provider_with_fallbacks("openai/gpt-4")

            # Should return the provider directly, not wrapped
            assert provider == mock_provider
            assert model_name == "gpt-4"

            # Verify get_provider was called with the right provider name
            mock_get_provider.assert_called_once_with("openai")

    def test_with_custom_fallback_config(self):
        """Test get_provider_with_fallbacks with a custom fallback configuration."""
        # Mock the FallbackProviderProxy
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_proxy:
            mock_proxy_instance = mock.Mock()
            mock_proxy.return_value = mock_proxy_instance

            # Create a fallback config
            fallback_config = FallbackConfig(
                max_retries=5,
                max_fallbacks=2,
                log_fallbacks=True,
                fallback_callback=lambda **kwargs: print("Fallback occurred")
            )

            # Call the function with fallback models and config
            provider, model_name = get_provider_with_fallbacks(
                primary_model="openai/gpt-4",
                fallback_models=["anthropic/claude-2"],
                fallback_config=fallback_config
            )

            # Check results
            assert provider == mock_proxy_instance
            assert model_name == "gpt-4"

            # Verify FallbackProviderProxy was called with the right parameters
            mock_proxy.assert_called_once()
            args, kwargs = mock_proxy.call_args
            assert args[0] == ["openai/gpt-4", "anthropic/claude-2"]
            assert kwargs["fallback_config"] == fallback_config

    def test_nested_model_name(self):
        """Test handling model names with nested paths."""
        # Test model name with multiple slashes
        with mock.patch("muxi_llm.providers.base.get_provider") as mock_get_provider:
            mock_provider = mock.Mock()
            mock_get_provider.return_value = mock_provider

            # Call with a nested model name
            provider, model_name = get_provider_with_fallbacks("openai/fine-tuned/davinci")

            # Check results
            assert provider == mock_provider
            assert model_name == "fine-tuned/davinci"

            # Verify get_provider was called with just the provider name
            mock_get_provider.assert_called_once_with("openai")

    def test_invalid_model_name(self):
        """Test handling invalid model names."""
        with pytest.raises(ValueError) as excinfo:
            get_provider_with_fallbacks("invalid-model-name")

        assert "does not contain a provider prefix" in str(excinfo.value)

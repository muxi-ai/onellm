#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete coverage tests for providers/base.py.

This file contains comprehensive tests designed to achieve 100% code coverage
for the providers/base.py module.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, parse_model_name, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY
)
from muxi_llm.utils.fallback import FallbackConfig


class FullCoverageProvider(Provider):
    """A concrete Provider implementation for testing."""

    @classmethod
    def get_provider_name(cls) -> str:
        """Override to ensure we cover the Provider.get_provider_name method."""
        # Call the parent method directly to ensure it's covered
        result = super().get_provider_name()
        # Verify the result matches our expectations
        assert result == "fullcoverage"
        return result

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Implement abstract method."""
        return {"choices": [{"message": {"content": "Coverage test"}}]}

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Implement abstract method."""
        return {"choices": [{"text": "Coverage test"}]}

    async def create_embedding(self, input, model, **kwargs):
        """Implement abstract method."""
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        """Implement abstract method."""
        return {"id": "file-123"}

    async def download_file(self, file_id, **kwargs):
        """Implement abstract method."""
        return b"file content"


class TestProviderBaseComplete:
    """Tests for achieving complete coverage of providers/base.py."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Save the original provider registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Clean up after each test."""
        # Restore the original provider registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_parse_model_name_valid(self):
        """Test parse_model_name with a valid model name."""
        provider, model = parse_model_name("openai/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_parse_model_name_invalid(self):
        """Test parse_model_name with an invalid model name."""
        with pytest.raises(ValueError) as excinfo:
            parse_model_name("invalid-model")

        error_msg = str(excinfo.value)
        assert "does not contain a provider prefix" in error_msg
        assert "Use format 'provider/model-name'" in error_msg

    def test_provider_get_provider_name_class_method(self):
        """Test Provider.get_provider_name as a class method."""
        # This directly tests line 91
        name = FullCoverageProvider.get_provider_name()
        assert name == "fullcoverage"

    def test_provider_get_provider_name_instance_method(self):
        """Test Provider.get_provider_name as an instance method."""
        # This also tests line 91 but through an instance
        provider = FullCoverageProvider()
        name = provider.get_provider_name()
        assert name == "fullcoverage"

    def test_provider_abstract_methods(self):
        """Test that Provider has properly marked abstract methods."""
        # This tests lines around 113
        # Create a class that doesn't implement all abstract methods
        class IncompleteProvider(Provider):
            # Missing implementations
            pass

        # Should raise TypeError due to abstract methods
        with pytest.raises(TypeError) as excinfo:
            IncompleteProvider()

        error_msg = str(excinfo.value)
        assert "Can't instantiate abstract class" in error_msg
        assert "create_chat_completion" in error_msg
        assert "create_completion" in error_msg
        assert "create_embedding" in error_msg
        assert "upload_file" in error_msg
        assert "download_file" in error_msg

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider."""
        # This will test standard path of get_provider
        _PROVIDER_REGISTRY.clear()
        register_provider("test", FullCoverageProvider)

        # Verify provider was registered
        assert "test" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["test"] is FullCoverageProvider

        # Get the provider and check it's the right type
        provider = get_provider("test")
        assert isinstance(provider, FullCoverageProvider)

    def test_get_provider_not_found(self):
        """Test get_provider with an unknown provider."""
        # This directly tests line 133
        _PROVIDER_REGISTRY.clear()
        register_provider("known", FullCoverageProvider)

        # Try to get an unknown provider
        with pytest.raises(ValueError) as excinfo:
            get_provider("unknown")

        error_msg = str(excinfo.value)
        assert "Provider 'unknown' is not supported" in error_msg
        assert "Supported providers: known" in error_msg

    def test_list_providers_empty(self):
        """Test list_providers with empty registry."""
        # This directly tests line 153
        _PROVIDER_REGISTRY.clear()
        providers = list_providers()
        assert isinstance(providers, list)
        assert len(providers) == 0

    def test_list_providers_populated(self):
        """Test list_providers with populated registry."""
        # This also tests line 153 but with providers
        _PROVIDER_REGISTRY.clear()
        register_provider("provider1", mock.MagicMock())
        register_provider("provider2", mock.MagicMock())

        providers = list_providers()
        assert isinstance(providers, list)
        assert set(providers) == {"provider1", "provider2"}
        assert len(providers) == 2

    def test_get_provider_with_fallbacks_no_fallbacks(self):
        """Test get_provider_with_fallbacks without fallbacks."""
        # This directly tests line 171
        _PROVIDER_REGISTRY.clear()
        register_provider("test", FullCoverageProvider)

        # Call with no fallbacks
        provider, model = get_provider_with_fallbacks("test/model-x")

        # Verify results
        assert isinstance(provider, FullCoverageProvider)
        assert model == "model-x"

    def test_get_provider_with_fallbacks_with_fallbacks(self):
        """Test get_provider_with_fallbacks with fallbacks."""
        # This tests the fallback path

        # Create a mock FallbackProviderProxy
        mock_proxy = mock.MagicMock()

        # Patch the import
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_proxy_class:
            mock_proxy_class.return_value = mock_proxy

            # Call with fallbacks
            provider, model = get_provider_with_fallbacks(
                "test/model-x",
                fallback_models=["other/model-y"]
            )

            # Verify FallbackProviderProxy was created
            mock_proxy_class.assert_called_once_with(
                ["test/model-x", "other/model-y"],
                None
            )

            # Verify results
            assert provider is mock_proxy
            assert model == "model-x"

    def test_get_provider_with_fallbacks_with_config(self):
        """Test get_provider_with_fallbacks with fallbacks and config."""
        # Create a mock FallbackProviderProxy
        mock_proxy = mock.MagicMock()

        # Create a fallback config
        fallback_config = FallbackConfig(max_fallbacks=2, log_fallbacks=True)

        # Patch the import
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_proxy_class:
            mock_proxy_class.return_value = mock_proxy

            # Call with fallbacks and config
            provider, model = get_provider_with_fallbacks(
                "test/model-x",
                fallback_models=["other/model-y"],
                fallback_config=fallback_config
            )

            # Verify FallbackProviderProxy was created with config
            mock_proxy_class.assert_called_once_with(
                ["test/model-x", "other/model-y"],
                fallback_config
            )

            # Verify results
            assert provider is mock_proxy
            assert model == "model-x"

    def test_comprehensive_workflow(self):
        """A comprehensive test combining all functionality."""
        # Clear registry for clean test
        _PROVIDER_REGISTRY.clear()

        # 1. Register provider
        register_provider("complete", FullCoverageProvider)

        # 2. List providers
        providers = list_providers()
        assert "complete" in providers

        # 3. Get provider
        provider = get_provider("complete")
        assert isinstance(provider, FullCoverageProvider)

        # 4. Parse model name
        provider_name, model_name = parse_model_name("complete/model-z")
        assert provider_name == "complete"
        assert model_name == "model-z"

        # 5. Get provider with fallbacks disabled
        provider, model = get_provider_with_fallbacks("complete/model-z")
        assert isinstance(provider, FullCoverageProvider)
        assert model == "model-z"

        # 6. Try unknown provider
        with pytest.raises(ValueError):
            get_provider("nonexistent")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final coverage tests for the providers/base.py module.

These tests are designed to achieve 100% coverage for the base provider module.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, get_provider,
    list_providers, get_provider_with_fallbacks, _PROVIDER_REGISTRY
)


class TestProviderBaseFinalCoverage:
    """Tests designed to achieve complete coverage of the Provider base class."""

    def setup_method(self):
        """Setup test environment before each test."""
        # Save a copy of the original registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Restore the original provider registry after each test."""
        # Restore the original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_provider_get_provider_name(self):
        """Test the Provider.get_provider_name method (line 91)."""
        # Create a concrete subclass of Provider for testing
        class TestProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                pass

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                pass

            async def create_embedding(self, input, model, **kwargs):
                pass

            async def upload_file(self, file, purpose, **kwargs):
                pass

            async def download_file(self, file_id, **kwargs):
                pass

        # Test the class method directly without instantiating
        assert TestProvider.get_provider_name() == "test"

        # Test with another class name
        class CustomNameProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                pass

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                pass

            async def create_embedding(self, input, model, **kwargs):
                pass

            async def upload_file(self, file, purpose, **kwargs):
                pass

            async def download_file(self, file_id, **kwargs):
                pass

        assert CustomNameProvider.get_provider_name() == "customname"

    def test_get_provider_not_found(self):
        """Test get_provider with an unknown provider (line 133)."""
        # Ensure the registry is empty or has known values
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY["test"] = mock.MagicMock()

        # Attempt to get a provider that doesn't exist
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent")

        # Verify the error message includes the supported providers
        error_msg = str(excinfo.value)
        assert "Provider 'nonexistent' is not supported" in error_msg
        assert "Supported providers: test" in error_msg

    def test_list_providers(self):
        """Test the list_providers function (line 153)."""
        # Set up a known registry state
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update({
            "provider1": mock.MagicMock(),
            "provider2": mock.MagicMock(),
            "provider3": mock.MagicMock()
        })

        # Get the list of providers
        providers = list_providers()

        # Verify the result
        assert isinstance(providers, list)
        assert set(providers) == {"provider1", "provider2", "provider3"}
        assert len(providers) == 3

    def test_get_provider_with_fallbacks_no_fallbacks(self):
        """Test get_provider_with_fallbacks without fallback models (line 171)."""
        # Mock the get_provider function
        with mock.patch("muxi_llm.providers.base.get_provider") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_get_provider.return_value = mock_provider

            # Call the function without fallback models
            provider, model = get_provider_with_fallbacks("openai/gpt-4")

            # Verify that get_provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

            # Verify the results
            assert provider == mock_provider
            assert model == "gpt-4"

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that Provider cannot be instantiated directly as an abstract class."""
        with pytest.raises(TypeError):
            # Attempting to instantiate an abstract class should raise TypeError
            Provider()

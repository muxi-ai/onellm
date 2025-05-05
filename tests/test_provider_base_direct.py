#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct coverage tests for the providers/base.py module.

These tests use direct function/method access to reach specific code paths.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, parse_model_name, get_provider,
    list_providers, get_provider_with_fallbacks, _PROVIDER_REGISTRY
)


class ConcreteProvider(Provider):
    """Concrete implementation of Provider for testing."""

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Concrete implementation of create_chat_completion."""
        return {"choices": [{"message": {"content": "Response"}}]}

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Concrete implementation of create_completion."""
        return {"choices": [{"text": "Response"}]}

    async def create_embedding(self, input, model, **kwargs):
        """Concrete implementation of create_embedding."""
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        """Concrete implementation of upload_file."""
        return {"id": "file_123"}

    async def download_file(self, file_id, **kwargs):
        """Concrete implementation of download_file."""
        return b"file_content"


class TestDirectProviderCoverage:
    """Tests that directly call specific functions in the base provider module."""

    def test_provider_class_name_method(self):
        """Test the get_provider_name method directly (line 91)."""
        # Create a provider instance
        provider = ConcreteProvider()

        # Call the method directly
        result = provider.get_provider_name()

        # Verify it returns the expected value
        assert result == "concrete"

        # Also test as a class method
        assert ConcreteProvider.get_provider_name() == "concrete"

    def test_abstractmethod_attribute(self):
        """Test that the abstractmethod decorator is applied correctly (line 113)."""
        # Check if create_chat_completion has the __isabstractmethod__ attribute
        assert getattr(Provider.create_chat_completion, "__isabstractmethod__", False) is True

    def test_provider_registry_get(self):
        """Test the get_provider function with a non-existent provider (line 133)."""
        # Save the original registry
        original_registry = _PROVIDER_REGISTRY.copy()

        try:
            # Clear the registry
            _PROVIDER_REGISTRY.clear()

            # Try to get a provider that doesn't exist
            with pytest.raises(ValueError) as excinfo:
                get_provider("nonexistent")

            # Verify the error message
            error_msg = str(excinfo.value)
            assert "Provider 'nonexistent' is not supported" in error_msg
            assert "Supported providers: " in error_msg

        finally:
            # Restore the original registry
            _PROVIDER_REGISTRY.clear()
            _PROVIDER_REGISTRY.update(original_registry)

    def test_list_providers_direct(self):
        """Test the list_providers function directly (line 153)."""
        # Save the original registry
        original_registry = _PROVIDER_REGISTRY.copy()

        try:
            # Set up a controlled registry state
            _PROVIDER_REGISTRY.clear()
            _PROVIDER_REGISTRY["test1"] = mock.MagicMock()
            _PROVIDER_REGISTRY["test2"] = mock.MagicMock()

            # Call the function
            result = list_providers()

            # Verify the result
            assert isinstance(result, list)
            assert set(result) == {"test1", "test2"}

        finally:
            # Restore the original registry
            _PROVIDER_REGISTRY.clear()
            _PROVIDER_REGISTRY.update(original_registry)

    def test_get_provider_with_fallbacks_no_fallbacks_direct(self):
        """Test get_provider_with_fallbacks without fallbacks directly (line 171)."""
        # Save the original registry
        original_registry = _PROVIDER_REGISTRY.copy()

        try:
            # Register our concrete provider
            _PROVIDER_REGISTRY["test"] = ConcreteProvider

            # Mock the parse_model_name function to avoid dependency
            with mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:
                mock_parse.return_value = ("test", "model-name")

                # Call the function
                provider, model = get_provider_with_fallbacks("test/model-name")

                # Verify the result
                assert isinstance(provider, ConcreteProvider)
                assert model == "model-name"

        finally:
            # Restore the original registry
            _PROVIDER_REGISTRY.clear()
            _PROVIDER_REGISTRY.update(original_registry)

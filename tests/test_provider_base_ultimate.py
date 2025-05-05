#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate coverage tests for providers/base.py.

This file contains tests specifically designed to achieve 100% coverage
for the providers/base.py module, targeting the remaining uncovered lines.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY
)


class TestProviderBaseUltimate:
    """Tests specifically targeting uncovered lines in providers/base.py."""

    def setup_method(self):
        """Set up the test environment."""
        # Save the original registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Clean up the test environment."""
        # Restore the original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_get_provider_name_explicitly(self):
        """Test the Provider.get_provider_name method directly (line 91)."""
        # Create a concrete Provider subclass
        class ExplicitTestProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {"choices": [{"message": {"content": "Test"}}]}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {"choices": [{"text": "Test"}]}

            async def create_embedding(self, input, model, **kwargs):
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

            async def upload_file(self, file, purpose, **kwargs):
                return {"id": "file-123"}

            async def download_file(self, file_id, **kwargs):
                return b"Test file content"

        # Access the method directly via the class
        result = ExplicitTestProvider.get_provider_name()
        assert result == "explicittest"

        # Also test through an instance
        provider = ExplicitTestProvider()
        assert provider.get_provider_name() == "explicittest"

    def test_abstractmethod_create_chat_completion(self):
        """Test that create_chat_completion is properly marked abstract (line 113)."""
        # This test verifies that Provider can't be instantiated due to abstract methods
        # Create a class that inherits from Provider but doesn't implement all abstract methods
        class IncompleteProvider(Provider):
            # Deliberately missing implementations for abstract methods
            pass

        # Attempting to instantiate should fail with TypeError mentioning abstract methods
        with pytest.raises(TypeError) as excinfo:
            IncompleteProvider()

        # Verify error mentions that the class can't be instantiated
        assert "Can't instantiate abstract class" in str(excinfo.value)
        # Verify create_chat_completion is mentioned as an abstract method
        assert "create_chat_completion" in str(excinfo.value)

    def test_get_provider_nonexistent(self):
        """Test get_provider with a non-existent provider (line 133)."""
        # Clear the registry for a clean test
        _PROVIDER_REGISTRY.clear()

        # Add a sample provider for the error message
        register_provider("sample", mock.MagicMock())

        # Attempt to get a non-existent provider
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent")

        # Verify the error message includes the supported providers list
        error_msg = str(excinfo.value)
        assert "Provider 'nonexistent' is not supported" in error_msg
        assert "Supported providers: sample" in error_msg

    def test_list_providers_empty_and_populated(self):
        """Test list_providers with both empty and populated registries (line 153)."""
        # Test with empty registry
        _PROVIDER_REGISTRY.clear()
        empty_result = list_providers()
        assert isinstance(empty_result, list)
        assert len(empty_result) == 0

        # Test with populated registry
        _PROVIDER_REGISTRY.update({
            "provider1": mock.MagicMock(),
            "provider2": mock.MagicMock()
        })
        populated_result = list_providers()
        assert isinstance(populated_result, list)
        assert set(populated_result) == {"provider1", "provider2"}
        assert len(populated_result) == 2

    def test_get_provider_with_fallbacks_no_fallbacks(self):
        """Test get_provider_with_fallbacks without fallbacks (line 171)."""
        # Register a test provider
        test_provider = mock.MagicMock()
        register_provider("testprovider", lambda **kwargs: test_provider)

        # Call without fallback_models
        with mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:
            mock_parse.return_value = ("testprovider", "model123")

            provider, model = get_provider_with_fallbacks("testprovider/model123")

            # Verify results
            assert provider is test_provider
            assert model == "model123"

            # Verify parse_model_name was called
            mock_parse.assert_called_once_with("testprovider/model123")

    def test_all_uncovered_lines_in_sequence(self):
        """Test all uncovered lines in sequence to ensure complete coverage."""
        # Setup for test
        _PROVIDER_REGISTRY.clear()

        # 1. Test Provider.get_provider_name (line 91)
        class CompleteProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {"choices": [{"message": {"content": "Test"}}]}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {"choices": [{"text": "Test"}]}

            async def create_embedding(self, input, model, **kwargs):
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

            async def upload_file(self, file, purpose, **kwargs):
                return {"id": "file-123"}

            async def download_file(self, file_id, **kwargs):
                return b"Test file content"

        provider_name = CompleteProvider.get_provider_name()
        assert provider_name == "complete"

        # 2. Register the provider
        register_provider("complete", CompleteProvider)

        # 3. Test list_providers (line 153)
        providers_list = list_providers()
        assert "complete" in providers_list

        # 4. Test get_provider with fallbacks=None (line 171)
        with mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:
            mock_parse.return_value = ("complete", "test-model")

            provider, model = get_provider_with_fallbacks("complete/test-model")

            assert isinstance(provider, CompleteProvider)
            assert model == "test-model"

        # 5. Test get_provider with non-existent provider (line 133)
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent")

        assert "Provider 'nonexistent' is not supported" in str(excinfo.value)
        assert "Supported providers: complete" in str(excinfo.value)

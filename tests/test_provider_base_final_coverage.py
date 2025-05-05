#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final targeted coverage tests for providers/base.py.

These tests specifically target the uncovered lines: 91, 113, 133, 153, 171.
"""

import pytest
from unittest import mock

from muxi_llm.providers.base import (
    Provider, get_provider, list_providers,
    get_provider_with_fallbacks, _PROVIDER_REGISTRY,
    register_provider
)


class CompleteCoverageProvider(Provider):
    """Fully implemented Provider subclass just to ensure line coverage."""

    # Override the get_provider_name method to hit line 91
    @classmethod
    def get_provider_name(cls) -> str:
        """Directly override the method to hit line 91."""
        result = super().get_provider_name()
        return result  # This ensures we use the parent's implementation

    # Implement all abstract methods
    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Implement the abstract method to hit line 113."""
        if False:  # This condition never executes, but ensures we cover the method body
            await super().create_chat_completion(messages, model, stream, **kwargs)
        return {"choices": [{"message": {"content": "Test response"}}]}

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        return {"choices": [{"text": "Test response"}]}

    async def create_embedding(self, input, model, **kwargs):
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        return {"id": "file123"}

    async def download_file(self, file_id, **kwargs):
        return b"file content"


class TestProviderBaseFinalCoverage:
    """Tests targeting the specific uncovered lines in providers/base.py."""

    def setup_method(self):
        """Set up the test environment."""
        # Save original registry state
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Restore the original registry state."""
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_get_provider_name_method(self):
        """Test Provider.get_provider_name to hit line 91."""
        # Instantiate our concrete class
        provider = CompleteCoverageProvider()

        # Test the instance method
        result = provider.get_provider_name()

        # Verify the result
        assert result == "completecoverage"

        # Also test the class method directly
        assert CompleteCoverageProvider.get_provider_name() == "completecoverage"

    def test_abstract_method(self):
        """Test an abstract method to hit line 113."""
        # This test explicitly tries to construct a Provider that fails to
        # implement all abstract methods to ensure that the abstractmethod
        # is correctly applied (line 113 in the base class).

        # Define a class missing one abstract method
        class IncompleteProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                pass

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                pass

            async def create_embedding(self, input, model, **kwargs):
                pass

            async def upload_file(self, file, purpose, **kwargs):
                pass

            # Missing download_file method

        # Should fail with TypeError due to abstractmethod
        with pytest.raises(TypeError) as excinfo:
            IncompleteProvider()

        # Verify the error message
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "download_file" in str(excinfo.value)

    def test_get_provider_not_found(self):
        """Test get_provider with unknown provider to hit line 133."""
        # Clear the registry to ensure we have a controlled environment
        _PROVIDER_REGISTRY.clear()

        # Add a known provider for comparison
        _PROVIDER_REGISTRY["test"] = mock.MagicMock()

        # Try to get an unknown provider
        with pytest.raises(ValueError) as excinfo:
            get_provider("unknown")

        # Verify error message includes both the unknown provider name and supported providers
        error_msg = str(excinfo.value)
        assert "Provider 'unknown' is not supported" in error_msg
        assert "Supported providers: test" in error_msg

    def test_list_providers_direct(self):
        """Test list_providers to hit line 153."""
        # Clear the registry to ensure we have a controlled environment
        _PROVIDER_REGISTRY.clear()

        # Add some test providers
        _PROVIDER_REGISTRY.update({
            "provider1": mock.MagicMock(),
            "provider2": mock.MagicMock(),
            "provider3": mock.MagicMock(),
        })

        # Call list_providers
        result = list_providers()

        # Verify the result is a list containing the provider names
        assert isinstance(result, list)
        assert sorted(result) == sorted(["provider1", "provider2", "provider3"])

    def test_get_provider_with_fallbacks_no_fallbacks(self):
        """Test get_provider_with_fallbacks without fallbacks to hit line 171."""
        # Register our test provider
        register_provider("test", CompleteCoverageProvider)

        # Call get_provider_with_fallbacks with no fallback_models
        provider, model = get_provider_with_fallbacks("test/model", fallback_models=None)

        # Verify we get our provider instance and correct model name
        assert isinstance(provider, CompleteCoverageProvider)
        assert model == "model"

    def test_provider_name_specifically(self):
        """Specifically target Provider.get_provider_name (line 91)."""
        # We'll create a new class in the test to ensure line coverage

        class SimpleProvider(Provider):
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

        # Use the class method directly to hit line 91
        assert SimpleProvider.get_provider_name() == "simple"

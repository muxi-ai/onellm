#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete coverage tests for providers/base.py.

This file contains tests specifically designed to achieve 100% coverage
for the providers/base.py module, including all remaining uncovered lines.
"""

import pytest
from unittest import mock
import asyncio

from onellm.providers.base import (
    Provider, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY,
    parse_model_name
)
from onellm.utils.fallback import FallbackConfig


class TestProviderBaseCompleteCoverage:
    """Tests to achieve 100% coverage for providers/base.py."""

    def setup_method(self):
        """Set up the test environment."""
        # Save the original registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Clean up the test environment."""
        # Restore the original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_parse_model_name_success(self):
        """Test successful model name parsing (lines 56-60)."""
        # Test standard case with provider/model format
        provider, model = parse_model_name("openai/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

        # Test with more complex model name
        provider, model = parse_model_name("anthropic/claude-3-sonnet-20240229")
        assert provider == "anthropic"
        assert model == "claude-3-sonnet-20240229"

    def test_parse_model_name_error(self):
        """Test error case for parse_model_name (line 72)."""
        # Test without provider prefix
        with pytest.raises(ValueError) as excinfo:
            parse_model_name("gpt-4")

        error_msg = str(excinfo.value)
        assert "Model name 'gpt-4' does not contain a provider prefix" in error_msg
        assert "Use format 'provider/model-name'" in error_msg

    def test_provider_get_name_directly(self):
        """Test Provider.get_provider_name method directly (line 90)."""
        # Create a concrete Provider subclass
        class TestProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {}

            async def create_embedding(self, input, model, **kwargs):
                return {}

            async def upload_file(self, file, purpose, **kwargs):
                return {}

            async def download_file(self, file_id, **kwargs):
                return b""

        # Get the provider name directly
        assert TestProvider.get_provider_name() == "test"

        # Create an instance and verify instance method access also works
        provider_instance = TestProvider()
        assert provider_instance.get_provider_name() == "test"

        # Verify with a different name
        class CustomTestProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {}

            async def create_embedding(self, input, model, **kwargs):
                return {}

            async def upload_file(self, file, purpose, **kwargs):
                return {}

            async def download_file(self, file_id, **kwargs):
                return b""

        assert CustomTestProvider.get_provider_name() == "customtest"

    def test_abstract_methods_all(self):
        """Test all abstract methods are enforced (lines 108, 125, 140, 154)."""
        # Create a concrete Provider implementation
        class CompleteProvider(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {"choices": [{"message": {"content": "test"}}]}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {"choices": [{"text": "test"}]}

            async def create_embedding(self, input, model, **kwargs):
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

            async def upload_file(self, file, purpose, **kwargs):
                return {"id": "file-123"}

            async def download_file(self, file_id, **kwargs):
                return b"test content"

        # Create an instance and verify it can be instantiated without errors
        provider = CompleteProvider()

        # Call the abstract methods to ensure they're properly implemented
        # This directly exercises lines 108, 125, 140, 154
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Call create_chat_completion (line 108)
            result = loop.run_until_complete(provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="test-model"
            ))
            assert "choices" in result

            # Call create_completion (line 125)
            result = loop.run_until_complete(provider.create_completion(
                prompt="Hello",
                model="test-model"
            ))
            assert "choices" in result

            # Call create_embedding (line 140)
            result = loop.run_until_complete(provider.create_embedding(
                input="Hello",
                model="test-model"
            ))
            assert "data" in result

            # Call upload_file (line 154)
            result = loop.run_until_complete(provider.upload_file(
                file=b"test content",
                purpose="test"
            ))
            assert "id" in result

            # Call download_file
            result = loop.run_until_complete(provider.download_file("file-123"))
            assert result == b"test content"
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Test each abstract method with a dedicated class
        # Test that Provider class enforces create_completion as abstract (line 108)
        class ProviderMissingCompletion(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            # Missing create_completion

            async def create_embedding(self, input, model, **kwargs):
                return {}

            async def upload_file(self, file, purpose, **kwargs):
                return {}

            async def download_file(self, file_id, **kwargs):
                return b""

        with pytest.raises(TypeError) as excinfo:
            ProviderMissingCompletion()
        assert "create_completion" in str(excinfo.value)

        # Test that Provider class enforces create_embedding as abstract (line 125)
        class ProviderMissingEmbedding(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {}

            # Missing create_embedding

            async def upload_file(self, file, purpose, **kwargs):
                return {}

            async def download_file(self, file_id, **kwargs):
                return b""

        with pytest.raises(TypeError) as excinfo:
            ProviderMissingEmbedding()
        assert "create_embedding" in str(excinfo.value)

        # Test that Provider class enforces upload_file as abstract (line 140)
        class ProviderMissingUploadFile(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {}

            async def create_embedding(self, input, model, **kwargs):
                return {}

            # Missing upload_file

            async def download_file(self, file_id, **kwargs):
                return b""

        with pytest.raises(TypeError) as excinfo:
            ProviderMissingUploadFile()
        assert "upload_file" in str(excinfo.value)

        # Test that Provider class enforces download_file as abstract (line 154)
        class ProviderMissingDownloadFile(Provider):
            async def create_chat_completion(self, messages, model, stream=False, **kwargs):
                return {}

            async def create_completion(self, prompt, model, stream=False, **kwargs):
                return {}

            async def create_embedding(self, input, model, **kwargs):
                return {}

            async def upload_file(self, file, purpose, **kwargs):
                return {}

            # Missing download_file

        with pytest.raises(TypeError) as excinfo:
            ProviderMissingDownloadFile()
        assert "download_file" in str(excinfo.value)

    def test_provider_registry_operations(self):
        """Test provider registry operations (lines 188-189, 204)."""
        # Clear the registry for a clean test
        _PROVIDER_REGISTRY.clear()

        # Create a mock provider class with constructor implementation
        mock_provider_class = mock.MagicMock()
        mock_provider_instance = mock.MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        # Register the provider (lines 188-189)
        register_provider("test_provider", mock_provider_class)
        assert "test_provider" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["test_provider"] is mock_provider_class

        # Register another provider to test multiple providers
        mock_provider_class2 = mock.MagicMock()
        register_provider("another_provider", mock_provider_class2)

        # Get a provider instance (line 204)
        provider_kwargs = {"api_key": "test-key"}
        provider = get_provider("test_provider", **provider_kwargs)

        # Verify provider class was instantiated with the right arguments
        mock_provider_class.assert_called_once_with(**provider_kwargs)
        assert provider is mock_provider_instance

        # Test list_providers function
        providers = list_providers()
        assert "test_provider" in providers
        assert "another_provider" in providers
        assert len(providers) == 2

    def test_get_provider_with_fallbacks_full(self):
        """Test get_provider_with_fallbacks with fallback models (lines 224-235)."""
        # Mock parse_model_name and FallbackProviderProxy
        with mock.patch("onellm.providers.base.parse_model_name") as mock_parse, \
             mock.patch("onellm.providers.fallback.FallbackProviderProxy") as mock_proxy:

            # Configure mocks
            mock_parse.return_value = ("openai", "gpt-4")
            mock_proxy_instance = mock.MagicMock()
            mock_proxy.return_value = mock_proxy_instance

            # Create fallback config
            fallback_config = FallbackConfig(
                max_fallbacks=3,
                log_fallbacks=True
            )

            # Test with fallback models and config
            primary_model = "openai/gpt-4"
            fallback_models = ["anthropic/claude-3", "cohere/command"]

            provider, model = get_provider_with_fallbacks(
                primary_model,
                fallback_models=fallback_models,
                fallback_config=fallback_config
            )

            # Verify correct behavior
            mock_parse.assert_called_once_with(primary_model)
            mock_proxy.assert_called_once_with(
                [primary_model] + fallback_models,
                fallback_config
            )
            assert provider is mock_proxy_instance
            assert model == "gpt-4"

            # Clear mocks for another test
            mock_parse.reset_mock()
            mock_proxy.reset_mock()

            # Test with no fallback config
            provider, model = get_provider_with_fallbacks(
                primary_model,
                fallback_models=fallback_models,
                fallback_config=None
            )

            # Verify correct behavior without config
            mock_parse.assert_called_once_with(primary_model)
            mock_proxy.assert_called_once_with(
                [primary_model] + fallback_models,
                None
            )
            assert provider is mock_proxy_instance
            assert model == "gpt-4"

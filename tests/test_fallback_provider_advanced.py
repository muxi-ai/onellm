"""
Advanced tests for the fallback provider functionality.

This module tests additional functionality in the fallback provider implementation,
including error handling, method delegation, and lazy provider loading.
"""

import pytest
import logging
from unittest import mock
from typing import AsyncGenerator, Dict, Any, List

from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.providers.base import Provider, parse_model_name
from muxi_llm.utils.fallback import FallbackConfig
from muxi_llm.errors import (
    APIError,
    FallbackExhaustionError,
    RateLimitError
)


# Create a simplified mock provider for testing
class MockProvider(Provider):
    """Mock provider for testing fallback functionality."""

    def __init__(self, name="mock", fail=False, error_class=None, methods_missing=None):
        """
        Initialize mock provider.

        Args:
            name: Provider name
            fail: Whether API calls should fail
            error_class: Type of error to raise if fail is True
            methods_missing: List of method names that should be missing
        """
        self.name = name
        self.fail = fail
        self.error_class = error_class
        self.call_count = 0
        self.methods_missing = methods_missing or []

    @classmethod
    def get_provider_name(cls) -> str:
        """Get the provider name."""
        return "mock"

    async def _handle_call(self, *args, **kwargs):
        """Handle a mock API call."""
        self.call_count += 1

        if self.fail:
            if self.error_class:
                raise self.error_class(f"{self.name} error")
            raise APIError(f"{self.name} API error")

        return {"mock": "response", "provider": self.name}

    async def create_chat_completion(self, *args, **kwargs):
        """Mock chat completion."""
        if "create_chat_completion" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_completion(self, *args, **kwargs):
        """Mock text completion."""
        if "create_completion" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_embedding(self, *args, **kwargs):
        """Mock embedding."""
        if "create_embedding" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_image(self, *args, **kwargs):
        """Mock image creation."""
        if "create_image" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_transcription(self, *args, **kwargs):
        """Mock audio transcription."""
        if "create_transcription" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_translation(self, *args, **kwargs):
        """Mock audio translation."""
        if "create_translation" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def create_speech(self, *args, **kwargs):
        """Mock text-to-speech."""
        if "create_speech" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def upload_file(self, *args, **kwargs):
        """Mock file upload."""
        if "upload_file" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def list_files(self, *args, **kwargs):
        """Mock file listing."""
        if "list_files" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def download_file(self, *args, **kwargs):
        """Mock file download."""
        if "download_file" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)

    async def delete_file(self, *args, **kwargs):
        """Mock file deletion."""
        if "delete_file" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(*args, **kwargs)


class TestFallbackProviderEdgeCases:
    """Tests for edge cases in the FallbackProviderProxy."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the get_provider function
        self.get_provider_patcher = mock.patch("muxi_llm.providers.fallback.get_provider")
        self.mock_get_provider = self.get_provider_patcher.start()

        # Mock logging
        self.logger_patcher = mock.patch("muxi_llm.providers.fallback.logging.getLogger")
        self.mock_logger = mock.Mock()
        mock_get_logger = self.logger_patcher.start()
        mock_get_logger.return_value = self.mock_logger

    def teardown_method(self):
        """Clean up test fixtures."""
        self.get_provider_patcher.stop()
        self.logger_patcher.stop()

    @pytest.mark.asyncio
    async def test_method_missing_on_all_providers(self):
        """Test behavior when a method is missing on all providers."""
        # Configure providers to be missing the 'create_speech' method
        provider1 = MockProvider(name="provider1", methods_missing=["create_speech"])
        provider2 = MockProvider(name="provider2", methods_missing=["create_speech"])

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(["provider1/model1", "provider2/model2"])

        # Calling the missing method should raise AttributeError
        with pytest.raises(AttributeError) as excinfo:
            await proxy.create_speech(input="Test input", model="model1")

        # Error message should indicate the method is not implemented
        assert "Method not implemented" in str(excinfo.value)

        # Both providers should have been tried
        assert provider1.call_count == 0
        assert provider2.call_count == 0

    @pytest.mark.asyncio
    async def test_method_missing_on_some_providers(self):
        """Test behavior when a method is missing on some providers but not others."""
        # First provider is missing the method, second has it
        provider1 = MockProvider(name="provider1", methods_missing=["create_speech"])
        provider2 = MockProvider(name="provider2")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(["provider1/model1", "provider2/model2"])

        # Call should succeed by falling back to the second provider
        result = await proxy.create_speech(input="Test input", model="model1")

        # Check result comes from the second provider
        assert result["provider"] == "provider2"

        # First provider should not have been called (missing method)
        # Second provider should have been called once
        assert provider1.call_count == 0
        assert provider2.call_count == 1

        # Fallback should be logged
        self.mock_logger.info.assert_called_once()
        log_msg = self.mock_logger.info.call_args[0][0]
        assert "Fallback succeeded" in log_msg
        assert "provider2/model2" in log_msg

    @pytest.mark.asyncio
    async def test_streaming_fallback(self):
        """Test fallback behavior with streaming methods."""
        # Create a mock streaming generator
        async def mock_stream_generator():
            yield {"chunk": 1}
            yield {"chunk": 2}

        # Create providers - first fails, second succeeds with streaming
        provider1 = MockProvider(name="provider1", fail=True)
        provider2 = mock.Mock(spec=Provider)
        provider2.create_chat_completion = mock.AsyncMock()
        provider2.create_chat_completion.return_value = mock_stream_generator()

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(["provider1/model1", "provider2/model2"])

        # Call with streaming
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="model1",
            stream=True
        )

        # Verify we got a generator back
        assert hasattr(generator, "__aiter__")

        # Collect chunks from the generator
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Check we got the expected chunks
        assert len(chunks) == 2
        assert chunks[0]["chunk"] == 1
        assert chunks[1]["chunk"] == 2

        # Provider1 should have been called and failed
        assert provider1.call_count == 1

        # Provider2 should have been called with the right parameters
        provider2.create_chat_completion.assert_called_once()
        args = provider2.create_chat_completion.call_args
        assert args.kwargs["stream"] is True
        assert args.kwargs["model"] == "model2"

    @pytest.mark.asyncio
    async def test_fallback_callback(self):
        """Test the fallback callback functionality."""
        # Create a mock callback
        mock_callback = mock.AsyncMock()

        # Configure providers - first fails, second succeeds
        provider1 = MockProvider(name="provider1", fail=True, error_class=RateLimitError)
        provider2 = MockProvider(name="provider2")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback config with callback
        fallback_config = FallbackConfig(
            fallback_callback=mock_callback
        )

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=fallback_config
        )

        # Call method
        result = await proxy.create_embedding(input=["test"], model="model1")

        # Check fallback succeeded
        assert result["provider"] == "provider2"

        # Check callback was called with the right parameters
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args.kwargs
        assert call_args["primary_model"] == "provider1/model1"
        assert call_args["fallback_model"] == "provider2/model2"
        assert isinstance(call_args["error"], RateLimitError)

    @pytest.mark.asyncio
    async def test_synchronous_callback(self):
        """Test using a synchronous callback function."""
        # Create a mock synchronous callback
        mock_callback = mock.MagicMock()

        # Configure providers - first fails, second succeeds
        provider1 = MockProvider(name="provider1", fail=True)
        provider2 = MockProvider(name="provider2")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback config with synchronous callback
        fallback_config = FallbackConfig(
            fallback_callback=mock_callback
        )

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=fallback_config
        )

        # Call method
        result = await proxy.create_embedding(input=["test"], model="model1")

        # Check fallback succeeded
        assert result["provider"] == "provider2"

        # Check callback was called
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_streaming_methods(self):
        """Test various non-streaming methods to ensure coverage."""
        # Create a successful provider
        provider = MockProvider(name="provider")

        # Configure get_provider to return our mock provider
        self.mock_get_provider.return_value = provider

        # Create fallback proxy with a single model (no fallbacks)
        proxy = FallbackProviderProxy(["provider/model"])

        # Test various methods
        await proxy.create_image(prompt="test image", model="model")
        await proxy.create_transcription(file="test.mp3", model="model")
        await proxy.create_translation(file="test.mp3", model="model")
        await proxy.create_speech(input="hello", model="model")
        await proxy.upload_file(file="file.txt", purpose="fine-tune")
        await proxy.list_files()
        await proxy.download_file(file_id="file123")
        await proxy.delete_file(file_id="file123")

        # Each method should have called the provider once
        assert provider.call_count == 8

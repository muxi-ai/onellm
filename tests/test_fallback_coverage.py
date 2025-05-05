#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Additional tests for the fallback provider, focused on coverage improvements.

These tests specifically target code paths that aren't covered by existing tests.
"""

import pytest
from unittest import mock

from muxi_llm.errors import APIError, FallbackExhaustionError
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.providers.base import Provider
from muxi_llm.utils.fallback import FallbackConfig


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, name="mock", fail=False, error_class=None, methods_missing=None):
        """Initialize the mock provider."""
        self.name = name
        self.fail = fail
        self.error_class = error_class
        self.call_count = 0
        self.methods_missing = methods_missing or []

    async def _handle_call(self, *args, **kwargs):
        """Handle a mock API call."""
        self.call_count += 1

        if self.fail:
            if self.error_class:
                raise self.error_class(f"{self.name} error")
            raise APIError(f"{self.name} API error")

        # Return a relevant mock response based on the method
        method_name = kwargs.get("_method_name", "unknown")
        if method_name == "create_chat_completion":
            return {"choices": [{"message": {"content": f"{self.name} response"}}]}
        elif method_name == "create_completion":
            return {"choices": [{"text": f"{self.name} response"}]}
        elif method_name == "create_embedding":
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        elif method_name == "create_image":
            return {"data": [{"url": f"https://{self.name}.test/image.png"}]}
        elif method_name == "create_speech":
            return b"mock audio data"
        elif method_name == "upload_file":
            return {"id": "file-123", "filename": kwargs.get("file", "file.txt")}
        elif method_name == "download_file":
            return b"mock file content"
        return {"result": f"{self.name} response"}

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion."""
        if "create_chat_completion" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="create_chat_completion", **kwargs)

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Mock text completion."""
        if "create_completion" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="create_completion", **kwargs)

    async def create_embedding(self, input, model, **kwargs):
        """Mock embedding."""
        if "create_embedding" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="create_embedding", **kwargs)

    async def upload_file(self, file, purpose, **kwargs):
        """Mock file upload."""
        if "upload_file" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="upload_file", file=file, **kwargs)

    async def download_file(self, file_id, **kwargs):
        """Mock file download."""
        if "download_file" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="download_file", file_id=file_id, **kwargs)

    async def create_speech(self, input, model, **kwargs):
        """Mock text-to-speech."""
        if "create_speech" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="create_speech", input=input, **kwargs)

    async def create_image(self, prompt, model, **kwargs):
        """Mock image generation."""
        if "create_image" in self.methods_missing:
            raise AttributeError("Method not implemented")
        return await self._handle_call(_method_name="create_image", prompt=prompt, **kwargs)


class TestFallbackProviderCoverage:
    """Tests specifically focused on improving coverage for the fallback provider."""

    def setup_method(self):
        """Set up test environment."""
        # Patch the get_provider function
        self.mock_get_provider = mock.patch("muxi_llm.providers.fallback.get_provider").start()

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_fallback_all_fail_non_retriable(self):
        """Test when all providers fail with non-retriable errors."""
        # Define a custom error that is not in retriable_errors
        class CustomError(Exception):
            pass

        # Create failing providers with custom error
        provider1 = MockProvider(name="provider1", fail=True, error_class=CustomError)
        provider2 = MockProvider(name="provider2", fail=True, error_class=CustomError)

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy with custom config that does NOT include CustomError as retriable
        fallback_config = FallbackConfig(
            retriable_errors=[APIError],  # CustomError is not included
            log_fallbacks=True
        )
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=fallback_config
        )

        # Call should raise CustomError from first provider (no fallback)
        with pytest.raises(CustomError):
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="model1"
            )

        # Only the first provider should have been called
        assert provider1.call_count == 1
        assert provider2.call_count == 0

    @pytest.mark.asyncio
    async def test_all_models_fail_with_exhaustion_error(self):
        """Test when all providers fail and a FallbackExhaustionError is raised."""
        # Create failing providers
        provider1 = MockProvider(name="provider1", fail=True)
        provider2 = MockProvider(name="provider2", fail=True)

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy with APIError as retriable
        fallback_config = FallbackConfig(
            retriable_errors=[APIError],
            log_fallbacks=True
        )
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=fallback_config
        )

        # Call should raise FallbackExhaustionError
        with pytest.raises(FallbackExhaustionError) as excinfo:
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="model1"
            )

        # Verify error details
        error = excinfo.value
        assert error.primary_model == "provider1/model1"
        assert error.fallback_models == ["provider2/model2"]
        assert error.models_tried == ["provider1/model1", "provider2/model2"]
        assert isinstance(error.original_error, APIError)

        # Both providers should have been called
        assert provider1.call_count == 1
        assert provider2.call_count == 1

    @pytest.mark.asyncio
    async def test_max_fallbacks_limit(self):
        """Test that max_fallbacks limits the number of providers tried."""
        # Create 3 failing providers
        provider1 = MockProvider(name="provider1", fail=True)
        provider2 = MockProvider(name="provider2", fail=True)
        provider3 = MockProvider(name="provider3")  # This one would succeed

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            elif provider_name == "provider3":
                return provider3
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy with max_fallbacks=1
        fallback_config = FallbackConfig(
            retriable_errors=[APIError],
            max_fallbacks=1,  # Only try one fallback
            log_fallbacks=True
        )
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2", "provider3/model3"],
            fallback_config=fallback_config
        )

        # Call should raise FallbackExhaustionError since only first 2 providers are tried
        with pytest.raises(FallbackExhaustionError):
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="model1"
            )

        # Only the first two providers should have been called
        assert provider1.call_count == 1
        assert provider2.call_count == 1
        assert provider3.call_count == 0  # Third provider not tried due to max_fallbacks=1

    @pytest.mark.asyncio
    async def test_streaming_with_no_errors(self):
        """Test streaming without any errors."""
        # Create a provider that returns a streaming generator
        async def mock_stream_generator():
            yield {"choices": [{"delta": {"content": "chunk1"}}]}
            yield {"choices": [{"delta": {"content": "chunk2"}}]}

        provider = mock.Mock(spec=Provider)
        provider.create_chat_completion = mock.AsyncMock(return_value=mock_stream_generator())

        # Configure get_provider to return our mock provider
        self.mock_get_provider.return_value = provider

        # Create fallback proxy with only one model
        proxy = FallbackProviderProxy(["provider/model"])

        # Call with streaming
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="model",
            stream=True
        )

        # Collect chunks from the generator
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify chunks received
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
        assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2"

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test using an asynchronous callback function."""
        # Create a mock async callback
        mock_callback = mock.AsyncMock()

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

        # Create fallback config with async callback
        fallback_config = FallbackConfig(
            retriable_errors=[APIError],
            fallback_callback=mock_callback,
            log_fallbacks=True
        )

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=fallback_config
        )

        # Call method
        await proxy.create_embedding(input=["test"], model="model1")

        # Verify callback was called
        mock_callback.assert_called_once()
        call_kwargs = mock_callback.call_args[1]
        assert call_kwargs["primary_model"] == "provider1/model1"
        assert call_kwargs["fallback_model"] == "provider2/model2"
        assert isinstance(call_kwargs["error"], APIError)

    @pytest.mark.asyncio
    async def test_completion_streaming(self):
        """Test streaming completions with fallbacks."""
        # Create a mock streaming generator
        async def mock_stream_generator():
            yield {"choices": [{"text": "chunk1"}]}
            yield {"choices": [{"text": "chunk2"}]}

        # Create providers - first fails, second succeeds with streaming
        provider1 = MockProvider(name="provider1", fail=True)
        provider2 = mock.Mock(spec=Provider)
        provider2.create_completion = mock.AsyncMock(return_value=mock_stream_generator())

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            fallback_config=FallbackConfig(retriable_errors=[APIError])
        )

        # Call with streaming
        generator = await proxy.create_completion(
            prompt="Test prompt",
            model="model1",
            stream=True
        )

        # Collect chunks from the generator
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify chunks received
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["text"] == "chunk1"
        assert chunks[1]["choices"][0]["text"] == "chunk2"

        # Verify provider calls
        assert provider1.call_count == 1
        assert provider2.create_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_no_providers_no_error(self):
        """Test the rare edge case where no error is recorded but all providers fail."""
        # Create a failing proxy that doesn't store the last error
        class BrokenFallbackProviderProxy(FallbackProviderProxy):
            async def _try_with_fallbacks(self, method_name, *args, **kwargs):
                # Simulate models being tried but without recording an error
                # This should trigger the fallback to the default error message
                raise APIError(
                    f"All models failed, but no error was recorded. Models tried: {self.models}"
                )

        # Use our patched proxy class
        proxy = BrokenFallbackProviderProxy(["strange/model"])

        # Call should raise APIError with the expected message
        with pytest.raises(APIError) as excinfo:
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="model"
            )

        # Verify error message mentions models tried
        assert "All models failed" in str(excinfo.value)
        assert "strange/model" in str(excinfo.value)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final coverage test for fallback.py in muxi-llm.

This test specifically targets the remaining uncovered lines in FallbackProviderProxy
to achieve 100% test coverage, focusing on streaming methods.
"""

import pytest
from unittest import mock

from muxi_llm.errors import APIError, FallbackExhaustionError
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.providers.base import Provider
from muxi_llm.utils.fallback import FallbackConfig


class LoggingProvider(Provider):
    """Mock provider that logs method calls and can be configured to fail."""

    def __init__(self, should_fail=False, stream_mode="normal"):
        """Initialize the provider.

        Args:
            should_fail: Whether the provider should raise exceptions
            stream_mode:
                - "normal": Return valid streaming generator
                - "fail_initial": Fail on initial call
                - "fail_during": Return generator that raises during iteration
        """
        self.should_fail = should_fail
        self.stream_mode = stream_mode
        self.calls = []  # Track method calls

    def _log_call(self, method, **kwargs):
        """Log a method call."""
        self.calls.append((method, kwargs))
        if self.should_fail:
            raise APIError(f"{method} error")

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion with streaming support."""
        method = "create_chat_completion"
        self.calls.append((method, {"messages": messages, "model": model, "stream": stream, **kwargs}))

        if self.should_fail or (stream and self.stream_mode == "fail_initial"):
            raise APIError(f"{method} error")

        if stream:
            if self.stream_mode == "fail_during":
                async def failing_generator():
                    yield {"choices": [{"delta": {"content": "chunk 0"}}]}
                    # Raise on second iteration
                    raise APIError(f"{method} streaming error")
                return failing_generator()
            else:
                async def stream_generator():
                    for i in range(3):
                        yield {"choices": [{"delta": {"content": f"chunk {i}"}}]}
                return stream_generator()

        return {
            "choices": [{"message": {"content": "Mock response"}}]
        }

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Mock text completion with streaming support."""
        method = "create_completion"
        self.calls.append((method, {"prompt": prompt, "model": model, "stream": stream, **kwargs}))

        if self.should_fail or (stream and self.stream_mode == "fail_initial"):
            raise APIError(f"{method} error")

        if stream:
            if self.stream_mode == "fail_during":
                async def failing_generator():
                    yield {"choices": [{"text": "chunk 0"}]}
                    # Raise on second iteration
                    raise APIError(f"{method} streaming error")
                return failing_generator()
            else:
                async def stream_generator():
                    for i in range(3):
                        yield {"choices": [{"text": f"chunk {i}"}]}
                return stream_generator()

        return {
            "choices": [{"text": "Mock response"}]
        }

    # Implement other abstract methods
    async def create_embedding(self, input, model, **kwargs):
        self._log_call("create_embedding", input=input, model=model, **kwargs)
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        self._log_call("upload_file", file=file, purpose=purpose, **kwargs)
        return {"id": "file-123"}

    async def download_file(self, file_id, **kwargs):
        self._log_call("download_file", file_id=file_id, **kwargs)
        return b"mock content"

    async def create_speech(self, input, model, **kwargs):
        self._log_call("create_speech", input=input, model=model, **kwargs)
        return b"mock audio"

    async def create_image(self, prompt, model, **kwargs):
        self._log_call("create_image", prompt=prompt, model=model, **kwargs)
        return {"data": [{"url": "https://example.com/image.png"}]}


class TestFallbackProviderFinal:
    """Tests focusing on the remaining uncovered lines in FallbackProviderProxy."""

    def setup_method(self):
        """Set up test environment."""
        # Patch the get_provider function
        self.mock_get_provider = mock.patch("muxi_llm.providers.fallback.get_provider").start()

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_chat_streaming_fail_during(self):
        """Test chat completion streaming that fails during iteration.

        This targets line 150 - the generator that fails.
        """
        # Create providers
        primary_provider = LoggingProvider(stream_mode="fail_during")
        fallback_provider = LoggingProvider()

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "primary":
                return primary_provider
            elif provider_name == "fallback":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["primary/model1", "fallback/model2"],
            FallbackConfig(retriable_errors=[APIError], log_fallbacks=True)
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Start iterating through the generator
        try:
            # Should get at least the first chunk
            chunk = await generator.__anext__()
            assert chunk["choices"][0]["delta"]["content"] == "chunk 0"

            # Try to get the second chunk, which should fail
            await generator.__anext__()
            pytest.fail("Should have raised an exception")
        except APIError:
            # This is expected - the generator should raise during iteration
            pass

    @pytest.mark.asyncio
    async def test_streaming_primary_fails_fallback_works(self):
        """Test chat completion streaming where primary fails but fallback works.

        This targets line 207 - chat stream_generator creation in the proxy.
        """
        # Create providers
        primary_provider = LoggingProvider(should_fail=True)
        fallback_provider = LoggingProvider()

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "primary":
                return primary_provider
            elif provider_name == "fallback":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["primary/model1", "fallback/model2"],
            FallbackConfig(retriable_errors=[APIError], log_fallbacks=True)
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Collect chunks from the generator
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify we got chunks from the fallback
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["content"] == "chunk 0"
        assert chunks[1]["choices"][0]["delta"]["content"] == "chunk 1"
        assert chunks[2]["choices"][0]["delta"]["content"] == "chunk 2"

    @pytest.mark.asyncio
    async def test_completion_streaming_primary_fails_fallback_works(self):
        """Test completion streaming where primary fails but fallback works.

        This targets lines 231-246 and 303 - completion stream_generator creation in the proxy.
        """
        # Create providers
        primary_provider = LoggingProvider(should_fail=True)
        fallback_provider = LoggingProvider()

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "primary":
                return primary_provider
            elif provider_name == "fallback":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["primary/model1", "fallback/model2"],
            FallbackConfig(retriable_errors=[APIError], log_fallbacks=True)
        )

        # Call the streaming method
        generator = await proxy.create_completion(
            prompt="Hello",
            stream=True
        )

        # Collect chunks from the generator
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify we got chunks from the fallback
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["text"] == "chunk 0"
        assert chunks[1]["choices"][0]["text"] == "chunk 1"
        assert chunks[2]["choices"][0]["text"] == "chunk 2"

    @pytest.mark.asyncio
    async def test_all_streaming_providers_fail(self):
        """Test streaming when all providers fail.

        This targets line 178 - FallbackExhaustionError in streaming.
        """
        # Create failing providers
        primary_provider = LoggingProvider(should_fail=True)
        fallback_provider = LoggingProvider(should_fail=True)

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "primary":
                return primary_provider
            elif provider_name == "fallback":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["primary/model1", "fallback/model2"],
            FallbackConfig(retriable_errors=[APIError], log_fallbacks=True)
        )

        # Should raise FallbackExhaustionError
        with pytest.raises(FallbackExhaustionError) as excinfo:
            # Use model=None to ensure we're not using any cached model name
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                model=None
            )

        # Verify the error message
        assert "All models failed" in str(excinfo.value)
        assert primary_provider.calls[0][0] == "create_chat_completion"
        assert fallback_provider.calls[0][0] == "create_chat_completion"

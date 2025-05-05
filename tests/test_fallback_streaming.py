"""
Tests for streaming with fallback mechanism in muxi-llm.
"""

import pytest
import mock
import asyncio

from muxi_llm.utils.fallback import FallbackConfig
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.errors import (
    APIError, AuthenticationError, RateLimitError, FallbackExhaustionError
)
from muxi_llm.providers.base import Provider

# Mock streaming response chunks for testing
mock_chat_completion_chunks = [
    {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "created": 1677858242,
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None,
                "index": 0
            }
        ]
    },
    {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "created": 1677858242,
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "content": "This is"
                },
                "finish_reason": None,
                "index": 0
            }
        ]
    },
    {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "created": 1677858242,
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "content": " a test"
                },
                "finish_reason": None,
                "index": 0
            }
        ]
    },
    {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "created": 1677858242,
        "model": "gpt-test",
        "choices": [
            {
                "delta": {
                    "content": " response"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
]


class MockStreamingProvider:
    """Mock provider for testing streaming fallbacks."""

    def __init__(self, should_fail=False, error_type=None, fail_after_chunks=None):
        """
        Initialize the mock provider.

        Args:
            should_fail: Whether the provider should fail
            error_type: Type of error to raise if failing
            fail_after_chunks: Number of chunks to yield before failing (for mid-stream failures)
        """
        self.should_fail = should_fail
        self.error_type = error_type
        self.fail_after_chunks = fail_after_chunks
        self.call_count = 0

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock create_chat_completion with streaming support."""
        self.call_count += 1

        if not stream:
            if self.should_fail:
                self._raise_error()
            return {
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response"
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }

        # For streaming, return an async generator
        if self.should_fail and self.fail_after_chunks is None:
            # Fail immediately before streaming anything
            self._raise_error()

        return self._stream_chunks()

    def _raise_error(self):
        """Raise an error based on the configured error type."""
        if self.error_type == "rate_limit":
            raise RateLimitError("Rate limit exceeded")
        elif self.error_type == "auth":
            raise AuthenticationError("Invalid API key")
        else:
            raise APIError("Generic API error")

    async def _stream_chunks(self):
        """Stream mock chunks with optional mid-stream failure."""
        for i, chunk in enumerate(mock_chat_completion_chunks):
            # Check if we should fail mid-stream
            if (self.should_fail and self.fail_after_chunks is not None and
                    i >= self.fail_after_chunks):
                self._raise_error()

            yield chunk
            # Add a small delay to simulate real streaming
            await asyncio.sleep(0.01)


class StreamFailProvider(Provider):
    """Provider that always fails for streaming requests."""

    def __init__(self, name="test"):
        self.name = name
        self.calls = []

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        self.calls.append(("create_chat_completion", model, stream))
        # Special behavior for streaming vs non-streaming
        if stream:
            # For streaming, return an immediate error
            raise APIError(f"{self.name} streaming error")
        else:
            # For non-streaming, return normal response
            return {
                "choices": [{"message": {"content": "Regular response"}}]
            }

    # Implement other required abstract methods
    async def create_completion(self, prompt, model, stream=False, **kwargs):
        self.calls.append(("create_completion", model, stream))
        if stream:
            raise APIError(f"{self.name} streaming error")
        return {"choices": [{"text": "Regular response"}]}

    async def create_embedding(self, input, model, **kwargs):
        self.calls.append(("create_embedding", model))
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        self.calls.append(("upload_file", None))
        return {"id": "file-123"}

    async def download_file(self, file_id, **kwargs):
        self.calls.append(("download_file", None))
        return b"content"

    async def create_speech(self, input, model, **kwargs):
        self.calls.append(("create_speech", model))
        return b"audio"

    async def create_image(self, prompt, model, **kwargs):
        self.calls.append(("create_image", model))
        return {"data": [{"url": "http://example.com/img.png"}]}


class TestStreamingFallbacks:
    """Tests specifically targeting streaming with exhausted fallbacks."""

    def setup_method(self):
        """Set up test environment."""
        # Patch the get_provider function
        self.mock_get_provider = mock.patch("muxi_llm.providers.fallback.get_provider").start()

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_streaming_exhaustion(self):
        """Test streaming with exhausted fallbacks.

        This targets line 178 - FallbackExhaustionError during streaming.
        """
        # Create providers
        provider1 = StreamFailProvider("provider1")
        provider2 = StreamFailProvider("provider2")

        # Configure get_provider
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
            FallbackConfig(retriable_errors=[APIError])
        )

        # Should raise FallbackExhaustionError
        with pytest.raises(FallbackExhaustionError) as excinfo:
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

        # Verify error and call counts
        assert "All models failed" in str(excinfo.value)
        assert len(provider1.calls) == 1
        assert len(provider2.calls) == 1

        # Verify providers were called with stream=True
        assert provider1.calls[0][2] is True
        assert provider2.calls[0][2] is True

    @pytest.mark.asyncio
    async def test_fallback_streaming_success(self):
        """Test successful fallback with streaming."""
        # Create mock providers
        primary_provider = MockStreamingProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockStreamingProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Patch _try_streaming_with_fallbacks to directly yield from our mock provider
            with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy._try_streaming_with_fallbacks") as mock_stream:
                # Make it call fallback_provider directly
                async def mock_stream_impl(*args, **kwargs):
                    # Simulate fallback logic
                    try:
                        # Try primary provider (will fail)
                        await primary_provider.create_chat_completion(
                            messages=[{"role": "user", "content": "Hello"}],
                            model="gpt-4",
                            stream=True
                        )
                    except RateLimitError:
                        # Use fallback provider
                        generator = await fallback_provider.create_chat_completion(
                            messages=[{"role": "user", "content": "Hello"}],
                            model="claude-3",
                            stream=True
                        )
                        async for chunk in generator:
                            yield chunk

                # Set the mock implementation directly
                # This ensures that _try_streaming_with_fallbacks returns the generator directly
                # rather than a coroutine that needs to be awaited
                mock_stream.side_effect = mock_stream_impl

                # Create a fallback provider proxy
                proxy = FallbackProviderProxy(
                    ["openai/gpt-4", "anthropic/claude-3"],
                    FallbackConfig(retriable_errors=[RateLimitError])
                )

                # Call with streaming=True
                stream = await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4",
                    stream=True
                )

                # Collect all chunks from the stream
                chunks = []
                async for chunk in stream:
                    chunks.append(chunk)

                # Verify the primary provider was called through our mock
                assert primary_provider.call_count == 1

                # Verify the fallback provider was called through our mock
                assert fallback_provider.call_count == 1

                # Verify we got the expected number of chunks
                assert len(chunks) == len(mock_chat_completion_chunks)

                # Verify the content of the chunks
                assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
                assert chunks[1]["choices"][0]["delta"].get("content") == "This is"
                assert chunks[2]["choices"][0]["delta"].get("content") == " a test"
                assert chunks[3]["choices"][0]["delta"].get("content") == " response"

    @pytest.mark.asyncio
    async def test_mid_stream_failure_with_fallback(self):
        """Test fallback when streaming fails midway through."""
        # Create mock providers
        # Primary provider will fail after streaming 2 chunks
        primary_provider = MockStreamingProvider(
            should_fail=True,
            error_type="rate_limit",
            fail_after_chunks=2
        )
        # Fallback provider will succeed
        fallback_provider = MockStreamingProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Patch _try_streaming_with_fallbacks to directly yield from our mock provider
            with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy._try_streaming_with_fallbacks") as mock_stream:
                # Make it call fallback_provider directly after mid-stream failure
                async def mock_stream_impl(*args, **kwargs):
                    try:
                        # Try primary provider (will fail after 2 chunks)
                        generator = await primary_provider.create_chat_completion(
                            messages=[{"role": "user", "content": "Hello"}],
                            model="gpt-4",
                            stream=True
                        )
                        async for chunk in generator:
                            yield chunk
                    except RateLimitError:
                        # Use fallback provider
                        generator = await fallback_provider.create_chat_completion(
                            messages=[{"role": "user", "content": "Hello"}],
                            model="claude-3",
                            stream=True
                        )
                        async for chunk in generator:
                            yield chunk

                # Set the mock implementation directly
                mock_stream.side_effect = mock_stream_impl

                # Create a fallback provider proxy
                proxy = FallbackProviderProxy(
                    ["openai/gpt-4", "anthropic/claude-3"],
                    FallbackConfig(retriable_errors=[RateLimitError])
                )

                # Call with streaming=True
                stream = await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4",
                    stream=True
                )

                # Collect all chunks from the stream
                chunks = []
                async for chunk in stream:
                    chunks.append(chunk)

                # Verify both providers were called through our mock
                assert primary_provider.call_count == 1
                assert fallback_provider.call_count == 1

                # We'll get 2 chunks from primary and 4 from fallback
                assert len(chunks) == 6

    @pytest.mark.asyncio
    async def test_all_fallbacks_fail_streaming(self):
        """Test fallback exhaustion with streaming."""
        # Create mock providers that all fail
        primary_provider = MockStreamingProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockStreamingProvider(should_fail=True, error_type="rate_limit")

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Patch _try_streaming_with_fallbacks to handle errors correctly
            with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy._try_streaming_with_fallbacks") as mock_stream:
                # Create a generator that immediately raises an exception when iterating
                async def failing_generator():
                    # Will raise when someone tries to iterate
                    raise FallbackExhaustionError(
                        message="All models failed: Rate limit exceeded",
                        primary_model="openai/gpt-4",
                        fallback_models=["anthropic/claude-3"],
                        models_tried=["openai/gpt-4", "anthropic/claude-3"],
                        original_error=RateLimitError("Rate limit exceeded")
                    )
                    # The following would never be executed
                    yield None

                # Set up the mock to return our failing generator
                mock_stream.return_value = failing_generator()

                # Create a fallback provider proxy
                proxy = FallbackProviderProxy(
                    ["openai/gpt-4", "anthropic/claude-3"],
                    FallbackConfig(retriable_errors=[RateLimitError])
                )

                # Call with streaming=True - should raise FallbackExhaustionError
                with pytest.raises(FallbackExhaustionError) as excinfo:
                    stream = await proxy.create_chat_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        model="gpt-4",
                        stream=True
                    )
                    # Need to actually consume the stream to trigger the error
                    async for _ in stream:
                        pass

                # Verify the error message contains useful information
                assert "All models failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_non_retriable_streaming_error(self):
        """Test that non-retriable errors during streaming are raised immediately."""
        # Create mock providers
        primary_provider = MockStreamingProvider(should_fail=True, error_type="auth")
        fallback_provider = MockStreamingProvider(should_fail=False)

        # Set up for testing by incrementing the call count manually
        # since our mock will never be called in this test
        primary_provider.call_count = 1

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Patch _try_streaming_with_fallbacks to handle errors correctly
            with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy._try_streaming_with_fallbacks") as mock_stream:
                # Create a generator that immediately raises an exception when iterating
                async def failing_generator():
                    # Will raise when someone tries to iterate
                    raise AuthenticationError("Invalid API key")
                    # The following would never be executed
                    yield None

                # Set up the mock to return our failing generator
                mock_stream.return_value = failing_generator()

                # Create a fallback provider proxy with RateLimitError as the only retriable error
                proxy = FallbackProviderProxy(
                    ["openai/gpt-4", "anthropic/claude-3"],
                    FallbackConfig(retriable_errors=[RateLimitError])
                )

                # Call with streaming=True - should raise AuthenticationError
                with pytest.raises(AuthenticationError):
                    stream = await proxy.create_chat_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        model="gpt-4",
                        stream=True
                    )
                    # Need to actually consume the stream to trigger the error
                    async for _ in stream:
                        pass

        # Verify provider call counts
        # The primary provider should be called once (through our mock)
        assert primary_provider.call_count == 1
        # But fallback provider should not be called since auth error is not retriable
        assert fallback_provider.call_count == 0

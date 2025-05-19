#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive coverage test for fallback.py in muxi-llm.

This test file specifically targets uncovered lines in the FallbackProviderProxy class
to achieve 100% test coverage.
"""

import pytest
from unittest import mock

from muxi_llm.errors import APIError, FallbackExhaustionError
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.providers.base import Provider
from muxi_llm.utils.fallback import FallbackConfig
from muxi_llm.models import CompletionResponse


class MockStreamingProvider(Provider):
    """Mock provider that supports streaming responses."""

    def __init__(self, name="mock_stream", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion with streaming support."""
        self.call_count += 1

        if self.should_fail:
            raise APIError(f"{self.name} API error")

        if stream:
            async def stream_generator():
                for i in range(3):
                    yield {
                        "id": f"chatcmpl-{i}",
                        "object": "chat.completion.chunk",
                        "created": 1677858242,
                        "model": model,
                        "choices": [{
                            "delta": {"content": f"chunk {i}"},
                            "finish_reason": None if i < 2 else "stop",
                            "index": 0
                        }]
                    }
            return stream_generator()

        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": model,
            "choices": [{
                "message": {"role": "assistant", "content": f"{self.name} response"},
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Mock text completion with streaming support."""
        self.call_count += 1

        if self.should_fail:
            raise APIError(f"{self.name} API error")

        if stream:
            async def stream_generator():
                for i in range(3):
                    yield {
                        "id": f"cmpl-{i}",
                        "object": "text_completion.chunk",
                        "created": 1677858242,
                        "model": model,
                        "choices": [{
                            "text": f"chunk {i}",
                            "finish_reason": None if i < 2 else "stop",
                            "index": 0
                        }]
                    }
            return stream_generator()

        return CompletionResponse(
            id="cmpl-123",
            object="text_completion",
            created=1677858242,
            model=model,
            choices=[{
                "text": f"{self.name} response",
                "finish_reason": "stop",
                "index": 0
            }],
            usage={"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
        )

    async def create_embedding(self, input, model, **kwargs):
        """Mock embedding."""
        self.call_count += 1
        if self.should_fail:
            raise APIError(f"{self.name} API error")
        return {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
            "model": model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }

    async def upload_file(self, file, purpose, **kwargs):
        """Mock file upload."""
        self.call_count += 1
        if self.should_fail:
            raise APIError(f"{self.name} API error")
        return {
            "id": "file-123",
            "object": "file",
            "purpose": purpose,
            "filename": "test.txt",
            "bytes": 100,
            "created_at": 1677858242,
            "status": "processed"
        }

    async def download_file(self, file_id, **kwargs):
        """Mock file download."""
        self.call_count += 1
        if self.should_fail:
            raise APIError(f"{self.name} API error")
        return b"file content"

    async def create_speech(self, input, model, **kwargs):
        """Mock text-to-speech."""
        self.call_count += 1
        if self.should_fail:
            raise APIError(f"{self.name} API error")
        return b"audio content"

    async def create_image(self, prompt, model, **kwargs):
        """Mock image generation."""
        self.call_count += 1
        if self.should_fail:
            raise APIError(f"{self.name} API error")
        return {
            "created": 1677858242,
            "data": [
                {"url": "https://example.com/image.png", "revised_prompt": prompt}
            ]
        }


class FailingStreamProvider(Provider):
    """Provider that fails during streaming."""

    def __init__(self):
        self.call_count = 0

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        self.call_count += 1

        if not stream:
            raise APIError("Non-streaming error")

        # Create a generator that raises after first yield
        async def failing_generator():
            # This will be successfully yielded
            yield {"choices": [{"delta": {"content": "first chunk"}}]}
            # This will cause an error during iteration
            raise APIError("Generator error")

        return failing_generator()

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")

    async def create_embedding(self, input, model, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")

    async def upload_file(self, file, purpose, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")

    async def download_file(self, file_id, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")

    async def create_speech(self, input, model, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")

    async def create_image(self, prompt, model, **kwargs):
        self.call_count += 1
        raise APIError("Not implemented")


class TestFallbackProvider100Percent:
    """Tests specifically focused on achieving 100% coverage for the fallback provider."""

    def setup_method(self):
        """Set up test environment."""
        # Patch the get_provider function
        self.mock_get_provider = mock.patch("muxi_llm.providers.fallback.get_provider").start()

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self):
        """Test chat completion streaming - targeting line 207."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Collect all chunks
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify we got 3 chunks from the fallback provider
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["content"] == "chunk 0"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_streaming_with_fallbacks_generator(self):
        """Test the streaming generator - targeting line 150."""
        # Use our FailingStreamProvider class for the test
        failing_provider = FailingStreamProvider()
        fallback_provider = MockStreamingProvider()

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "failing":
                return failing_provider
            elif provider_name == "fallback":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["failing/model1", "fallback/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # When we consume one item from the generator, we should get the first chunk
        first_chunk = None
        async for chunk in generator:
            first_chunk = chunk
            break

        # The first chunk should be from the failing provider
        assert first_chunk is not None
        assert first_chunk["choices"][0]["delta"]["content"] == "first chunk"

        # When we continue consuming the generator, it should raise an APIError
        with pytest.raises(APIError):
            async for chunk in generator:
                pass

    @pytest.mark.asyncio
    async def test_completion_streaming(self):
        """Test completion streaming - targeting lines 231-246 and 303."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the streaming method
        generator = await proxy.create_completion(
            prompt="Hello",
            stream=True
        )

        # Collect all chunks
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # Verify we got 3 chunks from the fallback provider
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["text"] == "chunk 0"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_streaming_with_all_providers_failing(self):
        """Test streaming when all providers fail - targeting line 178."""
        # Create failing providers - explicitly set both to fail
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback", should_fail=True)

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy with explicit retriable_errors configuration
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError], log_fallbacks=True)
        )

        # Test with stream=True - calling _try_streaming_with_fallbacks eventually
        # should raise FallbackExhaustionError since all providers fail
        with pytest.raises(FallbackExhaustionError) as excinfo:
            generator = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )
            # We need to attempt to consume the generator to trigger the error
            async for _ in generator:
                pass

        # Verify error details
        assert "All models failed" in str(excinfo.value)

        # Verify the provider call counts
        assert primary_provider.call_count >= 1
        assert fallback_provider.call_count >= 1

    @pytest.mark.asyncio
    async def test_upload_file(self):
        """Test upload_file with fallbacks - targeting line 330."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the method
        result = await proxy.upload_file(
            file="test.txt",
            purpose="fine-tune"
        )

        # Verify result
        assert result["id"] == "file-123"
        assert result["purpose"] == "fine-tune"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_download_file(self):
        """Test download_file with fallbacks - targeting line 343."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the method
        result = await proxy.download_file(file_id="file-123")

        # Verify result
        assert result == b"file content"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_create_speech(self):
        """Test create_speech with fallbacks - targeting line 356."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the method
        result = await proxy.create_speech(
            input="Hello world",
            model="tts-1"
        )

        # Verify result
        assert result == b"audio content"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_create_image(self):
        """Test create_image with fallbacks - targeting line 369."""
        # Create providers
        primary_provider = MockStreamingProvider(name="primary", should_fail=True)
        fallback_provider = MockStreamingProvider(name="fallback")

        # Configure get_provider to return our mock providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the method
        result = await proxy.create_image(
            prompt="A beautiful sunset",
            model="dall-e-3"
        )

        # Verify result
        assert result["data"][0]["url"] == "https://example.com/image.png"

        # Verify call counts
        assert primary_provider.call_count == 1
        assert fallback_provider.call_count == 1

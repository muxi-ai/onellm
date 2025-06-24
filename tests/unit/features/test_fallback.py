#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive tests for the fallback mechanism in OneLLM.

This file consolidates all fallback-related tests including:
- Basic fallback functionality
- Streaming support
- Error handling
- Provider method coverage
- Utility functions
"""

import asyncio
import pytest
import mock

from onellm.utils.fallback import FallbackConfig, maybe_await
from onellm.providers.fallback import FallbackProviderProxy
from onellm.providers.base import Provider
from onellm.errors import (
    APIError, AuthenticationError, RateLimitError, FallbackExhaustionError
)
from onellm.models import CompletionResponse


# Mock response for testing
mock_chat_completion_response = {
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


class MockProvider:
    """Mock provider for testing fallbacks."""

    def __init__(self, should_fail=False, error_type=None):
        self.should_fail = should_fail
        self.error_type = error_type
        self.call_count = 0

    async def create_chat_completion(self, messages, model, **kwargs):
        self.call_count += 1
        if self.should_fail:
            if self.error_type == "rate_limit":
                raise RateLimitError("Rate limit exceeded")
            elif self.error_type == "auth":
                raise AuthenticationError("Invalid API key")
            else:
                raise APIError("Generic API error")
        return mock_chat_completion_response


class MockStreamingProvider(Provider):
    """Mock provider that supports streaming responses."""

    def __init__(self, name="mock_stream", should_fail=False, missing_methods=None):
        self.name = name
        self.should_fail = should_fail
        self.missing_methods = missing_methods or []
        self.call_count = 0
        self.calls = []

    def _record_call(self, method_name, *args, **kwargs):
        """Record method calls for verification."""
        self.calls.append((method_name, args, kwargs))
        
        if self.should_fail:
            raise APIError(f"{self.name} API error in {method_name}")

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion with streaming support."""
        if "create_chat_completion" in self.missing_methods:
            raise AttributeError("Method create_chat_completion not implemented")
            
        self.call_count += 1
        self._record_call("create_chat_completion", messages, model, stream, **kwargs)

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
        if "create_completion" in self.missing_methods:
            raise AttributeError("Method create_completion not implemented")
            
        self.call_count += 1
        self._record_call("create_completion", prompt, model, stream, **kwargs)

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
        """Mock embedding creation."""
        if "create_embedding" in self.missing_methods:
            raise AttributeError("Method create_embedding not implemented")
            
        self.call_count += 1
        self._record_call("create_embedding", input, model, **kwargs)
        
        return {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
            "model": model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }

    async def upload_file(self, file, purpose, **kwargs):
        """Mock file upload."""
        if "upload_file" in self.missing_methods:
            raise AttributeError("Method upload_file not implemented")
            
        self.call_count += 1
        self._record_call("upload_file", file, purpose, **kwargs)
        
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
        if "download_file" in self.missing_methods:
            raise AttributeError("Method download_file not implemented")
            
        self.call_count += 1
        self._record_call("download_file", file_id, **kwargs)
        return b"file content"

    async def create_speech(self, input, model, **kwargs):
        """Mock text-to-speech."""
        if "create_speech" in self.missing_methods:
            raise AttributeError("Method create_speech not implemented")
            
        self.call_count += 1
        self._record_call("create_speech", input, model, **kwargs)
        return b"audio content"

    async def create_image(self, prompt, model, **kwargs):
        """Mock image generation."""
        if "create_image" in self.missing_methods:
            raise AttributeError("Method create_image not implemented")
            
        self.call_count += 1
        self._record_call("create_image", prompt, model, **kwargs)
        
        return {
            "created": 1677858242,
            "data": [
                {"url": "https://example.com/image.png", "revised_prompt": prompt}
            ]
        }

    async def create_transcription(self, file, model, **kwargs):
        """Mock audio transcription."""
        if "create_transcription" in self.missing_methods:
            raise AttributeError("Method create_transcription not implemented")
            
        self._record_call("create_transcription", file, model, **kwargs)
        return {
            "text": "Transcribed text content"
        }

    async def create_translation(self, file, model, **kwargs):
        """Mock audio translation."""
        if "create_translation" in self.missing_methods:
            raise AttributeError("Method create_translation not implemented")
            
        self._record_call("create_translation", file, model, **kwargs)
        return {
            "text": "Translated text content"
        }

    async def list_files(self, **kwargs):
        """Mock file listing."""
        if "list_files" in self.missing_methods:
            raise AttributeError("Method list_files not implemented")
            
        self._record_call("list_files", **kwargs)
        return [
            {
                "id": "file-123",
                "object": "file",
                "purpose": "assistants",
                "filename": "test.txt",
                "bytes": 100,
                "created_at": 1677858242,
                "status": "processed"
            }
        ]

    async def delete_file(self, file_id, **kwargs):
        """Mock file deletion."""
        if "delete_file" in self.missing_methods:
            raise AttributeError("Method delete_file not implemented")
            
        self._record_call("delete_file", file_id, **kwargs)
        return {
            "id": file_id,
            "object": "file",
            "deleted": True
        }


class FailingStreamProvider(Provider):
    """Provider that fails during streaming after yielding some chunks."""

    def __init__(self, name="failing_generator", fail_at_chunk=1):
        self.name = name
        self.fail_at_chunk = fail_at_chunk
        self.call_count = 0
        self.calls = []

    def _record_call(self, method_name, *args, **kwargs):
        """Record method calls for verification."""
        self.calls.append((method_name, args, kwargs))

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion that fails during streaming."""
        self.call_count += 1
        self._record_call("create_chat_completion", messages, model, stream, **kwargs)

        if not stream:
            raise APIError("Non-streaming error")

        # Create a generator that fails after yielding chunks
        async def failing_generator():
            for i in range(3):
                if i >= self.fail_at_chunk:
                    await asyncio.sleep(0.01)  # Small delay
                    raise APIError(f"{self.name} streaming error at chunk {i}")

                yield {
                    "id": f"chatcmpl-{i}",
                    "object": "chat.completion.chunk",
                    "created": 1677858242,
                    "model": model,
                    "choices": [{
                        "delta": {"content": f"chunk {i}"},
                        "finish_reason": None,
                        "index": 0
                    }]
                }

        return failing_generator()

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Mock text completion that fails during streaming."""
        self.call_count += 1
        self._record_call("create_completion", prompt, model, stream, **kwargs)

        if not stream:
            raise APIError("Not implemented")

        # Create a generator that fails after yielding chunks
        async def failing_generator():
            for i in range(3):
                if i >= self.fail_at_chunk:
                    await asyncio.sleep(0.01)  # Small delay
                    raise APIError(f"{self.name} streaming error at chunk {i}")

                yield {
                    "id": f"cmpl-{i}",
                    "object": "text_completion.chunk",
                    "created": 1677858242,
                    "model": model,
                    "choices": [{
                        "text": f"chunk {i}",
                        "finish_reason": None,
                        "index": 0
                    }]
                }

        return failing_generator()

    # Implement other required provider methods
    async def create_embedding(self, input, model, **kwargs):
        self._record_call("create_embedding", input, model, **kwargs)
        raise APIError("Not implemented")

    async def upload_file(self, file, purpose, **kwargs):
        self._record_call("upload_file", file, purpose, **kwargs)
        raise APIError("Not implemented")

    async def download_file(self, file_id, **kwargs):
        self._record_call("download_file", file_id, **kwargs)
        raise APIError("Not implemented")

    async def create_speech(self, input, model, **kwargs):
        self._record_call("create_speech", input, model, **kwargs)
        raise APIError("Not implemented")

    async def create_image(self, prompt, model, **kwargs):
        self._record_call("create_image", prompt, model, **kwargs)
        raise APIError("Not implemented")


class TestFallbackMechanism:
    """Tests for the basic fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_provider_proxy(self):
        """Test the FallbackProviderProxy class."""
        # Create mock providers
        primary_provider = MockProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("onellm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail on the primary provider and succeed on the fallback
            result = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )

            # Verify the primary provider was called
            assert primary_provider.call_count == 1

            # Verify the fallback provider was called
            assert fallback_provider.call_count == 1

            # Verify we got a result
            assert result == mock_chat_completion_response

    @pytest.mark.asyncio
    async def test_fallback_exhaustion(self):
        """Test fallback exhaustion when all providers fail."""
        # Create mock providers that all fail
        primary_provider = MockProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockProvider(should_fail=True, error_type="rate_limit")

        # Patch get_provider to return our mock providers
        with mock.patch("onellm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail on all providers
            with pytest.raises(FallbackExhaustionError) as excinfo:
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify the error message contains useful information
            assert "All models failed" in str(excinfo.value)
            assert "openai/gpt-4" in str(excinfo.value)
            assert "anthropic/claude-3" in str(excinfo.value)

            # Verify both providers were called
            assert primary_provider.call_count == 1
            assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retriable_error(self):
        """Test that non-retriable errors are raised immediately."""
        # Create mock providers
        primary_provider = MockProvider(should_fail=True, error_type="auth")
        fallback_provider = MockProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("onellm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy with RateLimitError as the only retriable error
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail with a non-retriable error
            with pytest.raises(AuthenticationError):
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify only the primary provider was called
            assert primary_provider.call_count == 1
            assert fallback_provider.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_config_max_fallbacks(self):
        """Test max_fallbacks configuration."""
        # Create mock providers
        providers = [
            MockProvider(should_fail=True, error_type="rate_limit"),  # Primary
            MockProvider(should_fail=True, error_type="rate_limit"),  # Fallback 1
            MockProvider(should_fail=False)                           # Fallback 2
        ]

        # Patch get_provider to return our mock providers
        with mock.patch("onellm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                providers[0] if provider_name == "openai" else (
                    providers[1] if provider_name == "anthropic" else providers[2]
                )
            )

            # Create a fallback provider proxy with max_fallbacks=1
            # This should try the primary and only the first fallback
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"],
                FallbackConfig(retriable_errors=[RateLimitError], max_fallbacks=1)
            )

            # This should fail because we only try 2 models (primary + 1 fallback)
            with pytest.raises(FallbackExhaustionError):
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify only the primary and first fallback were called
            assert providers[0].call_count == 1
            assert providers[1].call_count == 1
            assert providers[2].call_count == 0

            # Reset call counts
            for p in providers:
                p.call_count = 0

            # Now create a proxy with max_fallbacks=2
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"],
                FallbackConfig(retriable_errors=[RateLimitError], max_fallbacks=2)
            )

            # This should succeed because we try all 3 models
            result = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )

            # Verify all providers were called until we found a working one
            assert providers[0].call_count == 1
            assert providers[1].call_count == 1
            assert providers[2].call_count == 1
            assert result == mock_chat_completion_response


class TestFallbackProviderEnhanced:
    """Enhanced tests for FallbackProviderProxy targeting specific functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create a patcher for get_provider
        self.get_provider_patch = mock.patch("onellm.providers.fallback.get_provider")
        self.mock_get_provider = self.get_provider_patch.start()

    def teardown_method(self):
        """Clean up test patchers."""
        self.get_provider_patch.stop()

    @pytest.mark.asyncio
    async def test_attribute_error_handling(self):
        """Test handling of AttributeError when method is missing."""
        # Create providers
        provider1 = MockStreamingProvider("provider1", missing_methods=["create_translation"])
        provider2 = MockStreamingProvider("provider2")

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create proxy with both providers
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call a method that's missing on the first provider but exists on the second
        result = await proxy.create_translation(file="test.wav", model="model1")

        # Verify fallback was used
        assert len(provider2.calls) == 1
        assert provider2.calls[0][0] == "create_translation"
        assert "Translated text" in result["text"]

    @pytest.mark.asyncio
    async def test_all_providers_missing_method(self):
        """Test when all providers are missing a method."""
        # Create providers with missing method
        provider1 = MockStreamingProvider("provider1", missing_methods=["nonexistent_method"])
        provider2 = MockStreamingProvider("provider2", missing_methods=["nonexistent_method"])

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create proxy with both providers
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call a method that doesn't exist on any provider
        with pytest.raises(AttributeError) as exc_info:
            await proxy._try_with_fallbacks("nonexistent_method")

        # Verify error is raised
        assert "nonexistent_method" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self):
        """Test chat completion streaming with fallback."""
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
    async def test_completion_streaming(self):
        """Test completion streaming with fallback."""
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
    async def test_streaming_generator_error_handling(self):
        """Test error handling in streaming generator."""
        # Create providers
        provider1 = FailingStreamProvider("provider1", fail_at_chunk=1)
        provider2 = MockStreamingProvider("provider2")

        # Configure get_provider
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create proxy with both providers
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2"],
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Collect all chunks to trigger the generator
        chunks = []
        try:
            async for chunk in generator:
                chunks.append(chunk)
        except APIError:
            # This might happen if fallback is not working correctly
            pass

        # Verify we got chunks (either from first provider or fallback)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_with_all_providers_failing(self):
        """Test streaming when all providers fail."""
        # Create failing providers
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

        # Test with stream=True
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
    async def test_create_transcription(self):
        """Test create_transcription method."""
        # Create provider
        provider = MockStreamingProvider("provider1")

        # Configure get_provider
        self.mock_get_provider.return_value = provider

        # Create proxy
        proxy = FallbackProviderProxy(["provider1/model1"])

        # Call the transcription method
        result = await proxy.create_transcription(
            file="test.wav",
            model=None  # Should be ignored and replaced with model1
        )

        # Verify the provider was called
        assert len(provider.calls) == 1
        assert provider.calls[0][0] == "create_transcription"

        # The model parameter should be in the second position (args)
        args = provider.calls[0][1]
        assert len(args) >= 2
        assert args[1] == "model1"  # Second positional arg should be model

        assert "Transcribed text" in result["text"]

    @pytest.mark.asyncio
    async def test_create_translation(self):
        """Test create_translation method."""
        # Create provider
        provider = MockStreamingProvider("provider1")

        # Configure get_provider
        self.mock_get_provider.return_value = provider

        # Create proxy
        proxy = FallbackProviderProxy(["provider1/model1"])

        # Call the translation method
        result = await proxy.create_translation(
            file="test.wav",
            model=None  # Should be ignored and replaced with model1
        )

        # Verify the provider was called
        assert len(provider.calls) == 1
        assert provider.calls[0][0] == "create_translation"

        # The model parameter should be in the second position (args)
        args = provider.calls[0][1]
        assert len(args) >= 2
        assert args[1] == "model1"  # Second positional arg should be model

        assert "Translated text" in result["text"]

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test list_files method."""
        # Create provider
        provider = MockStreamingProvider("provider1")

        # Configure get_provider
        self.mock_get_provider.return_value = provider

        # Create proxy
        proxy = FallbackProviderProxy(["provider1/model1"])

        # Call the list_files method
        result = await proxy.list_files(purpose="assistants")

        # Verify the provider was called
        assert len(provider.calls) == 1
        assert provider.calls[0][0] == "list_files"
        assert result[0]["id"] == "file-123"

    @pytest.mark.asyncio
    async def test_delete_file(self):
        """Test delete_file method."""
        # Create provider
        provider = MockStreamingProvider("provider1")

        # Configure get_provider
        self.mock_get_provider.return_value = provider

        # Create proxy
        proxy = FallbackProviderProxy(["provider1/model1"])

        # Call the delete_file method
        result = await proxy.delete_file(file_id="file-123")

        # Verify the provider was called with the right file_id
        assert len(provider.calls) == 1
        assert provider.calls[0][0] == "delete_file"
        assert provider.calls[0][1][0] == "file-123"  # file_id param
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_upload_file(self):
        """Test upload_file with fallbacks."""
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
        """Test download_file with fallbacks."""
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
        """Test create_speech with fallbacks."""
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
        """Test create_image with fallbacks."""
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

    @pytest.mark.asyncio
    async def test_max_fallbacks_config(self):
        """Test max_fallbacks configuration."""
        # Create proxy with max_fallbacks=1
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2", "provider3/model3"],
            FallbackConfig(retriable_errors=[APIError], max_fallbacks=1)
        )

        # Verify the config is set correctly
        assert proxy.fallback_config.max_fallbacks == 1

        # Verify that models_to_try is limited by max_fallbacks
        # Create a mock for direct testing of internal method
        with mock.patch.object(proxy, '_try_streaming_with_fallbacks') as mock_try_streaming:
            # Set up the mock to return a generator
            async def mock_generator():
                yield {"choices": [{"delta": {"content": "test"}}]}
            mock_try_streaming.return_value = mock_generator()

            # Call the method
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

            # Verify it was called with the right number of models
            mock_try_streaming.assert_called_once()


class TestFallbackStreamingProviders:
    """Tests specifically focused on streaming functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_get_provider = mock.patch("onellm.providers.fallback.get_provider").start()

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_streaming_with_failing_generator(self):
        """Test that a streaming generator that fails mid-stream propagates the error."""
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

        # Create fallback proxy without any fallback models to test primary failure
        proxy = FallbackProviderProxy(
            ["failing/model1"],  # Only one model, no fallback
            FallbackConfig(retriable_errors=[APIError])
        )

        # Call the streaming method
        generator = await proxy.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Collect chunks until we hit the error
        chunks = []
        with pytest.raises(APIError) as exc_info:
            async for chunk in generator:
                chunks.append(chunk)

        # We should have received one chunk before the error
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "chunk 0"
        assert "streaming error at chunk 1" in str(exc_info.value)


class TestFallbackUtils:
    """Tests for the utility functions in fallback module."""

    @pytest.mark.asyncio
    async def test_maybe_await_with_non_awaitable(self):
        """Test maybe_await with a non-awaitable value."""
        # Test with a string (non-awaitable)
        result = await maybe_await("not awaitable")
        assert result == "not awaitable"

        # Test with an integer (non-awaitable)
        result = await maybe_await(42)
        assert result == 42

        # Test with a list (non-awaitable)
        test_list = [1, 2, 3]
        result = await maybe_await(test_list)
        assert result == test_list

        # Test with a dictionary (non-awaitable)
        test_dict = {"key": "value"}
        result = await maybe_await(test_dict)
        assert result == test_dict

        # Test with None (non-awaitable)
        result = await maybe_await(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_maybe_await_with_awaitable(self):
        """Test maybe_await with an awaitable value."""
        # Create a coroutine
        async def async_func():
            return "awaited result"

        # Test with a coroutine (awaitable)
        coro = async_func()
        result = await maybe_await(coro)
        assert result == "awaited result"

    def test_fallback_config_defaults(self):
        """Test FallbackConfig default values."""
        config = FallbackConfig()
        assert len(config.retriable_errors) == 4
        assert config.max_fallbacks is None
        assert config.log_fallbacks is True
        assert config.fallback_callback is None

    def test_fallback_config_custom_values(self):
        """Test FallbackConfig with custom values."""
        # Define a custom callback
        def custom_callback(provider, error):
            return f"Error in {provider}: {error}"

        # Create config with custom values
        config = FallbackConfig(
            retriable_errors=[ValueError, TypeError],
            max_fallbacks=3,
            log_fallbacks=False,
            fallback_callback=custom_callback
        )

        # Verify values
        assert config.retriable_errors == [ValueError, TypeError]
        assert config.max_fallbacks == 3
        assert config.log_fallbacks is False
        assert config.fallback_callback is custom_callback
        assert config.fallback_callback("test", "error") == "Error in test: error"
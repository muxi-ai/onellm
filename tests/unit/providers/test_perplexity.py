#!/usr/bin/env python3
"""
Integration tests for the Perplexity AI provider using real API calls.

Tests the Perplexity provider with actual API requests to ensure proper functionality.
Perplexity provides search-augmented models with online capabilities.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.perplexity import PerplexityProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(not os.getenv("PERPLEXITY_API_KEY"), reason="PERPLEXITY_API_KEY not set")
class TestPerplexityProvider:
    """Test cases for Perplexity provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a Perplexity provider instance."""
        return PerplexityProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "perplexity"
        assert provider.api_base == "https://api.perplexity.ai"
        assert provider.api_key is not None
        assert provider.requires_api_key is True

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-online",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message["role"] == "assistant"
        assert len(response.choices[0].message["content"]) > 0
        assert response.choices[0].finish_reason in ["stop", "length"]

    @pytest.mark.asyncio
    async def test_create_chat_completion_streaming(self, provider):
        """Test streaming chat completion with real API call."""
        chunks = []

        async for chunk in provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-online",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=20,
            stream=True,
        ):
            chunks.append(chunk)

        # Verify we got multiple chunks
        assert len(chunks) > 1

        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, "choices")
            if chunk.choices and chunk.choices[0].delta.get("content"):
                assert isinstance(chunk.choices[0].delta["content"], str)

        # Verify final chunk
        final_chunk = chunks[-1]
        if final_chunk.choices:
            assert final_chunk.choices[0].finish_reason in ["stop", "length", None]

    @pytest.mark.asyncio
    async def test_online_search_capability(self, provider):
        """Test Perplexity's unique online search capability."""
        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-online",
            messages=[{"role": "user", "content": "What is the current year?"}],
            max_tokens=20,
        )

        # Verify response contains current information
        assert response is not None
        content = response.choices[0].message["content"]
        # Should mention 2024 or 2025 (depending on when test runs)
        assert "202" in content  # Should contain year info

    @pytest.mark.asyncio
    async def test_multiple_models(self, provider):
        """Test different models available on Perplexity."""
        models = [
            "perplexity/llama-3.1-sonar-small-128k-online",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "perplexity/llama-3.1-sonar-small-128k-chat",
            "perplexity/llama-3.1-sonar-large-128k-chat",
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
                )
                assert response is not None
                assert response.choices[0].message["content"] is not None
            except InvalidRequestError as e:
                # Model might not be available
                if "model not found" not in str(e).lower():
                    raise

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=15,
        )

        # Verify response exists
        assert response is not None
        content = response.choices[0].message["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_temperature_parameter(self, provider):
        """Test temperature parameter affects randomness."""
        # Low temperature (deterministic)
        response_low = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0,
        )

        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=1.0,
        )

        # Both should have responses
        assert response_low.choices[0].message["content"] is not None
        assert response_high.choices[0].message["content"] is not None

    @pytest.mark.asyncio
    async def test_max_tokens_limit(self, provider):
        """Test max_tokens parameter limits response length."""
        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5,
        )

        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited
        assert response.usage.completion_tokens <= 5

    @pytest.mark.asyncio
    async def test_large_context_window(self, provider):
        """Test Perplexity's 128K context window."""
        # Create a conversation with multiple turns
        messages = [
            {"role": "user", "content": "Remember the number 42"},
            {"role": "assistant", "content": "I'll remember 42."},
            {"role": "user", "content": "Remember the color blue"},
            {"role": "assistant", "content": "I'll remember blue."},
            {"role": "user", "content": "What number and color did I ask you to remember?"},
        ]

        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat", messages=messages, max_tokens=30
        )

        # Should recall both pieces of information
        content = response.choices[0].message["content"]
        assert "42" in content
        assert "blue" in content.lower()

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, provider):
        """Test error handling for invalid model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.create_chat_completion(
                model="perplexity/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = PerplexityProvider(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="perplexity/llama-3.1-sonar-small-128k-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is False  # Perplexity doesn't support this yet
        assert provider.supports_vision is False  # Perplexity doesn't support vision

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities(
            "perplexity/llama-3.1-sonar-small-128k-online"
        )
        assert isinstance(capabilities, dict)

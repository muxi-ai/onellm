#!/usr/bin/env python3
"""
Integration tests for the Vercel AI Gateway provider using real API calls.

Tests the Vercel provider with actual API requests to ensure proper functionality.
Vercel AI Gateway provides access to 100+ models through a unified OpenAI-compatible interface.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.vercel import VercelProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(not os.getenv("VERCEL_AI_API_KEY"), reason="VERCEL_AI_API_KEY not set")
class TestVercelProvider:
    """Test cases for Vercel AI Gateway provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a Vercel provider instance."""
        return VercelProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "vercel"
        assert provider.api_base == "https://ai-gateway.vercel.sh/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="vercel/openai/gpt-4o-mini",
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
            model="vercel/openai/gpt-4o-mini",
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

    @pytest.mark.asyncio
    async def test_multiple_providers_through_vercel(self, provider):
        """Test accessing different providers through Vercel AI Gateway."""
        # Test different model providers available through Vercel
        models = [
            "vercel/openai/gpt-4o-mini",  # OpenAI model
            "vercel/anthropic/claude-sonnet-4",  # Anthropic model
            "vercel/google/gemini-2.0-flash-exp",  # Google model
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
                )
                assert response is not None
                assert response.choices[0].message["content"] is not None
            except InvalidRequestError as e:
                # Model might not be available or deprecated
                if (
                    "model not found" not in str(e).lower()
                    and "not available" not in str(e).lower()
                ):
                    raise

    @pytest.mark.asyncio
    async def test_provider_routing(self, provider):
        """Test that Vercel properly routes to different providers."""
        response = await provider.create_chat_completion(
            model="vercel/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What model are you?"}],
            max_tokens=20,
        )

        # Vercel should return the actual model used
        assert response is not None
        assert response.model is not None

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="vercel/openai/gpt-4o-mini",
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
            model="vercel/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0,
        )

        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="vercel/openai/gpt-4o-mini",
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
            model="vercel/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5,
        )

        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited
        assert response.usage.completion_tokens <= 5

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, provider):
        """Test error handling for invalid model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.create_chat_completion(
                model="vercel/invalid-provider/invalid-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = VercelProvider(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="vercel/openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True
        assert provider.supports_vision is True

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("vercel/openai/gpt-4o-mini")
        assert isinstance(capabilities, dict)

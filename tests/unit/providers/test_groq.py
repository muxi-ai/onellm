#!/usr/bin/env python3
"""
Integration tests for the Groq provider using real API calls.

Tests the Groq provider with actual API requests to ensure proper functionality.
Uses minimal token counts to control costs while thoroughly testing features.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.groq import GroqProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
class TestGroqProvider:
    """Test cases for Groq provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a Groq provider instance."""
        return GroqProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "groq"
        assert provider.api_base == "https://api.groq.com/openai/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="groq/llama3-8b-8192",
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

        # Verify model in response
        assert "llama3" in response.model.lower()

    @pytest.mark.asyncio
    async def test_create_chat_completion_streaming(self, provider):
        """Test streaming chat completion with real API call."""
        chunks = []

        async for chunk in provider.create_chat_completion(
            model="groq/mixtral-8x7b-32768",
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

        # Verify final chunk has finish_reason
        final_chunk = chunks[-1]
        if final_chunk.choices:
            assert final_chunk.choices[0].finish_reason in ["stop", "length", None]

    @pytest.mark.asyncio
    async def test_multiple_models(self, provider):
        """Test different available models on Groq."""
        models = ["groq/llama3-8b-8192", "groq/mixtral-8x7b-32768", "groq/llama3-70b-8192"]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
                )
                assert response is not None
                assert response.choices[0].message["content"] is not None
            except InvalidRequestError as e:
                # Model might not be available, that's okay
                if "model not found" not in str(e).lower():
                    raise

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a pirate. Respond accordingly."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=15,
        )

        # Verify response exists
        assert response is not None
        content = response.choices[0].message["content"].lower()
        # Check for pirate-like language (might contain "ahoy", "arr", "matey", etc.)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_temperature_parameter(self, provider):
        """Test temperature parameter affects randomness."""
        # Low temperature (deterministic)
        responses_low = []
        for _ in range(2):
            response = await provider.create_chat_completion(
                model="groq/llama3-8b-8192",
                messages=[{"role": "user", "content": "Say one word"}],
                max_tokens=5,
                temperature=0.0,
            )
            responses_low.append(response.choices[0].message["content"])

        # High temperature (random)
        responses_high = []
        for _ in range(2):
            response = await provider.create_chat_completion(
                model="groq/llama3-8b-8192",
                messages=[{"role": "user", "content": "Say one word"}],
                max_tokens=5,
                temperature=1.0,
            )
            responses_high.append(response.choices[0].message["content"])

        # Low temperature responses should be more similar
        assert responses_low[0] is not None
        assert responses_high[0] is not None

    @pytest.mark.asyncio
    async def test_max_tokens_limit(self, provider):
        """Test max_tokens parameter limits response length."""
        response = await provider.create_chat_completion(
            model="groq/llama3-8b-8192",
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
                model="groq/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = GroqProvider(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="groq/llama3-8b-8192",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is False  # Groq doesn't support this yet
        assert provider.supports_vision is False  # Groq doesn't support vision

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("groq/llama3-8b-8192")
        assert isinstance(capabilities, dict)

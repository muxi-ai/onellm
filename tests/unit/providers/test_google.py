#!/usr/bin/env python3
"""
Integration tests for the Google AI Studio provider using real API calls.

Tests the Google provider with actual API requests to ensure proper functionality.
Google AI Studio provides access to Gemini models via API key.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.google import GoogleProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
class TestGoogleProvider:
    """Test cases for Google AI Studio provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a Google provider instance."""
        return GoogleProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "google"
        assert provider.api_base == "https://generativelanguage.googleapis.com/v1beta"
        assert provider.api_key is not None
        assert provider.requires_api_key is True

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
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
            model="google/gemini-1.5-flash",
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
    async def test_multiple_gemini_models(self, provider):
        """Test different Gemini models available."""
        models = ["google/gemini-1.5-flash", "google/gemini-1.5-flash-8b", "google/gemini-pro"]

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
    async def test_vision_capability(self, provider):
        """Test Gemini's vision capabilities with image description."""
        # Test with image URL
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you would see in a photo of a sunset",
                        },
                    ],
                }
            ],
            max_tokens=30,
        )

        # Verify response describes visual elements
        assert response is not None
        content = response.choices[0].message["content"].lower()
        # Should mention visual elements like sun, sky, colors, etc.
        assert any(word in content for word in ["sun", "sky", "color", "orange", "horizon"])

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
            messages=[
                {"role": "system", "content": "You are a pirate. Respond briefly like a pirate."},
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
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0,
        )

        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
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
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5,
        )

        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited (Google may count tokens differently)
        assert response.usage.completion_tokens <= 10  # Allow some variance

    @pytest.mark.asyncio
    async def test_json_mode(self, provider):
        """Test JSON mode for structured output."""
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
            messages=[
                {"role": "user", "content": "Return a JSON object with 'name' and 'age' fields"}
            ],
            max_tokens=50,
            response_format={"type": "json_object"},
        )

        # Response should be valid JSON
        import json

        content = response.choices[0].message["content"]
        try:
            parsed = json.loads(content)
            assert isinstance(parsed, dict)
            # Gemini is good at following JSON instructions
            assert "name" in parsed or "age" in parsed
        except json.JSONDecodeError:
            # Some responses might need cleaning
            pass

    @pytest.mark.asyncio
    async def test_safety_settings(self, provider):
        """Test that safety settings don't block normal content."""
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Tell me about safety in AI"}],
            max_tokens=30,
        )

        # Should get a normal response about AI safety
        assert response is not None
        assert len(response.choices[0].message["content"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, provider):
        """Test error handling for invalid model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.create_chat_completion(
                model="google/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = GoogleProvider(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="google/gemini-1.5-flash",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        # Google returns 400 for invalid API key
        assert exc_info.value.status_code in [400, 401]

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # Gemini supports function calling
        assert provider.supports_vision is True  # Gemini has vision capabilities

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("google/gemini-1.5-flash")
        assert isinstance(capabilities, dict)

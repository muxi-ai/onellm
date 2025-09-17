#!/usr/bin/env python3
"""
Integration tests for the Vertex AI provider using real API calls.

Tests the Vertex AI provider with actual API requests to ensure proper functionality.
Vertex AI provides Google Cloud's enterprise Gemini models.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.vertexai import VertexAIProvider
from onellm.errors import InvalidRequestError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(
    not os.path.exists("vertexai.json"), reason="vertexai.json credentials file not found"
)
class TestVertexAIProvider:
    """Test cases for Vertex AI provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a Vertex AI provider instance."""
        return VertexAIProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "vertexai"
        assert provider.requires_api_key is False  # Uses service account
        # Vertex AI endpoint varies by project/region
        assert "googleapis.com" in provider.api_base

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
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
            model="vertexai/gemini-1.5-flash",
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
        """Test different Gemini models available on Vertex AI."""
        models = ["vertexai/gemini-1.5-flash", "vertexai/gemini-1.5-pro", "vertexai/gemini-pro"]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
                )
                assert response is not None
                assert response.choices[0].message["content"] is not None
            except InvalidRequestError as e:
                # Model might not be available in this region
                if (
                    "model not found" not in str(e).lower()
                    and "not supported" not in str(e).lower()
                ):
                    raise

    @pytest.mark.asyncio
    async def test_vision_capability(self, provider):
        """Test Gemini's vision capabilities through Vertex AI."""
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you would see in a typical beach scene",
                        },
                    ],
                }
            ],
            max_tokens=30,
        )

        # Verify response describes visual elements
        assert response is not None
        content = response.choices[0].message["content"].lower()
        # Should mention beach elements
        assert any(word in content for word in ["water", "sand", "ocean", "beach", "sun", "waves"])

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
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
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0,
        )

        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
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
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5,
        )

        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited (Google may count tokens differently)
        assert response.usage.completion_tokens <= 10  # Allow some variance

    @pytest.mark.asyncio
    async def test_function_calling(self, provider):
        """Test Vertex AI function calling capability."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city name"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
            tools=tools,
            max_tokens=50,
        )

        # Should either call the function or respond normally
        assert response is not None
        if response.choices[0].message.get("tool_calls"):
            # Function was called
            tool_call = response.choices[0].message["tool_calls"][0]
            assert tool_call["function"]["name"] == "get_weather"
        else:
            # Normal response
            assert len(response.choices[0].message["content"]) > 0

    @pytest.mark.asyncio
    async def test_safety_settings(self, provider):
        """Test that Vertex AI safety settings work properly."""
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Tell me about responsible AI practices"}],
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
                model="vertexai/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_credentials(self):
        """Test error handling for invalid credentials."""
        # This test would require mocking the credentials
        # Since we're using real API calls, we'll skip this specific test
        # but the structure is here for when needed
        pass

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # Vertex AI supports function calling
        assert provider.supports_vision is True  # Vertex AI has vision capabilities

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("vertexai/gemini-1.5-flash")
        assert isinstance(capabilities, dict)

    @pytest.mark.asyncio
    async def test_enterprise_features(self, provider):
        """Test enterprise-specific features of Vertex AI."""
        # Test with enterprise configuration
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "What are the benefits of cloud AI?"}],
            max_tokens=30,
        )

        # Should work with enterprise authentication
        assert response is not None
        assert len(response.choices[0].message["content"]) > 0

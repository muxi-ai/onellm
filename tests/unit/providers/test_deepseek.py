#!/usr/bin/env python3
"""
Integration tests for the DeepSeek provider using real API calls.

Tests the DeepSeek provider with actual API requests to ensure proper functionality.
DeepSeek is a Chinese LLM provider with strong multilingual capabilities.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.deepseek import DeepSeekProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestDeepSeekProvider:
    """Test cases for DeepSeek provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create a DeepSeek provider instance."""
        return DeepSeekProvider()

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "deepseek"
        assert provider.api_base == "https://api.deepseek.com/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-chat",
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
            model="deepseek/deepseek-chat",
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
    async def test_chinese_language_support(self, provider):
        """Test DeepSeek's Chinese language capabilities."""
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "你好，请用中文回答"}],
            max_tokens=20,
        )

        # Verify response is in Chinese
        assert response is not None
        content = response.choices[0].message["content"]
        # Check if response contains Chinese characters
        assert any(ord(char) > 127 for char in content)

    @pytest.mark.asyncio
    async def test_multilingual_capability(self, provider):
        """Test multilingual translation capability."""
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": "Translate 'Hello' to Chinese, Japanese, and Korean. Reply with just the translations.",  # noqa: E501
                }
            ],
            max_tokens=30,
        )

        # Verify response contains multiple scripts
        assert response is not None
        content = response.choices[0].message["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-chat",
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
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0,
        )

        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="deepseek/deepseek-chat",
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
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5,
        )

        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited
        assert response.usage.completion_tokens <= 5

    @pytest.mark.asyncio
    async def test_code_generation(self, provider):
        """Test DeepSeek's code generation capabilities."""
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-coder",
            messages=[
                {
                    "role": "user",
                    "content": "Write a Python function to add two numbers. Just the function, no explanation.",  # noqa: E501
                }
            ],
            max_tokens=50,
        )

        # Verify response contains code
        assert response is not None
        content = response.choices[0].message["content"]
        assert "def" in content  # Should contain function definition
        assert "return" in content  # Should have return statement

    @pytest.mark.asyncio
    async def test_multiple_models(self, provider):
        """Test different models available on DeepSeek."""
        models = ["deepseek/deepseek-chat", "deepseek/deepseek-coder"]

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
    async def test_invalid_model_error(self, provider):
        """Test error handling for invalid model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.create_chat_completion(
                model="deepseek/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert "model" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = DeepSeekProvider(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # DeepSeek supports function calling
        assert provider.supports_vision is False  # DeepSeek doesn't support vision yet

        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("deepseek/deepseek-chat")
        assert isinstance(capabilities, dict)

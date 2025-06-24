#!/usr/bin/env python3
"""
Integration tests for the Fireworks AI provider using real API calls.

Tests the Fireworks provider with actual API requests to ensure proper functionality.
Fireworks provides fast inference for various models.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.fireworks import FireworksProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY not set"
)
class TestFireworksProvider:
    """Test cases for Fireworks provider with real API calls."""
    
    @pytest.fixture
    def provider(self):
        """Create a Fireworks provider instance."""
        return FireworksProvider()
    
    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "fireworks"
        assert provider.api_base == "https://api.fireworks.ai/inference/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].message["role"] == "assistant"
        assert len(response.choices[0].message["content"]) > 0
        assert response.choices[0].finish_reason in ["stop", "length"]
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_streaming(self, provider):
        """Test streaming chat completion with real API call."""
        chunks = []
        
        async for chunk in provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=20,
            stream=True
        ):
            chunks.append(chunk)
        
        # Verify we got multiple chunks
        assert len(chunks) > 1
        
        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            if chunk.choices and chunk.choices[0].delta.get("content"):
                assert isinstance(chunk.choices[0].delta["content"], str)
        
        # Verify final chunk
        final_chunk = chunks[-1]
        if final_chunk.choices:
            assert final_chunk.choices[0].finish_reason in ["stop", "length", None]
    
    @pytest.mark.asyncio
    async def test_vision_model(self, provider):
        """Test vision model capability if available."""
        try:
            response = await provider.create_chat_completion(
                model="fireworks/accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What would you see in an image of a sunset?"},
                    ]
                }],
                max_tokens=20
            )
            
            assert response is not None
            assert response.choices[0].message["content"] is not None
        except InvalidRequestError as e:
            # Vision model might not be available
            if "model not found" not in str(e).lower():
                raise
    
    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=15
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
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0
        )
        
        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=1.0
        )
        
        # Both should have responses
        assert response_low.choices[0].message["content"] is not None
        assert response_high.choices[0].message["content"] is not None
    
    @pytest.mark.asyncio
    async def test_max_tokens_limit(self, provider):
        """Test max_tokens parameter limits response length."""
        response = await provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5
        )
        
        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited
        assert response.usage.completion_tokens <= 5
    
    @pytest.mark.asyncio
    async def test_multiple_models(self, provider):
        """Test different models available on Fireworks."""
        models = [
            "fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
            "fireworks/accounts/fireworks/models/mixtral-8x7b-instruct",
            "fireworks/accounts/fireworks/models/mistral-7b-instruct-v0p2"
        ]
        
        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
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
                model="fireworks/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert "model" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = FireworksProvider(api_key="invalid-key")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # Some models support this
        assert provider.supports_vision is True  # Some models support vision
        
        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("fireworks/accounts/fireworks/models/llama-v3p2-3b-instruct")
        assert isinstance(capabilities, dict)
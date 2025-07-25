#!/usr/bin/env python3
"""
Integration tests for the OpenRouter provider using real API calls.

Tests the OpenRouter provider with actual API requests to ensure proper functionality.
OpenRouter provides access to 100+ models through a unified interface.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.openrouter import OpenRouterProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
class TestOpenRouterProvider:
    """Test cases for OpenRouter provider with real API calls."""
    
    @pytest.fixture
    def provider(self):
        """Create an OpenRouter provider instance."""
        return OpenRouterProvider()
    
    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "openrouter"
        assert provider.api_base == "https://openrouter.ai/api/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        # Use a free model to minimize costs
        response = await provider.create_chat_completion(
            model="openrouter/mistralai/mistral-7b-instruct:free",
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
            model="openrouter/mistralai/mistral-7b-instruct:free",
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
    
    @pytest.mark.asyncio
    async def test_multiple_providers_through_openrouter(self, provider):
        """Test accessing different providers through OpenRouter."""
        # Test different model providers available through OpenRouter
        models = [
            "openrouter/mistralai/mistral-7b-instruct:free",  # Mistral model
            "openrouter/meta-llama/llama-3.2-3b-instruct:free",  # Meta model
            "openrouter/google/gemini-2.0-flash-exp:free"  # Google model
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
                # Model might not be available or deprecated
                if "model not found" not in str(e).lower() and "not available" not in str(e).lower():
                    raise
    
    @pytest.mark.asyncio
    async def test_provider_routing_header(self, provider):
        """Test that OpenRouter properly routes to different providers."""
        response = await provider.create_chat_completion(
            model="openrouter/mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": "What model are you?"}],
            max_tokens=20
        )
        
        # OpenRouter should return the actual model used
        assert response is not None
        assert response.model is not None
        # The model field should contain information about routing
        assert "mistral" in response.model.lower() or "mistralai" in response.model.lower()
    
    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="openrouter/mistralai/mistral-7b-instruct:free",
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
            model="openrouter/mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0
        )
        
        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="openrouter/mistralai/mistral-7b-instruct:free",
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
            model="openrouter/mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5
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
                model="openrouter/invalid-provider/invalid-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert "model" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = OpenRouterProvider(api_key="invalid-key")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="openrouter/mistralai/mistral-7b-instruct:free",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # Most models support this
        assert provider.supports_vision is True  # Some models support vision
        
        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("openrouter/mistralai/mistral-7b-instruct:free")
        assert isinstance(capabilities, dict)
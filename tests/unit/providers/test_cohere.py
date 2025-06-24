#!/usr/bin/env python3
"""
Integration tests for the Cohere provider using real API calls.

Tests the Cohere provider with actual API requests to ensure proper functionality.
Cohere provides enterprise NLP capabilities with RAG support.
"""

import os
import pytest
from dotenv import load_dotenv

from onellm.providers.cohere import CohereProvider
from onellm.errors import InvalidRequestError, AuthenticationError

# Load environment variables
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"),
    reason="COHERE_API_KEY not set"
)
class TestCohereProvider:
    """Test cases for Cohere provider with real API calls."""
    
    @pytest.fixture
    def provider(self):
        """Create a Cohere provider instance."""
        return CohereProvider()
    
    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.provider_name == "cohere"
        assert provider.api_base == "https://api.cohere.ai/v1"
        assert provider.api_key is not None
        assert provider.requires_api_key is True
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, provider):
        """Test basic chat completion with real API call."""
        response = await provider.create_chat_completion(
            model="cohere/command-r",
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
            model="cohere/command-r",
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
    async def test_create_embedding(self, provider):
        """Test embedding generation with real API call."""
        response = await provider.create_embedding(
            model="cohere/embed-english-v3.0",
            input="Hello, world!"
        )
        
        # Verify embedding response
        assert response is not None
        assert hasattr(response, 'data')
        assert len(response.data) == 1
        assert hasattr(response.data[0], 'embedding')
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) > 0
        # Cohere embeddings are typically 1024 or 4096 dimensions
        assert len(response.data[0].embedding) in [1024, 4096]
    
    @pytest.mark.asyncio
    async def test_multiple_cohere_models(self, provider):
        """Test different Cohere models available."""
        chat_models = [
            "cohere/command-r",
            "cohere/command-r-plus",
            "cohere/command"
        ]
        
        for model in chat_models:
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
    async def test_embedding_models(self, provider):
        """Test different Cohere embedding models."""
        embedding_models = [
            "cohere/embed-english-v3.0",
            "cohere/embed-multilingual-v3.0"
        ]
        
        for model in embedding_models:
            try:
                response = await provider.create_embedding(
                    model=model,
                    input="Test text"
                )
                assert response is not None
                assert len(response.data) == 1
                assert isinstance(response.data[0].embedding, list)
            except InvalidRequestError as e:
                # Model might not be available
                if "model not found" not in str(e).lower():
                    raise
    
    @pytest.mark.asyncio
    async def test_system_message(self, provider):
        """Test chat completion with system message."""
        response = await provider.create_chat_completion(
            model="cohere/command-r",
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
            model="cohere/command-r",
            messages=[{"role": "user", "content": "Say one word"}],
            max_tokens=5,
            temperature=0.0
        )
        
        # High temperature (random)
        response_high = await provider.create_chat_completion(
            model="cohere/command-r",
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
            model="cohere/command-r",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            max_tokens=5
        )
        
        # Response should be truncated
        assert response.choices[0].finish_reason == "length"
        # Token count should be limited
        assert response.usage.completion_tokens <= 5
    
    @pytest.mark.asyncio
    async def test_multilingual_capability(self, provider):
        """Test Cohere's multilingual capabilities."""
        response = await provider.create_chat_completion(
            model="cohere/command-r-plus",
            messages=[{"role": "user", "content": "Bonjour, comment allez-vous?"}],
            max_tokens=20
        )
        
        # Should respond appropriately to French
        assert response is not None
        content = response.choices[0].message["content"]
        assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self, provider):
        """Test creating embeddings for multiple texts."""
        texts = ["Hello world", "Goodbye world", "How are you?"]
        
        response = await provider.create_embedding(
            model="cohere/embed-english-v3.0",
            input=texts
        )
        
        # Should get embeddings for all texts
        assert response is not None
        assert len(response.data) == len(texts)
        
        for i, embedding_obj in enumerate(response.data):
            assert isinstance(embedding_obj.embedding, list)
            assert len(embedding_obj.embedding) > 0
            assert embedding_obj.index == i
    
    @pytest.mark.asyncio
    async def test_invalid_model_error(self, provider):
        """Test error handling for invalid model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.create_chat_completion(
                model="cohere/invalid-model-xyz",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert "model" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test error handling for invalid API key."""
        provider = CohereProvider(api_key="invalid-key")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await provider.create_chat_completion(
                model="cohere/command-r",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_provider_capabilities(self, provider):
        """Test provider capability flags."""
        assert provider.supports_streaming is True
        assert provider.supports_function_calling is True  # Cohere supports tools
        assert provider.supports_vision is False  # Cohere doesn't support vision yet
        
        # Test model-specific capabilities
        capabilities = provider.get_model_capabilities("cohere/command-r")
        assert isinstance(capabilities, dict)
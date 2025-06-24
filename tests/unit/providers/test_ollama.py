#!/usr/bin/env python3
"""
Tests for the Ollama provider.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from onellm.providers.ollama import OllamaProvider
from onellm.errors import InvalidRequestError, ServiceUnavailableError, ResourceNotFoundError


class TestOllamaProvider:
    """Test cases for Ollama provider."""
    
    def test_init_default(self):
        """Test initialization with default settings."""
        provider = OllamaProvider()
        assert provider.provider_name == "ollama"
        assert provider.api_base == "http://localhost:11434"
        assert provider.api_key == "not-required"
        assert not provider.requires_api_key
    
    def test_init_custom_base(self):
        """Test initialization with custom API base."""
        provider = OllamaProvider(api_base="http://custom:11434")
        assert provider.api_base == "http://custom:11434"
    
    def test_parse_ollama_model_simple(self):
        """Test parsing simple model names."""
        provider = OllamaProvider()
        
        # Simple model name
        model, endpoint = provider._parse_ollama_model("llama3:8b")
        assert model == "llama3:8b"
        assert endpoint == "http://localhost:11434"
    
    def test_parse_ollama_model_with_endpoint(self):
        """Test parsing model names with endpoints."""
        provider = OllamaProvider()
        
        # Model with endpoint
        model, endpoint = provider._parse_ollama_model("llama3:8b@server:11434")
        assert model == "llama3:8b"
        assert endpoint == "http://server:11434"
        
        # Model with IP endpoint
        model, endpoint = provider._parse_ollama_model("mixtral:8x7b@10.0.0.5:11434")
        assert model == "mixtral:8x7b"
        assert endpoint == "http://10.0.0.5:11434"
        
        # Model with http prefix
        model, endpoint = provider._parse_ollama_model("llama3:8b@http://server:11434")
        assert model == "llama3:8b"
        assert endpoint == "http://server:11434"
        
        # Model with https prefix
        model, endpoint = provider._parse_ollama_model("llama3:8b@https://server:11434")
        assert model == "llama3:8b"
        assert endpoint == "https://server:11434"
    
    def test_parse_ollama_model_complex_tags(self):
        """Test parsing model names with complex tags."""
        provider = OllamaProvider()
        
        # Complex model tag
        model, endpoint = provider._parse_ollama_model(
            "llama3:70b-instruct-q4_K_M@gpu-server:11434"
        )
        assert model == "llama3:70b-instruct-q4_K_M"
        assert endpoint == "http://gpu-server:11434"
    
    def test_parse_ollama_model_invalid_endpoint(self):
        """Test parsing model with invalid endpoint format."""
        provider = OllamaProvider()
        
        # Missing port
        with pytest.raises(InvalidRequestError) as exc:
            provider._parse_ollama_model("llama3:8b@server")
        assert "Invalid endpoint format" in str(exc.value)
        
        # Invalid format
        with pytest.raises(InvalidRequestError) as exc:
            provider._parse_ollama_model("llama3:8b@server:port")
        assert "Invalid endpoint format" in str(exc.value)
    
    def test_is_vision_model(self):
        """Test vision model detection."""
        provider = OllamaProvider()
        
        # Vision models
        assert provider._is_vision_model("llava")
        assert provider._is_vision_model("llava:latest")
        assert provider._is_vision_model("bakllava:34b")
        assert provider._is_vision_model("llava-llama3:latest")
        assert provider._is_vision_model("llava-phi3")
        assert provider._is_vision_model("moondream:latest")
        assert provider._is_vision_model("llama3.2-vision:11b")
        
        # Vision models with endpoints
        assert provider._is_vision_model("llava:latest@server:11434")
        assert provider._is_vision_model("llama3.2-vision:11b@10.0.0.5:11434")
        
        # Non-vision models
        assert not provider._is_vision_model("llama3:8b")
        assert not provider._is_vision_model("mistral:7b")
        assert not provider._is_vision_model("mixtral:8x7b")
    
    @pytest.mark.asyncio
    async def test_check_model_available_success(self):
        """Test checking model availability - success case."""
        provider = OllamaProvider()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "models": [
                    {"name": "llama3:8b"},
                    {"name": "mistral:7b"},
                    {"name": "llava:latest"}
                ]
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Check available model
            assert await provider._check_model_available("llama3:8b", "http://localhost:11434")
            
            # Check unavailable model
            assert not await provider._check_model_available("gpt-4", "http://localhost:11434")
            
            # Check cached result (should not make another request)
            assert await provider._check_model_available("llama3:8b", "http://localhost:11434")
    
    @pytest.mark.asyncio
    async def test_check_model_available_failure(self):
        """Test checking model availability - failure case."""
        provider = OllamaProvider()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Should return True on failure (assume available)
            assert await provider._check_model_available("llama3:8b", "http://localhost:11434")
    
    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test listing models - success case."""
        provider = OllamaProvider()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "models": [
                    {"name": "llama3:8b", "size": 4000000000},
                    {"name": "mistral:7b", "size": 3500000000},
                    {"name": "llava:latest", "size": 5000000000}
                ]
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            models = await provider.list_models()
            assert models == ["llama3:8b", "mistral:7b", "llava:latest"]
    
    @pytest.mark.asyncio
    async def test_list_models_custom_endpoint(self):
        """Test listing models with custom endpoint."""
        provider = OllamaProvider()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "models": [{"name": "mixtral:8x7b"}]
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            models = await provider.list_models("http://gpu-server:11434")
            assert models == ["mixtral:8x7b"]
            
            # Check that the correct URL was called
            call_args = mock_session.return_value.__aenter__.return_value.get.call_args
            assert call_args[0][0] == "http://gpu-server:11434/api/tags"
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_with_endpoint(self):
        """Test chat completion with model@endpoint syntax."""
        provider = OllamaProvider()
        
        # Mock the parent's create_chat_completion
        with patch.object(provider, "_make_request") as mock_request:
            mock_request.return_value = {
                "id": "test-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "llama3:8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }]
            }
            
            # Also mock model availability check
            with patch.object(provider, "_check_model_available", return_value=True):
                response = await provider.create_chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="llama3:8b@gpu-server:11434"
                )
                
                # Check that model was parsed correctly in request
                call_args = mock_request.call_args
                assert call_args[1]["data"]["model"] == "llama3:8b"
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_ollama_params(self):
        """Test chat completion with Ollama-specific parameters."""
        provider = OllamaProvider()
        
        with patch.object(provider, "_make_request") as mock_request:
            mock_request.return_value = {
                "id": "test-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "llama3:8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "4"},
                    "finish_reason": "stop"
                }]
            }
            
            with patch.object(provider, "_check_model_available", return_value=True):
                response = await provider.create_chat_completion(
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    model="llama3:8b",
                    num_gpu=1,
                    num_thread=8,
                    num_ctx=2048,
                    temperature=0.1
                )
                
                # Check that Ollama params were passed correctly
                call_args = mock_request.call_args
                data = call_args[1]["data"]
                assert "options" in data
                assert data["options"]["num_gpu"] == 1
                assert data["options"]["num_thread"] == 8
                assert data["options"]["num_ctx"] == 2048
                assert data["options"]["temperature"] == 0.1
    
    @pytest.mark.asyncio
    async def test_error_handling_connection_error(self):
        """Test error handling for connection errors."""
        provider = OllamaProvider()
        
        with patch.object(provider, "_check_model_available", return_value=True):
            # Mock parent's _make_request to raise ServiceUnavailableError
            original_make_request = provider._make_request
            
            async def mock_make_request(*args, **kwargs):
                # Simulate connection error
                raise ServiceUnavailableError(
                    "Connection refused",
                    provider="ollama",
                    status_code=503
                )
            
            provider._make_request = mock_make_request
            
            with pytest.raises(ServiceUnavailableError) as exc:
                await provider.create_chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="llama3:8b"
                )
            
            assert "Cannot connect to Ollama server" in str(exc.value)
            assert "ollama serve" in str(exc.value)
    
    @pytest.mark.asyncio
    async def test_error_handling_model_not_found(self):
        """Test error handling for model not found errors."""
        provider = OllamaProvider()
        
        with patch.object(provider, "_check_model_available", return_value=False):
            # Mock parent's _make_request to raise ResourceNotFoundError
            original_make_request = provider._make_request
            
            async def mock_make_request(*args, **kwargs):
                # Simulate model not found
                raise ResourceNotFoundError(
                    "model not found",
                    provider="ollama",
                    status_code=404
                )
            
            provider._make_request = mock_make_request
            
            with pytest.raises(ResourceNotFoundError) as exc:
                await provider.create_chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="unknown-model"
                )
            
            assert "Model 'unknown-model' not found" in str(exc.value)
            assert "ollama pull unknown-model" in str(exc.value)
#!/usr/bin/env python3
"""Test suite for Azure OpenAI provider."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from onellm.errors import AuthenticationError, InvalidRequestError
from onellm.models import ChatCompletionResponse, EmbeddingResponse
from onellm.providers.azure import AzureProvider


@pytest.fixture
def azure_config():
    """Create a temporary Azure configuration file."""
    config = {
        "key1": "test-key-1",
        "key2": "test-key-2",
        "region": "uksouth",
        "endpoint": "https://test.openai.azure.com/",
        "deployment": {
            "gpt-4o-mini": {
                "endpoint": "https://test-eastus.openai.azure.com/",
                "model_name": "gpt-4o-mini",
                "deployment": "gpt-4o-mini-deployment",
                "subscription_key": "deployment-key-1",
                "api_version": "2024-12-01-preview",
            },
            "o4-mini": {
                "endpoint": "https://test-westus.openai.azure.com/",
                "model_name": "o4-mini",
                "deployment": "o4-mini-deployment",
                "subscription_key": "deployment-key-2",
                "api_version": "2024-12-01-preview",
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def azure_provider(azure_config):
    """Create an Azure provider instance with test configuration."""
    return AzureProvider(azure_config_path=azure_config)


class TestAzureProvider:
    """Test cases for Azure OpenAI provider."""

    def test_init_with_config_file(self, azure_config):
        """Test initialization with configuration file."""
        provider = AzureProvider(azure_config_path=azure_config)
        assert provider.key1 == "test-key-1"
        assert provider.key2 == "test-key-2"
        assert provider.region == "uksouth"
        assert provider.endpoint == "https://test.openai.azure.com"
        assert len(provider.deployments) == 2

    def test_init_missing_config(self):
        """Test initialization with missing configuration file."""
        with pytest.raises(AuthenticationError) as exc_info:
            AzureProvider(azure_config_path="/nonexistent/path.json")
        assert "Azure configuration file not found" in str(exc_info.value)

    def test_deployment_config(self, azure_provider):
        """Test deployment configuration retrieval."""
        # Test specific deployment
        config = azure_provider._get_deployment_config("gpt-4o-mini")
        assert config["endpoint"] == "https://test-eastus.openai.azure.com/"
        assert config["deployment"] == "gpt-4o-mini-deployment"
        assert config["subscription_key"] == "deployment-key-1"

        # Test fallback for unknown model
        config = azure_provider._get_deployment_config("unknown-model")
        assert config["endpoint"] == "https://test.openai.azure.com"
        assert config["deployment"] == "unknown-model"
        assert config["subscription_key"] == "test-key-1"

    def test_headers(self, azure_provider):
        """Test header generation."""
        config = azure_provider._get_deployment_config("gpt-4o-mini")
        headers = azure_provider._get_headers(config)

        assert headers["Content-Type"] == "application/json"
        assert headers["api-key"] == "deployment-key-1"
        assert "Authorization" not in headers  # Azure uses api-key header

    def test_url_construction(self, azure_provider):
        """Test URL construction for API requests."""
        config = azure_provider._get_deployment_config("gpt-4o-mini")
        url = azure_provider._get_url(config, "/chat/completions")

        expected = "https://test-eastus.openai.azure.com/openai/deployments/gpt-4o-mini-deployment/chat/completions?api-version=2024-12-01-preview"  # noqa: E501
        assert url == expected

    @pytest.mark.asyncio
    async def test_chat_completion(self, azure_provider):
        """Test chat completion functionality."""
        # Mock the HTTP response
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! I'm Azure OpenAI."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(azure_provider, "_make_request", return_value=mock_response):
            response = await azure_provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
            )

            assert isinstance(response, ChatCompletionResponse)
            assert response.choices[0].message["content"] == "Hello! I'm Azure OpenAI."
            assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, azure_provider):
        """Test streaming chat completion."""

        # Mock streaming chunks
        async def mock_stream():
            chunks = [
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Hello"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "delta": {"content": " world!"}, "finish_reason": "stop"}
                    ],
                },
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(azure_provider, "_make_request", return_value=mock_stream()):
            stream = await azure_provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini", stream=True
            )

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].choices[0].delta.content == "Hello"
            assert chunks[1].choices[0].delta.content == " world!"

    @pytest.mark.asyncio
    async def test_vision_model_validation(self, azure_provider):
        """Test vision model validation."""
        # Test with vision-capable model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]

        # Should not raise for vision model
        processed = azure_provider._process_messages_for_vision(messages, "gpt-4o")
        assert len(processed) == 1

        # Should raise for non-vision model
        with pytest.raises(InvalidRequestError) as exc_info:
            azure_provider._process_messages_for_vision(messages, "gpt-3.5-turbo")
        assert "does not support vision inputs" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embedding(self, azure_provider):
        """Test embedding creation."""
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        with patch.object(azure_provider, "_make_request", return_value=mock_response):
            response = await azure_provider.create_embedding(
                input="Test text", model="text-embedding-ada-002"
            )

            assert isinstance(response, EmbeddingResponse)
            assert len(response.data) == 1
            assert len(response.data[0].embedding) == 5

    @pytest.mark.asyncio
    async def test_file_operations_not_supported(self, azure_provider):
        """Test that file operations raise appropriate errors."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await azure_provider.upload_file("test.txt", "assistants")
        assert "does not support file uploads" in str(exc_info.value)

        with pytest.raises(InvalidRequestError) as exc_info:
            await azure_provider.download_file("file-123")
        assert "does not support file downloads" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcription(self, azure_provider):
        """Test audio transcription."""
        mock_response = {
            "text": "This is a test transcription.",
            "task": "transcribe",
            "language": "en",
            "duration": 5.0,
        }

        with patch.object(azure_provider, "_make_request", return_value=mock_response):
            # Test with file path
            with patch("builtins.open", MagicMock()):
                result = await azure_provider.create_transcription(
                    file="test.mp3", model="whisper-1"
                )

                assert result.text == "This is a test transcription."
                assert result.language == "en"

    @pytest.mark.asyncio
    async def test_image_generation(self, azure_provider):
        """Test image generation."""
        mock_response = {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A beautiful sunset over mountains",
                }
            ],
        }

        with patch.object(azure_provider, "_make_request", return_value=mock_response):
            result = await azure_provider.create_image(
                prompt="A sunset over mountains", model="dall-e-3", size="1024x1024"
            )

            assert result.created == 1234567890
            assert len(result.data) == 1
            assert result.data[0]["url"] == "https://example.com/image.png"

    @pytest.mark.asyncio
    async def test_error_handling(self, azure_provider):
        """Test error response handling."""
        error_response = MagicMock()
        error_response.status = 401
        error_response.json = AsyncMock(
            return_value={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            }
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await azure_provider._handle_response(error_response)
        assert "Invalid API key" in str(exc_info.value)

    def test_capability_flags(self):
        """Test that capability flags are set correctly."""
        assert AzureProvider.json_mode_support is True
        assert AzureProvider.vision_support is True
        assert AzureProvider.streaming_support is True
        assert AzureProvider.token_by_token_support is True
        assert AzureProvider.audio_input_support is False
        assert AzureProvider.video_input_support is False
        assert AzureProvider.realtime_support is False

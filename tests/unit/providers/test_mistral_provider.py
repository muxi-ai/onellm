#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the Mistral AI provider implementation.

These tests verify that the Mistral provider correctly handles various request types
and formats responses appropriately.
"""

import pytest
import mock
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from onellm.providers import get_provider
from onellm.providers.mistral import MistralProvider
from onellm.errors import AuthenticationError, InvalidRequestError
from onellm import config as onellm_config


@pytest.fixture(autouse=True)
def reset_mistral_config():
    """Reset mistral config between tests to avoid state pollution."""
    # Store original config
    original = onellm_config.get_provider_config("mistral").copy()
    yield
    # Restore original config
    onellm_config.config["providers"]["mistral"] = original


class MockResponse:
    """Mock aiohttp response object."""

    def __init__(self, status: int, data: Dict[str, Any]):
        self.status = status
        self._data = data

    async def json(self):
        return self._data

    async def read(self):
        return b"test data"

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Set a mock Mistral API key environment variable."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock for aiohttp.ClientSession."""
    with mock.patch("aiohttp.ClientSession") as mock_session:
        # Create a response for chat_completion
        chat_response = MockResponse(
            status=200,
            data={
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response from Mistral",
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
            },
        )

        # Create session instance mock
        session_instance = AsyncMock()
        # session.request() returns a context manager that yields chat_response
        session_instance.request.return_value.__aenter__.return_value = chat_response

        # Set up the ClientSession constructor to work as a context manager
        mock_session.return_value.__aenter__.return_value = session_instance
        mock_session.return_value.__aexit__.return_value = None

        yield mock_session


class TestMistralProvider:
    """Tests for the Mistral provider."""

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization fails with no API key."""
        # Clear the environment variable to ensure no API key is present
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        # Directly patch the config module to remove any API key
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            # Create a configuration with no API key
            mock_get_config.return_value = {}

            # Should raise AuthenticationError
            with pytest.raises(AuthenticationError) as exc_info:
                MistralProvider()

            # Check error message
            assert "Mistral API key is required" in str(exc_info.value)

    def test_init_with_api_key(self, mock_env_api_key):
        """Test successful initialization with API key."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()
            assert provider.api_key == "test-key"
            assert provider.api_base == "https://api.mistral.ai/v1"

    def test_provider_registration(self):
        """Test that the Mistral provider is properly registered."""
        provider = get_provider("mistral")
        assert isinstance(provider, MistralProvider)

    def test_capability_flags(self):
        """Test that the provider has correct capability flags."""
        provider = get_provider("mistral")

        # Check supported capabilities
        assert provider.json_mode_support is True
        assert provider.vision_support is True
        assert provider.streaming_support is True
        assert provider.token_by_token_support is True

        # Check unsupported capabilities
        assert provider.audio_input_support is False
        assert provider.video_input_support is False
        assert provider.realtime_support is False

    def test_headers(self, mock_env_api_key):
        """Test that headers are correctly formatted."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()
            headers = provider._get_headers()

            assert headers["Content-Type"] == "application/json"
            assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_env_api_key, mock_aiohttp_session):
        """Test chat completion functionality."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()

            messages = [{"role": "user", "content": "Hello, how are you?"}]

            response = await provider.create_chat_completion(
                messages=messages, model="mistral-large-latest"
            )

            # Verify response structure
            assert response.id == "test-id"
            assert response.model == "mistral-large-latest"
            assert len(response.choices) == 1
            assert response.choices[0].message["content"] == "This is a test response from Mistral"
            assert response.usage["total_tokens"] == 35

    @pytest.mark.asyncio
    async def test_completion(self, mock_env_api_key, mock_aiohttp_session):
        """Test text completion functionality."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()

            # Mock completion response
            completion_response = MockResponse(
                status=200,
                data={
                    "id": "test-completion-id",
                    "object": "text_completion",
                    "created": 1677858242,
                    "model": "mistral-large-latest",
                    "choices": [
                        {
                            "text": "This is a completion response",
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20},
                },
            )

            # Update mock to return completion response
            mock_aiohttp_session.return_value.__aenter__.return_value.request.return_value = (
                completion_response
            )

            response = await provider.create_completion(
                prompt="Complete this text:", model="mistral-large-latest"
            )

            # Verify response structure
            assert response.id == "test-completion-id"
            assert response.model == "mistral-large-latest"
            assert len(response.choices) == 1
            assert response.choices[0].text == "This is a completion response"

    @pytest.mark.asyncio
    async def test_embedding(self, mock_env_api_key, mock_aiohttp_session):
        """Test embedding functionality."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()

            # Mock embedding response
            embedding_response = MockResponse(
                status=200,
                data={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}
                    ],
                    "model": "mistral-embed",
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )

            # Update mock to return embedding response
            mock_aiohttp_session.return_value.__aenter__.return_value.request.return_value = (
                embedding_response
            )

            response = await provider.create_embedding(
                input="Test text to embed", model="mistral-embed"
            )

            # Verify response structure
            assert response.object == "list"
            assert len(response.data) == 1
            assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert response.model == "mistral-embed"

    def test_vision_model_validation(self, mock_env_api_key):
        """Test vision model validation."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key"}

            provider = MistralProvider()

            # Test with non-vision model and image content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="  # noqa: E501
                            },
                        },
                    ],
                }
            ]

            # Should raise InvalidRequestError for non-vision model
            with pytest.raises(InvalidRequestError) as exc_info:
                provider._process_messages_for_vision(messages, "mistral-large-latest")

            assert "does not support vision inputs" in str(exc_info.value)

            # Should work fine with vision model
            try:
                processed = provider._process_messages_for_vision(messages, "pixtral-12b-2409")
                assert len(processed) == 1
            except InvalidRequestError:
                pytest.fail("Should not raise error for vision model")

    def test_api_base_configuration(self, mock_env_api_key):
        """Test that API base URL can be configured."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "test-key",
                "api_base": "https://custom.mistral.ai/v1",
            }

            provider = MistralProvider()
            assert provider.api_base == "https://custom.mistral.ai/v1"

    def test_timeout_configuration(self, mock_env_api_key):
        """Test that timeout can be configured."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key", "timeout": 60.0}

            provider = MistralProvider()
            assert provider.timeout == 60.0

    def test_max_retries_configuration(self, mock_env_api_key):
        """Test that max retries can be configured."""
        with patch("onellm.providers.mistral.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-key", "max_retries": 5}

            provider = MistralProvider()
            assert provider.max_retries == 5
            assert provider.retry_config.max_retries == 5


if __name__ == "__main__":
    pytest.main([__file__])

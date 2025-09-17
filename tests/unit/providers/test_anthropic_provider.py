#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the Anthropic provider implementation.

These tests verify that the Anthropic provider correctly handles various request types,
converts between OpenAI and Anthropic formats, and manages unique Anthropic features.
"""

import pytest
import mock
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from onellm.providers import get_provider
from onellm.providers.anthropic import AnthropicProvider
from onellm.errors import AuthenticationError, InvalidRequestError


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
    """Set a mock Anthropic API key environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock for aiohttp.ClientSession."""
    with mock.patch("aiohttp.ClientSession") as mock_session:
        # Create a session instance
        session_instance = AsyncMock()

        # Create a response for messages endpoint (Anthropic native format)
        anthropic_response = MockResponse(
            status=200,
            data={
                "id": "msg_test-id",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test response from Claude"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 25},
            },
        )

        # Set up request to return our mock response
        request_context = AsyncMock()
        request_context.__aenter__.return_value = anthropic_response
        request_context.__aexit__.return_value = None
        session_instance.request = MagicMock(return_value=request_context)

        # Set up the ClientSession constructor to work as a context manager
        mock_session.return_value = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        mock_session.return_value.__aexit__.return_value = None

        yield mock_session


class TestAnthropicProvider:
    """Tests for the Anthropic provider."""

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization fails with no API key."""
        # Clear the environment variable to ensure no API key is present
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Directly patch the config module to remove any API key
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            # Create a configuration with no API key
            mock_get_config.return_value = {}

            # Should raise AuthenticationError
            with pytest.raises(AuthenticationError) as exc_info:
                AnthropicProvider()

            # Check error message
            assert "Anthropic API key is required" in str(exc_info.value)

    def test_init_with_api_key(self, mock_env_api_key):
        """Test successful initialization with API key."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()
            assert provider.api_key == "sk-ant-test-key"
            assert provider.api_base == "https://api.anthropic.com/v1"

    def test_provider_registration(self, mock_env_api_key):
        """Test that the Anthropic provider is properly registered."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "sk-ant-test-key",
                "api_base": "https://api.anthropic.com/v1",
                "timeout": 60.0,
                "max_retries": 3,
            }

            provider = get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_capability_flags(self, mock_env_api_key):
        """Test that the provider has correct capability flags."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "sk-ant-test-key",
                "api_base": "https://api.anthropic.com/v1",
                "timeout": 60.0,
                "max_retries": 3,
            }

            provider = get_provider("anthropic")

        # Check supported capabilities
        assert provider.vision_support is True
        assert provider.streaming_support is True
        assert provider.token_by_token_support is True

        # Check unsupported capabilities
        assert provider.json_mode_support is False  # Anthropic doesn't have explicit JSON mode
        assert provider.audio_input_support is False
        assert provider.video_input_support is False
        assert provider.realtime_support is False

    def test_headers(self, mock_env_api_key):
        """Test that headers are correctly formatted for Anthropic API."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()
            headers = provider._get_headers()

            assert headers["Content-Type"] == "application/json"
            assert headers["x-api-key"] == "sk-ant-test-key"
            assert headers["anthropic-version"] == "2023-06-01"

    def test_convert_openai_to_anthropic_messages(self, mock_env_api_key):
        """Test conversion from OpenAI message format to Anthropic format."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            # Test simple text messages
            openai_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]

            anthropic_messages = provider._convert_openai_to_anthropic_messages(openai_messages)

            # System message should be filtered out (handled separately)
            assert len(anthropic_messages) == 2
            assert anthropic_messages[0]["role"] == "user"
            assert anthropic_messages[0]["content"] == "Hello, how are you?"
            assert anthropic_messages[1]["role"] == "assistant"
            assert anthropic_messages[1]["content"] == "I'm doing well, thank you!"

    def test_extract_system_message(self, mock_env_api_key):
        """Test extraction of system message from OpenAI messages."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            messages_with_system = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]

            system_message = provider._extract_system_message(messages_with_system)
            assert system_message == "You are a helpful assistant."

            messages_without_system = [{"role": "user", "content": "Hello!"}]

            system_message = provider._extract_system_message(messages_without_system)
            assert system_message is None

    def test_convert_complex_content(self, mock_env_api_key):
        """Test conversion of complex content with images."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            # Test complex content with image
            openai_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD=="
                            },
                        },
                    ],
                }
            ]

            anthropic_messages = provider._convert_openai_to_anthropic_messages(openai_messages)

            assert len(anthropic_messages) == 1
            assert anthropic_messages[0]["role"] == "user"
            assert isinstance(anthropic_messages[0]["content"], list)
            assert len(anthropic_messages[0]["content"]) == 2

            # Check text content
            assert anthropic_messages[0]["content"][0]["type"] == "text"
            assert anthropic_messages[0]["content"][0]["text"] == "What's in this image?"

            # Check image content
            assert anthropic_messages[0]["content"][1]["type"] == "image"
            assert anthropic_messages[0]["content"][1]["source"]["type"] == "base64"
            assert anthropic_messages[0]["content"][1]["source"]["media_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_env_api_key, mock_aiohttp_session):
        """Test chat completion functionality."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            messages = [{"role": "user", "content": "Hello, how are you?"}]

            response = await provider.create_chat_completion(
                messages=messages, model="claude-3-5-sonnet-20241022", max_tokens=1000
            )

            # Verify response structure (converted to OpenAI format)
            assert response.id == "msg_test-id"
            assert response.model == "claude-3-5-sonnet-20241022"
            assert len(response.choices) == 1
            assert response.choices[0].message["content"] == "This is a test response from Claude"
            assert response.usage["total_tokens"] == 35  # 10 + 25
            assert response.usage["prompt_tokens_cached"] == 0
            assert response.usage["prompt_tokens_uncached"] == 10
            assert response.usage["completion_tokens_cached"] == 0
            assert response.usage["completion_tokens_uncached"] == 25

    @pytest.mark.asyncio
    async def test_completion_converts_to_chat(self, mock_env_api_key, mock_aiohttp_session):
        """Test that completion gets converted to chat completion."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            response = await provider.create_completion(
                prompt="Complete this text:", model="claude-3-5-sonnet-20241022"
            )

            # Verify response structure (completion format)
            assert response.id == "msg_test-id"
            assert response.object == "text_completion"
            assert len(response.choices) == 1
            assert response.choices[0].text == "This is a test response from Claude"

    @pytest.mark.asyncio
    async def test_embedding_not_supported(self, mock_env_api_key):
        """Test that embedding raises appropriate error."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            with pytest.raises(InvalidRequestError) as exc_info:
                await provider.create_embedding(
                    input="Test text to embed", model="claude-3-5-sonnet-20241022"
                )

            assert "does not provide embedding models" in str(exc_info.value)

    def test_anthropic_to_openai_response_conversion(self, mock_env_api_key):
        """Test conversion from Anthropic response to OpenAI format."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            anthropic_response = {
                "id": "msg_test",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }

            openai_response = provider._convert_anthropic_to_openai_response(
                anthropic_response, "claude-3-5-sonnet-20241022"
            )

            assert openai_response.id == "msg_test"
            assert openai_response.choices[0].message["content"] == "Hello from Claude!"
            assert openai_response.choices[0].finish_reason == "end_turn"
            assert openai_response.usage["prompt_tokens"] == 5
            assert openai_response.usage["prompt_tokens_cached"] == 0
            assert openai_response.usage["prompt_tokens_uncached"] == 5
            assert openai_response.usage["completion_tokens"] == 10
            assert openai_response.usage["completion_tokens_cached"] == 0
            assert openai_response.usage["completion_tokens_uncached"] == 10
            assert openai_response.usage["total_tokens"] == 15

    def test_anthropic_usage_with_cache_metrics(self, mock_env_api_key):
        """Usage normalization should propagate cache hit metrics."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            anthropic_response = {
                "id": "msg_cached",
                "content": [{"type": "text", "text": "Hello from cache"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 30,
                },
            }

            openai_response = provider._convert_anthropic_to_openai_response(
                anthropic_response, "claude-3-5-sonnet-20241022"
            )

            assert openai_response.usage["prompt_tokens"] == 50
            assert openai_response.usage["prompt_tokens_cached"] == 20
            assert openai_response.usage["prompt_tokens_uncached"] == 30
            assert openai_response.usage["completion_tokens"] == 5
            assert openai_response.usage["completion_tokens_cached"] == 0
            assert openai_response.usage["completion_tokens_uncached"] == 5
            assert openai_response.usage["total_tokens"] == 55

    def test_api_base_configuration(self, mock_env_api_key):
        """Test that API base URL can be configured."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "sk-ant-test-key",
                "api_base": "https://custom.anthropic.com/v1",
            }

            provider = AnthropicProvider()
            assert provider.api_base == "https://custom.anthropic.com/v1"

    def test_timeout_configuration(self, mock_env_api_key):
        """Test that timeout can be configured."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key", "timeout": 60.0}

            provider = AnthropicProvider()
            assert provider.timeout == 60.0

    def test_max_retries_configuration(self, mock_env_api_key):
        """Test that max retries can be configured."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key", "max_retries": 5}

            provider = AnthropicProvider()
            assert provider.max_retries == 5
            assert provider.retry_config.max_retries == 5

    @pytest.mark.asyncio
    async def test_chat_completion_with_thinking(self, mock_env_api_key, mock_aiohttp_session):
        """Test chat completion with Anthropic's thinking feature."""
        with patch("onellm.providers.anthropic.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "sk-ant-test-key"}

            provider = AnthropicProvider()

            messages = [{"role": "user", "content": "Solve this math problem: 2 + 2"}]

            # Mock request to capture the data sent
            with patch.object(provider, "_make_request") as mock_request:
                mock_request.return_value = {
                    "id": "msg_thinking_test",
                    "content": [{"type": "text", "text": "The answer is 4"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 8},
                }

                await provider.create_chat_completion(
                    messages=messages,
                    model="claude-opus-4-20250514",
                    max_tokens=1000,
                    thinking={"enabled": True, "budget_tokens": 20000},
                )

                # Verify thinking parameter was passed through
                call_args = mock_request.call_args
                assert call_args[1]["data"]["thinking"] == {"enabled": True, "budget_tokens": 20000}


if __name__ == "__main__":
    pytest.main([__file__])

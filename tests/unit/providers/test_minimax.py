#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the MiniMax provider implementation.

These tests verify that the MiniMax provider correctly uses the Anthropic-compatible
interface and properly configures the API endpoint for MiniMax's API.
"""

import pytest
import mock
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from onellm.providers import get_provider
from onellm.providers.minimax import MinimaxProvider
from onellm.providers.anthropic_compatible import AnthropicCompatibleProvider
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
    """Set a mock MiniMax API key environment variable."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock for get_session_safe to return a mock session."""
    with mock.patch("onellm.providers.anthropic.get_session_safe") as mock_get_session:
        # Create a session instance
        session_instance = MagicMock()
        session_instance.close = AsyncMock()

        # Create a response for messages endpoint (Anthropic-compatible format)
        minimax_response = MockResponse(
            status=200,
            data={
                "id": "msg_minimax-test-id",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test response from MiniMax-M2"}],
                "model": "MiniMax-M2",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 30},
            },
        )

        # Set up request to return our mock response
        request_context = AsyncMock()
        request_context.__aenter__.return_value = minimax_response
        request_context.__aexit__.return_value = None
        session_instance.request = MagicMock(return_value=request_context)

        # Set up get_session_safe to return (session, is_pooled) tuple
        mock_get_session.return_value = (session_instance, False)

        yield mock_get_session


class TestMinimaxProvider:
    """Tests for the MiniMax provider."""

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization fails with no API key."""
        # Clear the environment variable to ensure no API key is present
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

        # Directly patch the config module to remove any API key
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            # Create a configuration with no API key
            mock_get_config.return_value = {}

            # Should raise AuthenticationError
            with pytest.raises(AuthenticationError) as exc_info:
                MinimaxProvider()

            # Check error message mentions MiniMax
            assert "Minimax API key is required" in str(exc_info.value)
            assert "MINIMAX_API_KEY" in str(exc_info.value)

    def test_init_with_api_key(self, mock_env_api_key):
        """Test successful initialization with API key."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()
            assert provider.api_key == "test-minimax-key"
            assert provider.api_base == "https://api.minimax.io/anthropic"
            assert provider.provider_name == "minimax"

    def test_inherits_from_anthropic_compatible(self):
        """Test that MinimaxProvider inherits from AnthropicCompatibleProvider."""
        assert issubclass(MinimaxProvider, AnthropicCompatibleProvider)

    def test_provider_registration(self, mock_env_api_key):
        """Test that the MiniMax provider is properly registered."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "test-minimax-key",
                "api_base": "https://api.minimax.io/anthropic",
                "timeout": 30.0,
                "max_retries": 3,
            }

            provider = get_provider("minimax")
            assert isinstance(provider, MinimaxProvider)

    def test_capability_flags(self, mock_env_api_key):
        """Test that the provider has correct capability flags."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "test-minimax-key",
                "api_base": "https://api.minimax.io/anthropic",
                "timeout": 30.0,
                "max_retries": 3,
            }

            provider = get_provider("minimax")

        # Check supported capabilities
        assert provider.streaming_support is True
        assert provider.token_by_token_support is True
        assert provider.thinking_support is True
        assert provider.tool_calling_support is True

        # Check unsupported capabilities
        assert provider.json_mode_support is False
        assert provider.vision_support is False  # MiniMax M2 doesn't support vision yet
        assert provider.audio_input_support is False
        assert provider.video_input_support is False
        assert provider.realtime_support is False

    def test_headers(self, mock_env_api_key):
        """Test that headers are correctly formatted for MiniMax API."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()
            headers = provider._get_headers()

            # MiniMax uses Anthropic-compatible headers
            assert headers["Content-Type"] == "application/json"
            assert headers["x-api-key"] == "test-minimax-key"
            assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_env_api_key, mock_aiohttp_session):
        """Test chat completion functionality."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()

            messages = [{"role": "user", "content": "Hello, how are you?"}]

            response = await provider.create_chat_completion(
                messages=messages, model="MiniMax-M2", max_tokens=1000
            )

            # Verify response structure (converted to OpenAI format)
            assert response.id == "msg_minimax-test-id"
            assert response.model == "MiniMax-M2"
            assert len(response.choices) == 1
            assert response.choices[0].message["content"] == "This is a test response from MiniMax-M2"
            assert response.usage["total_tokens"] == 42  # 12 + 30

    @pytest.mark.asyncio
    async def test_completion_converts_to_chat(self, mock_env_api_key, mock_aiohttp_session):
        """Test that completion gets converted to chat completion."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()

            response = await provider.create_completion(
                prompt="Complete this text:", model="MiniMax-M2"
            )

            # Verify response structure (completion format)
            assert response.id == "msg_minimax-test-id"
            assert response.object == "text_completion"
            assert len(response.choices) == 1
            assert response.choices[0].text == "This is a test response from MiniMax-M2"

    @pytest.mark.asyncio
    async def test_embedding_not_supported(self, mock_env_api_key):
        """Test that embedding raises appropriate error."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()

            with pytest.raises(InvalidRequestError) as exc_info:
                await provider.create_embedding(input="Test text to embed", model="MiniMax-M2")

            # Inherited from Anthropic provider
            assert "does not provide embedding models" in str(exc_info.value)

    def test_api_base_default(self, mock_env_api_key):
        """Test that API base URL defaults to MiniMax endpoint."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()
            assert provider.api_base == "https://api.minimax.io/anthropic"

    def test_api_base_configuration(self, mock_env_api_key):
        """Test that API base URL can be overridden (e.g., for China users)."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {
                "api_key": "test-minimax-key",
                "api_base": "https://api.minimaxi.com/anthropic",  # China endpoint
            }

            provider = MinimaxProvider()
            assert provider.api_base == "https://api.minimaxi.com/anthropic"

    def test_timeout_configuration(self, mock_env_api_key):
        """Test that timeout can be configured."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key", "timeout": 60.0}

            provider = MinimaxProvider()
            assert provider.timeout == 60.0

    def test_max_retries_configuration(self, mock_env_api_key):
        """Test that max retries can be configured."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key", "max_retries": 5}

            provider = MinimaxProvider()
            assert provider.max_retries == 5
            assert provider.retry_config.max_retries == 5

    @pytest.mark.asyncio
    async def test_chat_completion_with_thinking(self, mock_env_api_key, mock_aiohttp_session):
        """Test chat completion with MiniMax's interleaved thinking feature."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()

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
                    model="MiniMax-M2",
                    max_tokens=1000,
                    thinking={"enabled": True, "budget_tokens": 20000},
                )

                # Verify thinking parameter was passed through
                call_args = mock_request.call_args
                assert call_args[1]["data"]["thinking"] == {"enabled": True, "budget_tokens": 20000}

    @pytest.mark.asyncio
    async def test_supports_minimax_m2_stable(self, mock_env_api_key, mock_aiohttp_session):
        """Test that MiniMax-M2-Stable model is supported."""
        with patch("onellm.providers.anthropic_compatible.get_provider_config") as mock_get_config:
            mock_get_config.return_value = {"api_key": "test-minimax-key"}

            provider = MinimaxProvider()

            messages = [{"role": "user", "content": "Test message"}]

            # Mock the response with M2-Stable model
            with patch.object(provider, "_make_request") as mock_request:
                mock_request.return_value = {
                    "id": "msg_stable_test",
                    "content": [{"type": "text", "text": "Response from M2-Stable"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 8},
                }

                response = await provider.create_chat_completion(
                    messages=messages, model="MiniMax-M2-Stable", max_tokens=1000
                )

                # Verify the model name was passed correctly
                call_args = mock_request.call_args
                assert call_args[1]["data"]["model"] == "MiniMax-M2-Stable"


if __name__ == "__main__":
    pytest.main([__file__])

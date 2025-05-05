"""
Tests for the OpenAI provider.
"""

import os
import pytest
import mock
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from muxi.llm.providers import get_provider
from muxi.llm.providers.openai import OpenAIProvider
from muxi.llm.errors import AuthenticationError
from muxi.llm.config import get_provider_config


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
    """Set a mock OpenAI API key environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock for aiohttp.ClientSession."""
    with mock.patch("aiohttp.ClientSession") as mock_session:
        # Create a session instance
        session_instance = AsyncMock()

        # Create a response for chat_completion
        chat_response = MockResponse(
            status=200,
            data={
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response"
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        )

        # Set up request to return our mock response
        session_instance.request = AsyncMock(return_value=chat_response)

        # Set up the ClientSession constructor to work as a context manager
        mock_session.return_value = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        mock_session.return_value.__aexit__.return_value = None

        yield mock_session


class TestOpenAIProvider:
    """Tests for the OpenAI provider."""

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization fails with no API key."""
        # Clear the environment variable to ensure no API key is present
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Directly patch the config module to remove any API key
        with mock.patch("muxi.llm.config.config", {
            "providers": {
                "openai": {
                    "api_key": None,
                    "api_base": "https://api.openai.com/v1",
                    "organization_id": None,
                    "timeout": 60,
                    "max_retries": 3
                }
            }
        }):
            # Verify the config no longer has an API key
            # Use import within the function to avoid circular imports
            from muxi.llm.config import get_provider_config as get_config
            config = get_config("openai")
            print(f"DEBUG: OpenAI config after patch: {config}")

            # Test that initialization fails without an API key
            with pytest.raises(AuthenticationError):
                OpenAIProvider()

    def test_init_with_api_key(self):
        """Test initialization with API key in kwargs."""
        provider = OpenAIProvider(api_key="sk-test-key")
        assert provider.api_key == "sk-test-key"

    def test_init_with_env_api_key(self, mock_env_api_key):
        """Test initialization with API key from environment."""
        provider = OpenAIProvider()
        assert provider.api_key == "sk-test-key"

    def test_get_headers(self):
        """Test get_headers method."""
        provider = OpenAIProvider(api_key="sk-test-key")
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_with_organization(self):
        """Test get_headers method with organization ID."""
        provider = OpenAIProvider(api_key="sk-test-key", organization_id="org-123")
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["OpenAI-Organization"] == "org-123"

    def test_get_provider_factory(self):
        """Test provider factory function."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            provider = get_provider("openai")
            assert isinstance(provider, OpenAIProvider)

    @pytest.mark.asyncio
    async def test_create_chat_completion(self):
        """Test create_chat_completion method."""
        # Create a mock response
        mock_response = MockResponse(
            status=200,
            data={
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response"
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        )

        # Create a mock request function
        mock_request = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.request = mock_request

        # Patch the context manager in the _make_request method
        with patch("aiohttp.ClientSession") as mock_client_session:
            # Setup the async context manager mock
            mock_client_session.return_value.__aenter__.return_value = mock_session

            # Call the method
            provider = OpenAIProvider(api_key="sk-test-key")
            response = await provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )

            # Verify response parsing
            assert response.choices[0].message["content"] == "This is a test response"
            assert response.choices[0].finish_reason == "stop"

            # Verify request was made with correct parameters
            call_args = mock_request.call_args
            assert call_args is not None
            args, kwargs = call_args

            assert kwargs["method"] == "POST"
            assert kwargs["url"] == "https://api.openai.com/v1/chat/completions"
            assert "messages" in kwargs["json"]
            assert kwargs["json"]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_create_embedding(self):
        """Test create_embedding method."""
        # Create a mock response for embeddings
        mock_response = MockResponse(
            status=200,
            data={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": 0
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        )

        # Create a mock request function
        mock_request = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.request = mock_request

        # Patch the context manager in the _make_request method
        with patch("aiohttp.ClientSession") as mock_client_session:
            # Setup the async context manager mock
            mock_client_session.return_value.__aenter__.return_value = mock_session

            # Call the method
            provider = OpenAIProvider(api_key="sk-test-key")
            response = await provider.create_embedding(
                input="Hello, world",
                model="text-embedding-ada-002"
            )

            # Verify response parsing
            assert response.data[0].embedding == [0.1, 0.2, 0.3]

            # Verify request was made with correct parameters
            call_args = mock_request.call_args
            assert call_args is not None
            args, kwargs = call_args

            assert kwargs["method"] == "POST"
            assert kwargs["url"] == "https://api.openai.com/v1/embeddings"
            assert kwargs["json"]["input"] == "Hello, world"
            assert kwargs["json"]["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        # Create an error response
        mock_response = MockResponse(
            status=401,
            data={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )

        # Create a mock request function
        mock_request = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.request = mock_request

        # Patch the context manager in the _make_request method
        with patch("aiohttp.ClientSession") as mock_client_session:
            # Setup the async context manager mock
            mock_client_session.return_value.__aenter__.return_value = mock_session

            # Call the method and expect an error
            provider = OpenAIProvider(api_key="sk-test-key")
            with pytest.raises(AuthenticationError) as excinfo:
                await provider.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo"
                )

            # Verify error details
            assert "Invalid API key" in str(excinfo.value)
            assert excinfo.value.status_code == 401

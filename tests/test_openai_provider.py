"""
Tests for the OpenAI provider.
"""

import os
import pytest
import mock
from typing import Dict, Any

import aiohttp
from aiohttp.test_utils import make_mocked_coro

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
        # Setup the session to properly support context manager protocol with async
        session_instance = mock.MagicMock()
        mock_session.return_value.__aenter__.return_value = session_instance

        # Add request method that returns a proper response
        response = MockResponse(
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
        session_instance.request = make_mocked_coro(response)
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
            from muxi.llm.config import get_provider_config
            config = get_provider_config("openai")
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
    async def test_create_chat_completion(self, mock_aiohttp_session):
        """Test create_chat_completion method."""
        provider = OpenAIProvider(api_key="sk-test-key")
        response = await provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )

        # Verify response parsing
        assert response.choices[0].message["content"] == "This is a test response"
        assert response.choices[0].finish_reason == "stop"

        # Verify request was made with correct parameters
        mock_request = mock_aiohttp_session.return_value.__aenter__.return_value.request
        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "https://api.openai.com/v1/chat/completions"
        assert "messages" in kwargs["data"]
        assert kwargs["data"]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_create_embedding(self, mock_aiohttp_session):
        """Test create_embedding method."""
        # Create a mock response specific to embeddings
        mock_aiohttp_session.return_value.__aenter__.return_value.request = make_mocked_coro(
            MockResponse(
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
        )

        provider = OpenAIProvider(api_key="sk-test-key")
        response = await provider.create_embedding(
            input="Hello, world",
            model="text-embedding-ada-002"
        )

        # Verify response parsing
        assert response.data[0].embedding == [0.1, 0.2, 0.3]

        # Verify request was made with correct parameters
        mock_request = mock_aiohttp_session.return_value.__aenter__.return_value.request
        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "https://api.openai.com/v1/embeddings"
        assert kwargs["data"]["input"] == "Hello, world"
        assert kwargs["data"]["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_aiohttp_session):
        """Test error handling."""
        # Mock an error response
        mock_aiohttp_session.return_value.__aenter__.return_value.request = make_mocked_coro(
            MockResponse(
                status=401,
                data={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key"
                    }
                }
            )
        )

        provider = OpenAIProvider(api_key="sk-test-key")

        # Test that the correct error is raised
        with pytest.raises(AuthenticationError) as excinfo:
            await provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )

        # Verify error details
        assert "Invalid API key" in str(excinfo.value)
        assert excinfo.value.status_code == 401

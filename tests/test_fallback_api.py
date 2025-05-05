"""
Tests for the fallback mechanism at the API level in muxi-llm.
"""

import pytest
from unittest.mock import MagicMock, patch

from muxi.llm import ChatCompletion, Completion, Embedding
from muxi.llm.errors import RateLimitError


async def async_return(value):
    """Helper to create a coroutine that returns a value."""
    return value


class TestFallbackAPI:
    """Tests for the fallback mechanism at the API level."""

    def test_chat_completion_with_fallbacks(self):
        """Test ChatCompletion with fallbacks."""
        # Create a mock for get_provider_with_fallbacks
        mock_provider = MagicMock()
        mock_provider.create_chat_completion.return_value = async_return({"result": "success"})

        with patch("muxi.llm.chat_completion.get_provider_with_fallbacks") as mock_get_provider:
            # Return a mocked provider and a model name
            mock_get_provider.return_value = (mock_provider, "gpt-4")

            # Call the API with fallback models
            result = ChatCompletion.create(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_models=["anthropic/claude-3", "google/gemini-pro"]
            )

            # Verify get_provider_with_fallbacks was called with the correct parameters
            mock_get_provider.assert_called_once_with(
                primary_model="openai/gpt-4",
                fallback_models=["anthropic/claude-3", "google/gemini-pro"],
                fallback_config=None
            )

            # Verify the provider's create_chat_completion method was called
            mock_provider.create_chat_completion.assert_called_once()

            # Verify we got the expected result
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_chat_completion_async_with_fallbacks(self):
        """Test ChatCompletion.acreate with fallbacks."""
        # Create a mock for get_provider_with_fallbacks
        mock_provider = MagicMock()
        # Create an awaitable mock result
        mock_provider.create_chat_completion.return_value = async_return({"result": "success"})

        with patch("muxi.llm.chat_completion.get_provider_with_fallbacks") as mock_get_provider:
            # Return a mocked provider and a model name
            mock_get_provider.return_value = (mock_provider, "gpt-4")

            # Call the API with fallback models
            result = await ChatCompletion.acreate(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_models=["anthropic/claude-3", "google/gemini-pro"],
                fallback_config={"max_fallbacks": 2}
            )

            # Verify get_provider_with_fallbacks was called with the correct parameters
            mock_get_provider.assert_called_once()
            args, kwargs = mock_get_provider.call_args
            assert kwargs["primary_model"] == "openai/gpt-4"
            assert kwargs["fallback_models"] == ["anthropic/claude-3", "google/gemini-pro"]
            assert kwargs["fallback_config"] is not None

            # Verify the provider's create_chat_completion method was called
            mock_provider.create_chat_completion.assert_called_once()

            # Verify we got the expected result
            assert result == {"result": "success"}

    def test_completion_with_fallbacks(self):
        """Test Completion with fallbacks."""
        # Create a mock for get_provider_with_fallbacks
        mock_provider = MagicMock()
        mock_provider.create_completion.return_value = async_return({"result": "success"})

        with patch("muxi.llm.completion.get_provider_with_fallbacks") as mock_get_provider:
            # Return a mocked provider and a model name
            mock_get_provider.return_value = (mock_provider, "text-davinci-003")

            # Call the API with fallback models
            result = Completion.create(
                model="openai/text-davinci-003",
                prompt="Hello",
                fallback_models=["anthropic/claude-instant-1"]
            )

            # Verify get_provider_with_fallbacks was called with the correct parameters
            mock_get_provider.assert_called_once_with(
                primary_model="openai/text-davinci-003",
                fallback_models=["anthropic/claude-instant-1"],
                fallback_config=None
            )

            # Verify the provider's create_completion method was called
            mock_provider.create_completion.assert_called_once()

            # Verify we got the expected result
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_embedding_async_with_fallbacks(self):
        """Test Embedding.acreate with fallbacks."""
        # Create a mock for get_provider_with_fallbacks
        mock_provider = MagicMock()
        # Create an awaitable mock result
        mock_provider.create_embedding.return_value = async_return({"result": "success"})

        with patch("muxi.llm.embedding.get_provider_with_fallbacks") as mock_get_provider:
            # Return a mocked provider and a model name
            mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

            # Call the API with fallback models
            result = await Embedding.acreate(
                model="openai/text-embedding-ada-002",
                input="Hello",
                fallback_models=["cohere/embed-english-v3.0"]
            )

            # Verify get_provider_with_fallbacks was called with the correct parameters
            mock_get_provider.assert_called_once_with(
                primary_model="openai/text-embedding-ada-002",
                fallback_models=["cohere/embed-english-v3.0"],
                fallback_config=None
            )

            # Verify the provider's create_embedding method was called
            mock_provider.create_embedding.assert_called_once()

            # Verify we got the expected result
            assert result == {"result": "success"}

    def test_integration_fallback_config(self):
        """Test passing fallback_config to API methods."""
        # Create a mock for get_provider_with_fallbacks
        mock_provider = MagicMock()
        mock_provider.create_chat_completion.return_value = async_return({"result": "success"})

        with patch("muxi.llm.chat_completion.get_provider_with_fallbacks") as mock_get_provider:
            # Return a mocked provider and a model name
            mock_get_provider.return_value = (mock_provider, "gpt-4")

            # Call the API with fallback models and config
            fallback_config = {
                "max_fallbacks": 2,
                "retriable_errors": [RateLimitError],
                "log_fallbacks": True
            }

            ChatCompletion.create(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_models=["anthropic/claude-3", "google/gemini-pro"],
                fallback_config=fallback_config
            )

            # Verify fallback_config was passed
            args, kwargs = mock_get_provider.call_args
            assert kwargs["fallback_config"] is not None

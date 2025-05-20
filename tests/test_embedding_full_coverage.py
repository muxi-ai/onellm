#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full coverage tests for onellm/embedding.py.

This file contains tests specifically designed to achieve 100% coverage
for the embedding.py module, targeting all uncovered lines and edge cases.
"""

import pytest
from unittest import mock

from onellm import Embedding
from onellm.models import EmbeddingResponse, EmbeddingData, UsageInfo
from onellm.errors import InvalidRequestError
from onellm.utils.fallback import FallbackConfig


# Helper to create async return values for mocks
async def async_return(value):
    """Helper to create a coroutine that returns a value."""
    return value


class TestEmbeddingFullCoverage:
    """Tests to achieve full coverage for embedding.py."""

    def setup_method(self):
        """Set up test fixtures."""
        # Standard mock response
        self.mock_response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                    index=0,
                    object="embedding"
                )
            ],
            model="text-embedding-ada-002",
            usage=UsageInfo(prompt_tokens=10, total_tokens=10)
        )

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @pytest.mark.asyncio
    async def test_acreate_method(self, mock_get_provider):
        """Test the asynchronous acreate method (line 84-121)."""
        # Set up the mock provider
        mock_provider = mock.AsyncMock()
        mock_provider.create_embedding.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call acreate directly
        result = await Embedding.acreate(
            model="openai/text-embedding-ada-002",
            input="Test text"
        )

        # Verify results
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.model == "text-embedding-ada-002"

        # Fix the assertion to match the actual structure
        if isinstance(result.usage, dict):
            assert result.usage.get("prompt_tokens") == 10
        else:
            assert result.usage.prompt_tokens == 10

        # Verify provider was called correctly
        mock_get_provider.assert_called_once_with(
            primary_model="openai/text-embedding-ada-002",
            fallback_models=None,
            fallback_config=None
        )
        mock_provider.create_embedding.assert_called_once()

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @mock.patch('onellm.embedding.asyncio.run')
    def test_create_with_fallback_models(self, mock_asyncio_run, mock_get_provider):
        """Test create with fallback models (line 53-82)."""
        # Set up mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return(self.mock_response)
        mock_asyncio_run.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call with fallback models
        fallback_models = ["openai/text-embedding-3-small", "openai/text-embedding-3-large"]
        result = Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Test text",
            fallback_models=fallback_models
        )

        # Verify results
        assert result.object == "list"
        assert result.model == "text-embedding-ada-002"

        # Verify provider was called with fallback models
        mock_get_provider.assert_called_once_with(
            primary_model="openai/text-embedding-ada-002",
            fallback_models=fallback_models,
            fallback_config=None
        )

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @mock.patch('onellm.embedding.asyncio.run')
    def test_create_with_fallback_config(self, mock_asyncio_run, mock_get_provider):
        """Test create with fallback configuration (line 78-82)."""
        # Set up mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return(self.mock_response)
        mock_asyncio_run.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call with fallback config
        fallback_config = {
            "max_fallbacks": 2,
            "log_fallbacks": True
        }
        result = Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Test text",
            fallback_config=fallback_config
        )

        # Verify results
        assert result.object == "list"

        # Verify FallbackConfig was created properly
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["fallback_config"] is not None
        assert isinstance(kwargs["fallback_config"], FallbackConfig)
        assert kwargs["fallback_config"].max_fallbacks == 2
        assert kwargs["fallback_config"].log_fallbacks is True

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @pytest.mark.asyncio
    async def test_acreate_with_fallback_config(self, mock_get_provider):
        """Test acreate with fallback configuration (line 108-112)."""
        # Set up mocks
        mock_provider = mock.AsyncMock()
        mock_provider.create_embedding.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call with fallback config
        fallback_config = {
            "max_fallbacks": 3,
            "log_fallbacks": True
        }
        result = await Embedding.acreate(
            model="openai/text-embedding-ada-002",
            input="Test text",
            fallback_config=fallback_config
        )

        # Verify results
        assert result.object == "list"

        # Verify FallbackConfig was created properly
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["fallback_config"] is not None
        assert isinstance(kwargs["fallback_config"], FallbackConfig)
        assert kwargs["fallback_config"].max_fallbacks == 3
        assert kwargs["fallback_config"].log_fallbacks is True

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    def test_validate_embedding_input_with_empty_strings_in_list(self, mock_get_provider):
        """Test validation with list containing empty strings (line 43-47)."""
        # Mock not needed in this test but included to avoid actual API calls

        # Test with list of empty strings
        with pytest.raises(InvalidRequestError, match="Input cannot be empty"):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=["", "", ""]
            )

        # Test with mix of empty and non-empty strings
        # This should pass validation
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return(self.mock_response)
        mock_get_provider.return_value = (mock_provider, "mock-model")

        with mock.patch('onellm.embedding.asyncio.run', return_value=self.mock_response):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=["", "non-empty", ""]
            )

        # Verify provider was called
        mock_get_provider.assert_called_once()

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    def test_validate_embedding_input_with_none_input(self, mock_get_provider):
        """Test validation with None input (line 41-42)."""
        # Mock not needed in this test but included to avoid actual API calls

        # Test with None input
        with pytest.raises(InvalidRequestError, match="Input cannot be empty"):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=None
            )

        # Verify provider was not called
        mock_get_provider.assert_not_called()

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @mock.patch('onellm.embedding.asyncio.run')
    def test_create_with_additional_kwargs(self, mock_asyncio_run, mock_get_provider):
        """Test create with additional kwargs passed to provider (line 81-82)."""
        # Set up mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return(self.mock_response)
        mock_asyncio_run.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call with additional kwargs
        Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Test text",
            encoding_format="float",
            dimensions=1536,
            user="user-123"
        )

        # Verify additional kwargs were passed to provider
        mock_provider.create_embedding.assert_called_once()
        args, kwargs = mock_provider.create_embedding.call_args
        assert kwargs["encoding_format"] == "float"
        assert kwargs["dimensions"] == 1536
        assert kwargs["user"] == "user-123"

    @mock.patch('onellm.embedding.validate_embedding_input')
    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @mock.patch('onellm.embedding.asyncio.run')
    def test_create_validation_call(self, mock_asyncio_run, mock_get_provider, mock_validate):
        """Test that validate_embedding_input is called by create (line 62)."""
        # Set up mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return(self.mock_response)
        mock_asyncio_run.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the method
        Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Test text"
        )

        # Verify validation was called
        mock_validate.assert_called_once_with("Test text")

    @mock.patch('onellm.embedding.validate_embedding_input')
    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @pytest.mark.asyncio
    async def test_acreate_validation_call(self, mock_get_provider, mock_validate):
        """Test that validate_embedding_input is called by acreate (line 116)."""
        # Set up mocks
        mock_provider = mock.AsyncMock()
        mock_provider.create_embedding.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the method
        await Embedding.acreate(
            model="openai/text-embedding-ada-002",
            input=["Test text 1", "Test text 2"]
        )

        # Verify validation was called
        mock_validate.assert_called_once_with(["Test text 1", "Test text 2"])

    @mock.patch('onellm.embedding.get_provider_with_fallbacks')
    @pytest.mark.asyncio
    async def test_acreate_with_additional_kwargs(self, mock_get_provider):
        """Test acreate with additional kwargs passed to provider (line 121)."""
        # Set up mocks
        mock_provider = mock.AsyncMock()
        mock_provider.create_embedding.return_value = self.mock_response
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call with additional kwargs
        await Embedding.acreate(
            model="openai/text-embedding-ada-002",
            input="Test text",
            encoding_format="float",
            dimensions=1536,
            user="user-123"
        )

        # Verify additional kwargs were passed to provider
        mock_provider.create_embedding.assert_called_once()
        args, kwargs = mock_provider.create_embedding.call_args
        assert kwargs["encoding_format"] == "float"
        assert kwargs["dimensions"] == 1536
        assert kwargs["user"] == "user-123"

"""
Tests for embedding validation in muxi-llm.

These tests specifically target validation logic for embedding inputs.
"""

import pytest
from unittest import mock

from muxi.llm.errors import InvalidRequestError


class TestEmbeddingValidation:
    """Tests for input validation of the embedding functionality."""

    @mock.patch('muxi.llm.providers.base.get_provider_with_fallbacks')
    def test_empty_string_input_raises_error(self, mock_get_provider):
        """Test that empty string inputs are rejected."""
        from muxi.llm import Embedding

        # The function should raise an error before the mocks are even called
        with pytest.raises(InvalidRequestError, match="Input cannot be empty"):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=""
            )

        # Verify the provider was not called since validation should prevent it
        mock_get_provider.assert_not_called()

    @mock.patch('muxi.llm.providers.base.get_provider_with_fallbacks')
    def test_empty_list_input_raises_error(self, mock_get_provider):
        """Test that empty list inputs are rejected."""
        from muxi.llm import Embedding

        # The function should raise an error before the mocks are even called
        with pytest.raises(InvalidRequestError, match="Input cannot be empty"):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=[]
            )

        # Verify the provider was not called since validation should prevent it
        mock_get_provider.assert_not_called()

    @mock.patch('muxi.llm.providers.base.get_provider_with_fallbacks')
    def test_list_with_empty_string_raises_error(self, mock_get_provider):
        """Test that lists with only empty strings are rejected."""
        from muxi.llm import Embedding

        # The function should raise an error before the mocks are even called
        with pytest.raises(InvalidRequestError, match="Input cannot be empty"):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=[""]
            )

        # Verify the provider was not called since validation should prevent it
        mock_get_provider.assert_not_called()

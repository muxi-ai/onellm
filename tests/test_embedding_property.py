"""
Property-based tests for the embedding functionality.

These tests use Hypothesis to generate test cases that verify
the embedding API behaves correctly under various inputs.
"""

from unittest import mock

import pytest
from hypothesis import given, strategies as st

from muxi_llm import Embedding
from muxi_llm.errors import InvalidRequestError


class TestEmbeddingProperties:
    """Property-based tests for the embedding functionality."""

    @given(
        input_text=st.lists(
            st.text(min_size=1, max_size=1000),
            min_size=1,
            max_size=10
        )
    )
    @mock.patch("muxi_llm.embedding.get_provider_with_fallbacks")
    def test_batch_embedding_input_structure(self, mock_get_provider, input_text):
        """Test that batch embedding correctly processes lists of strings."""
        # Setup mock
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(input_text))],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the API
        Embedding.create(
            model="openai/text-embedding-ada-002",
            input=input_text
        )

        # Verify the provider was called with correct input format
        args, kwargs = mock_provider.create_embedding.call_args
        assert "input" in kwargs
        assert isinstance(kwargs["input"], list)
        assert all(isinstance(item, str) for item in kwargs["input"])
        assert len(kwargs["input"]) == len(input_text)

    @given(
        input_text=st.text(min_size=1, max_size=1000)
    )
    @mock.patch("muxi_llm.embedding.get_provider_with_fallbacks")
    def test_single_string_embedding(self, mock_get_provider, input_text):
        """Test that single string inputs are correctly processed."""
        # Setup mock
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the API
        Embedding.create(
            model="openai/text-embedding-ada-002",
            input=input_text
        )

        # Verify the provider was called with string converted to list
        args, kwargs = mock_provider.create_embedding.call_args
        assert "input" in kwargs
        assert isinstance(kwargs["input"], str)
        assert kwargs["input"] == input_text

    @given(
        dimension=st.integers(min_value=2, max_value=2048)
    )
    @mock.patch("muxi_llm.embedding.get_provider_with_fallbacks")
    def test_embedding_dimension_property(self, mock_get_provider, dimension):
        """Test that embeddings have the expected dimension property."""
        # Setup mock
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = {
            "object": "list",
            "data": [{"embedding": [0.1] * dimension, "index": 0}],
            "model": "custom-embedding-model",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        mock_get_provider.return_value = (mock_provider, "custom-embedding-model")

        # Call the API
        result = Embedding.create(
            model="openai/custom-embedding-model",
            input="Test text"
        )

        # Verify the result has the expected dimension
        assert len(result.data[0].embedding) == dimension

    @given(
        empty_input=st.one_of(
            st.just([]),
            st.just(""),
            st.just([""])
        )
    )
    @mock.patch("muxi_llm.embedding.get_provider_with_fallbacks")
    def test_empty_input_validation(self, mock_get_provider, empty_input):
        """Test that empty inputs are properly validated."""
        # Setup mock
        mock_provider = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the API, expecting an error for empty input
        with pytest.raises(InvalidRequestError):
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=empty_input
            )

        # Verify the provider was never called
        mock_provider.create_embedding.assert_not_called()

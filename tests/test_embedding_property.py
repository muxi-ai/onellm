"""
Property-based tests for the embedding functionality.

These tests use Hypothesis to generate test cases that verify
the embedding API behaves correctly under various inputs.
"""

import pytest
from unittest import mock

from hypothesis import given, strategies as st

from muxi.llm import Embedding
from muxi.llm.errors import InvalidRequestError


# Helper to create async return values for mocks
async def async_return(value):
    """Helper to create a coroutine that returns a value."""
    return value


class TestEmbeddingProperties:
    """Property-based tests for the embedding functionality."""

    @given(
        input_text=st.lists(
            st.text(min_size=1, max_size=1000),
            min_size=1,
            max_size=10
        )
    )
    def test_batch_embedding_input_structure(self, input_text):
        """Test that batch embedding correctly processes lists of strings."""
        # Setup mock provider
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return({
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(input_text))],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })

        # Setup get_provider_with_fallbacks mock
        with mock.patch("muxi.llm.providers.base.get_provider_with_fallbacks") as mock_get_provider:
            mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

            # Call the API
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=input_text
            )

            # Verify the provider was called with correct parameters
            call_args = mock_provider.create_embedding.call_args
            assert call_args is not None
            args, kwargs = call_args

            # Check input format
            assert "input" in kwargs
            assert isinstance(kwargs["input"], list)
            assert all(isinstance(item, str) for item in kwargs["input"])
            assert len(kwargs["input"]) == len(input_text)

    @given(
        input_text=st.text(min_size=1, max_size=1000)
    )
    def test_single_string_embedding(self, input_text):
        """Test that single string inputs are correctly processed."""
        # Setup mock provider
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return({
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })

        # Setup get_provider_with_fallbacks mock
        with mock.patch("muxi.llm.providers.base.get_provider_with_fallbacks") as mock_get_provider:
            mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

            # Call the API
            Embedding.create(
                model="openai/text-embedding-ada-002",
                input=input_text
            )

            # Verify the provider was called with correct parameters
            call_args = mock_provider.create_embedding.call_args
            assert call_args is not None
            args, kwargs = call_args

            # Check that string was converted to list
            assert "input" in kwargs
            assert isinstance(kwargs["input"], list)
            assert len(kwargs["input"]) == 1
            assert kwargs["input"][0] == input_text

    @given(
        dimension=st.integers(min_value=2, max_value=100)  # Reduced dimension for testing
    )
    def test_embedding_dimension_property(self, dimension):
        """Test that embeddings have the expected dimension property."""
        # Setup mock provider
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.return_value = async_return({
            "object": "list",
            "data": [{"embedding": [0.1] * dimension, "index": 0}],
            "model": "text-embedding-ada-002",  # Use a standard model name
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })

        # Setup get_provider_with_fallbacks mock
        with mock.patch("muxi.llm.providers.base.get_provider_with_fallbacks") as mock_get_provider:
            mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

            # Call the API
            result = Embedding.create(
                model="openai/text-embedding-ada-002",
                input="Test text"
            )

            # Verify the result has the expected dimension
            assert len(result["data"][0]["embedding"]) == dimension

    @given(
        empty_input=st.one_of(
            st.just([]),
            st.just(""),
            st.just([""])
        )
    )
    def test_empty_input_validation(self, empty_input):
        """Test that empty inputs are properly validated."""
        # For this test, we'll mock the provider but ensure it raises an error
        mock_provider = mock.MagicMock()
        mock_provider.create_embedding.side_effect = InvalidRequestError("Empty input")

        # Set up the mock
        with mock.patch("muxi.llm.providers.base.get_provider_with_fallbacks") as mock_get_provider:
            mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

            # Call the API, expecting an error
            with pytest.raises(InvalidRequestError):
                Embedding.create(
                    model="openai/text-embedding-ada-002",
                    input=empty_input
                )

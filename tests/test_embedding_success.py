"""
Tests for successful embedding creation.

These tests verify that embeddings are properly created when valid inputs are provided.
"""

from unittest import mock

from muxi_llm import Embedding
from muxi_llm.models import EmbeddingResponse, EmbeddingData, UsageInfo


# Helper to create async return values for mocks
async def async_return(value):
    """Helper to create a coroutine that returns a value."""
    return value


class TestEmbeddingSuccess:
    """Tests for successful embedding creation."""

    @mock.patch('muxi_llm.embedding.asyncio.run')
    @mock.patch('muxi_llm.embedding.get_provider_with_fallbacks')
    def test_embedding_create_with_valid_input(self, mock_get_provider, mock_asyncio_run):
        """Test embedding creation with valid input."""
        # Setup mock provider
        mock_provider = mock.MagicMock()

        # Create mock response
        mock_response = EmbeddingResponse(
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

        # Make asyncio.run return our mocked response directly
        mock_asyncio_run.return_value = mock_response

        # Set up get_provider_with_fallbacks mock
        mock_get_provider.return_value = (mock_provider, "text-embedding-ada-002")

        # Call the embedding API
        result = Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Test text"
        )

        # Verify the provider was set up correctly
        mock_get_provider.assert_called_once_with(
            primary_model="openai/text-embedding-ada-002",
            fallback_models=None,
            fallback_config=None
        )

        # Verify that asyncio.run was called
        mock_asyncio_run.assert_called_once()

        # Verify the result matches our mock
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.data[0].index == 0
        assert result.model == "text-embedding-ada-002"
        assert result.usage is not None

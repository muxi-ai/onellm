"""
Complete tests for embedding.py to reach 100% code coverage.

This file specifically targets the lines that are not being covered in other tests.
"""

import pytest
from unittest import mock

from muxi_llm import Embedding
from muxi_llm.utils.fallback import FallbackConfig


class TestEmbeddingComplete:
    """Tests to reach complete coverage of embedding.py."""

    @mock.patch('muxi_llm.providers.base.get_provider_with_fallbacks')
    @mock.patch('asyncio.run')
    def test_create_with_fallback_config(self, mock_asyncio_run, mock_get_provider):
        """Test Embedding.create with fallback_config parameter, covering line 94."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, 'model-name')
        mock_asyncio_run.return_value = mock.MagicMock()

        # Call with fallback_config to hit line 94
        Embedding.create(
            model="openai/text-embedding-ada-002",
            input="Hello world",
            fallback_config={
                "max_retries": 3,
                "backoff_factor": 0.5
            }
        )

        # Verify FallbackConfig was created
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert isinstance(kwargs['fallback_config'], FallbackConfig)
        assert kwargs['fallback_config'].max_retries == 3
        assert kwargs['fallback_config'].backoff_factor == 0.5

    @pytest.mark.asyncio
    @mock.patch('muxi_llm.providers.base.get_provider_with_fallbacks')
    async def test_acreate_with_fallback_config(self, mock_get_provider):
        """Test Embedding.acreate with fallback_config parameter, covering line 148."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_embedding_response = mock.MagicMock()
        mock_provider.create_embedding.return_value = mock_embedding_response
        mock_get_provider.return_value = (mock_provider, 'model-name')

        # Call with fallback_config to hit line 148
        result = await Embedding.acreate(
            model="openai/text-embedding-ada-002",
            input="Hello world",
            fallback_config={
                "max_retries": 3,
                "backoff_factor": 0.5
            }
        )

        # Verify FallbackConfig was created and result returned
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert isinstance(kwargs['fallback_config'], FallbackConfig)
        assert kwargs['fallback_config'].max_retries == 3
        assert kwargs['fallback_config'].backoff_factor == 0.5
        assert result == mock_embedding_response

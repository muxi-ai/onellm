"""
Advanced tests for the Completion class.

This module tests edge cases and error handling in the Completion class,
focusing on areas that were previously untested.
"""

import asyncio
import pytest
from unittest import mock
from typing import AsyncGenerator, Dict, List

from onellm import Completion
from onellm.models import CompletionResponse, CompletionChoice
from onellm.utils.fallback import FallbackConfig
from onellm.errors import APIError


class TestCompletionAdvanced:
    """Tests for advanced functionality of the Completion class."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock provider
        self.mock_provider = mock.Mock()
        self.mock_provider.create_completion = mock.AsyncMock()

        # Create a mock response
        self.mock_response = CompletionResponse(
            id="test-id",
            object="text_completion",
            created=1000000000,
            model="test-model",
            choices=[
                CompletionChoice(
                    text="Completed text",
                    index=0,
                    logprobs=None,
                    finish_reason="stop"
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        )
        self.mock_provider.create_completion.return_value = self.mock_response

        # Set up patch for get_provider_with_fallbacks
        self.provider_patcher = mock.patch("onellm.completion.get_provider_with_fallbacks")
        self.mock_get_provider = self.provider_patcher.start()
        self.mock_get_provider.return_value = (self.mock_provider, "test-model")

    def teardown_method(self):
        """Clean up after tests."""
        self.provider_patcher.stop()

    def test_create_with_invalid_input(self):
        """Test Completion.create with invalid inputs."""
        # Test with empty prompt
        with pytest.raises(ValueError):
            Completion.create(
                model="test-provider/test-model",
                prompt=""
            )

    @mock.patch("asyncio.new_event_loop")
    @mock.patch("asyncio.set_event_loop")
    def test_create_with_streaming_error(self, mock_set_loop, mock_new_loop):
        """Test error handling in streaming mode."""
        # Set up a mock loop
        mock_loop = mock.Mock()
        mock_new_loop.return_value = mock_loop

        # Set up the loop to raise an exception
        mock_loop.run_until_complete.side_effect = RuntimeError("Loop failure")

        # Call with streaming
        with pytest.raises(RuntimeError):
            Completion.create(
                model="test-provider/test-model",
                prompt="Test prompt",
                stream=True
            )

        # Verify proper cleanup
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once()

    @mock.patch("asyncio.run")
    def test_create_with_run_error(self, mock_run):
        """Test error handling in non-streaming mode."""
        # Set up asyncio.run to raise an exception
        mock_run.side_effect = RuntimeError("Run failure")

        # Call without streaming
        with pytest.raises(RuntimeError):
            Completion.create(
                model="test-provider/test-model",
                prompt="Test prompt"
            )

        # Verify asyncio.run was called
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_acreate_with_streaming_error(self):
        """Test error handling in async streaming mode."""
        # Configure provider to raise an error
        error = APIError("API failure")
        self.mock_provider.create_completion.side_effect = error

        # Call with streaming
        with pytest.raises(APIError):
            await Completion.acreate(
                model="test-provider/test-model",
                prompt="Test prompt",
                stream=True
            )

        # Verify provider was called with stream=True
        call_args = self.mock_provider.create_completion.call_args
        assert call_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_acreate_with_complex_fallback_config(self):
        """Test Completion.acreate with a complex fallback configuration."""
        # Create a complex fallback config with compatible parameters
        fallback_config = {
            "retriable_errors": [APIError, TimeoutError],
            "max_fallbacks": 3,
            "log_fallbacks": True,
            "fallback_callback": lambda **kwargs: print("Fallback occurred")
        }

        # Call acreate with the config
        result = await Completion.acreate(
            model="test-provider/test-model",
            prompt="Test prompt",
            fallback_config=fallback_config
        )

        # Check that get_provider_with_fallbacks was called with correct config
        fallback_config_obj = self.mock_get_provider.call_args.kwargs["fallback_config"]
        assert isinstance(fallback_config_obj, FallbackConfig)
        assert fallback_config_obj.max_fallbacks == 3
        assert fallback_config_obj.log_fallbacks is True
        assert len(fallback_config_obj.retriable_errors) == 2

        # Check that the result is correct
        assert result == self.mock_response

    @pytest.mark.asyncio
    async def test_acreate_with_streaming(self):
        """Test async Completion.acreate with streaming enabled."""
        # Create a mock stream generator
        async def mock_stream_generator():
            yield {"choices": [{"text": "part 1", "index": 0}]}
            yield {"choices": [{"text": "part 2", "index": 0}]}

        # Set up the provider to return our generator
        stream_gen = mock_stream_generator()
        self.mock_provider.create_completion.return_value = stream_gen

        # Call acreate with streaming
        result = await Completion.acreate(
            model="test-provider/test-model",
            prompt="Test prompt",
            stream=True
        )

        # Check that the provider was called with stream=True
        assert self.mock_provider.create_completion.call_args.kwargs["stream"] is True

        # Check that we got the stream generator back
        assert result == stream_gen

        # Check that we can iterate the generator
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["text"] == "part 1"
        assert chunks[1]["choices"][0]["text"] == "part 2"

    def test_stream_completion_without_asyncio(self):
        """Test handling streaming completions without an existing event loop."""
        # Create a mock stream generator
        async def mock_stream_generator():
            yield {"choices": [{"text": "part 1", "index": 0}]}
            yield {"choices": [{"text": "part 2", "index": 0}]}

        # Set up the provider to return our generator
        gen_instance = mock_stream_generator()
        self.mock_provider.create_completion.return_value = gen_instance

        # Mock asyncio.new_event_loop and set_event_loop
        with mock.patch("asyncio.new_event_loop") as mock_new_loop, \
             mock.patch("asyncio.set_event_loop") as mock_set_loop:

            # Set up the mock loop
            mock_loop = mock.Mock()
            mock_new_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = gen_instance

            # Call create with streaming
            result = Completion.create(
                model="test-provider/test-model",
                prompt="Test prompt",
                stream=True
            )

            # Verify that we set up the loop correctly
            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)
            mock_loop.run_until_complete.assert_called_once()

            # Check that the result is the generator instance that was returned
            # We can't compare generators directly, so let's just check it's a generator
            assert result is gen_instance

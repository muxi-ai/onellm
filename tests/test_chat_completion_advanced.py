"""
Advanced tests for the ChatCompletion class.

This module tests edge cases and error handling in the ChatCompletion class,
focusing particularly on uncovered code paths (lines 57, 69-71, 81, 126-138)
related to various edge cases and error handling.
"""

import pytest
from unittest import mock

from muxi_llm import ChatCompletion
from muxi_llm.models import ChatCompletionResponse, ChatCompletionChunk
from muxi_llm.utils.fallback import FallbackConfig


class TestChatCompletionAdvanced:
    """Tests for advanced functionality of the ChatCompletion class."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock provider
        self.mock_provider = mock.Mock()
        self.mock_provider.create_chat_completion = mock.AsyncMock()

        # Create a mock response
        self.mock_response = mock.MagicMock(spec=ChatCompletionResponse)
        self.mock_provider.create_chat_completion.return_value = self.mock_response

        # Set up patch for get_provider_with_fallbacks
        self.provider_patcher = mock.patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
        self.mock_get_provider = self.provider_patcher.start()
        self.mock_get_provider.return_value = (self.mock_provider, "test-model")

    def teardown_method(self):
        """Clean up after tests."""
        self.provider_patcher.stop()

    @mock.patch("asyncio.new_event_loop")
    @mock.patch("asyncio.set_event_loop")
    def test_create_with_streaming_loop_exception(self, mock_set_loop, mock_new_loop):
        """Test error handling in streaming mode with loop exception.

        This specifically targets the uncovered lines 69-71 in chat_completion.py
        where a new loop is created for streaming responses.
        """
        # Set up a mock loop
        mock_loop = mock.Mock()
        mock_new_loop.return_value = mock_loop

        # Set up the loop to raise an exception
        mock_loop.run_until_complete.side_effect = RuntimeError("Loop failure")

        # Call with streaming
        with pytest.raises(RuntimeError):
            ChatCompletion.create(
                model="test-provider/test-model",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

        # Verify proper loop creation and cleanup
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once()

    @mock.patch("asyncio.new_event_loop")
    @mock.patch("asyncio.set_event_loop")
    def test_create_with_streaming_loop_management(self, mock_set_loop, mock_new_loop):
        """Test the loop setup/management for streaming responses.

        This test ensures the event loop handling in lines 69-71 works properly
        when no exceptions occur.
        """
        # Create a mock stream generator
        async def mock_stream_generator():
            yield ChatCompletionChunk(id="chunk1", choices=[])
            yield ChatCompletionChunk(id="chunk2", choices=[])

        # Set up a mock loop
        mock_loop = mock.Mock()
        mock_new_loop.return_value = mock_loop

        # Set up the mock stream generator
        generator = mock_stream_generator()
        mock_loop.run_until_complete.return_value = generator

        # Configure provider to return the generator
        self.mock_provider.create_chat_completion.return_value = generator

        # Call with streaming
        result = ChatCompletion.create(
            model="test-provider/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Verify proper setup
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once()

        # Verify the generator was returned
        assert result is generator

    @mock.patch("asyncio.new_event_loop")
    @mock.patch("asyncio.set_event_loop")
    def test_create_with_streaming_empty_response(self, mock_set_loop, mock_new_loop):
        """Test the streaming path with an empty generator."""
        # Set up a mock loop
        mock_loop = mock.Mock()
        mock_new_loop.return_value = mock_loop

        # Create an empty async generator for the response
        async def empty_generator():
            # This generator doesn't yield anything
            if False:
                yield None

        # Configure loop to return the generator
        generator = empty_generator()
        mock_loop.run_until_complete.return_value = generator

        # Call with streaming
        result = ChatCompletion.create(
            model="test-provider/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Verify proper setup and the generator was returned
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        assert result is generator

    @mock.patch("asyncio.run")
    def test_create_with_runtime_error(self, mock_run):
        """Test handling of runtime errors in non-streaming mode (line 81)."""
        # Configure asyncio.run to raise a RuntimeError
        mock_run.side_effect = RuntimeError("Async runtime error")

        # Call create without streaming
        with pytest.raises(RuntimeError) as exc_info:
            ChatCompletion.create(
                model="test-provider/test-model",
                messages=[{"role": "user", "content": "Hello"}]
            )

        # Verify the error was propagated
        assert "Async runtime error" in str(exc_info.value)
        mock_run.assert_called_once()

    def test_create_with_fallback_config(self):
        """Test create with fallback configuration (line 57)."""
        # Create a fallback config
        fallback_config = {
            "max_fallbacks": 3,
            "log_fallbacks": True
        }

        # Call create with the fallback config
        ChatCompletion.create(
            model="test-provider/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            fallback_config=fallback_config
        )

        # Verify get_provider_with_fallbacks was called with a FallbackConfig object
        args, kwargs = self.mock_get_provider.call_args
        assert isinstance(kwargs["fallback_config"], FallbackConfig)
        assert kwargs["fallback_config"].max_fallbacks == 3
        assert kwargs["fallback_config"].log_fallbacks is True

    @pytest.mark.asyncio
    async def test_acreate_with_custom_parameters(self):
        """Test acreate with custom parameters (lines 126-138)."""
        # Call acreate with additional parameters
        await ChatCompletion.acreate(
            model="test-provider/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.7,
            fallback_models=["fallback-model-1", "fallback-model-2"]
        )

        # Verify provider was called with correct parameters
        self.mock_provider.create_chat_completion.assert_awaited_once()
        args, kwargs = self.mock_provider.create_chat_completion.call_args

        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["model"] == "test-model"
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["stream"] is False  # Default value

        # Verify get_provider_with_fallbacks was called with fallback_models
        args, kwargs = self.mock_get_provider.call_args
        assert kwargs["fallback_models"] == ["fallback-model-1", "fallback-model-2"]

    @pytest.mark.asyncio
    async def test_acreate_with_complex_messages(self):
        """Test acreate with complex message structures (lines 126-138)."""
        # Create complex messages with different formats
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            },
            {"role": "assistant", "content": "How can I help you?"},
            {"role": "user", "content": "Tell me a joke"}
        ]

        # Call acreate with the complex messages
        await ChatCompletion.acreate(
            model="test-provider/test-model",
            messages=messages
        )

        # Verify the complex messages were passed to the provider
        self.mock_provider.create_chat_completion.assert_awaited_once()
        args, kwargs = self.mock_provider.create_chat_completion.call_args
        assert kwargs["messages"] == messages

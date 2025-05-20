"""
Additional tests for the speech functionality.

These tests focus on the Speech class, fallback behavior,
and the synchronous wrapper function.
"""

import pytest
from unittest import mock
import asyncio

from onellm import Speech
from onellm.errors import (
    ServiceUnavailableError,
    RateLimitError
)
from onellm.utils.fallback import FallbackConfig


@pytest.mark.asyncio
async def test_fallback_models():
    """Test that fallback models are correctly passed to get_provider_with_fallbacks."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_speech = mock.AsyncMock(return_value=b"fake audio data")

    # Mock get_provider_with_fallbacks
    with mock.patch("onellm.speech.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "tts-1")

        # Call with fallback models
        result = await Speech.create(
            input="Test speech with fallbacks",
            model="openai/tts-1",
            voice="alloy",
            fallback_models=["openai/tts-1-hd", "elevenlabs/eleven-monolingual-v1"]
        )

        # Check provider setup
        mock_get_provider_with_fallbacks.assert_called_with(
            primary_model="openai/tts-1",
            fallback_models=["openai/tts-1-hd", "elevenlabs/eleven-monolingual-v1"],
            fallback_config=None
        )

        # Check provider method was called with correct params
        mock_provider.create_speech.assert_called_with(
            "Test speech with fallbacks", "tts-1", "alloy"
        )


@pytest.mark.asyncio
async def test_fallback_config():
    """Test that fallback configuration is correctly passed and used."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_speech = mock.AsyncMock(return_value=b"fake audio data")

    # Mock get_provider_with_fallbacks
    with mock.patch("onellm.speech.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "tts-1")

        # Create a fallback config
        fallback_config = {
            "retriable_errors": [RateLimitError, ServiceUnavailableError],
            "max_fallbacks": 2,
            "log_fallbacks": True
        }

        # Call with fallback config
        result = await Speech.create(
            input="Test speech with fallback config",
            model="openai/tts-1",
            voice="alloy",
            fallback_models=["openai/tts-1-hd"],
            fallback_config=fallback_config
        )

        # Check that FallbackConfig was created and passed correctly
        mock_get_provider_with_fallbacks.assert_called_once()
        fb_config = mock_get_provider_with_fallbacks.call_args[1]["fallback_config"]
        assert isinstance(fb_config, FallbackConfig)
        assert RateLimitError in fb_config.retriable_errors
        assert ServiceUnavailableError in fb_config.retriable_errors
        assert fb_config.max_fallbacks == 2
        assert fb_config.log_fallbacks is True


@pytest.mark.asyncio
async def test_output_file():
    """Test that output_file parameter correctly saves the audio data."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_audio_data = b"fake audio data"
    mock_provider.create_speech = mock.AsyncMock(return_value=mock_audio_data)

    # Mock get_provider_with_fallbacks
    with mock.patch("onellm.speech.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "tts-1")

        # Mock file open
        mock_open_file = mock.mock_open()

        # Call with output_file
        with mock.patch("builtins.open", mock_open_file):
            result = await Speech.create(
                input="Test speech with output file",
                model="openai/tts-1",
                voice="alloy",
                output_file="test_speech.mp3"
            )

        # Check that file was opened and written to
        mock_open_file.assert_called_once_with("test_speech.mp3", "wb")
        mock_open_file().write.assert_called_once_with(mock_audio_data)

        # Check that the audio data was still returned
        assert result == mock_audio_data


def test_sync_wrapper():
    """Test the synchronous wrapper function."""
    # Mock the returned audio data
    mock_audio_data = b"fake audio data"

    # Mock asyncio.run to return the fake audio data
    with mock.patch("asyncio.run") as mock_asyncio_run:
        mock_asyncio_run.return_value = mock_audio_data

        # Call the sync method
        result = Speech.create_sync(
            input="Test synchronous speech",
            model="openai/tts-1",
            voice="alloy",
            response_format="mp3"
        )

        # Check that asyncio.run was called (with any argument since the coroutine object changes)
        assert mock_asyncio_run.called

        # Check that the result was returned correctly
        assert result == mock_audio_data


@pytest.mark.asyncio
async def test_additional_params():
    """Test that additional parameters are correctly passed through."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_speech = mock.AsyncMock(return_value=b"fake audio data")

    # Mock get_provider_with_fallbacks
    with mock.patch("onellm.speech.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "tts-1")

        # Call with additional parameters
        await Speech.create(
            input="Test speech with additional params",
            model="openai/tts-1",
            voice="alloy",
            response_format="mp3",
            speed=1.5,
            custom_param="test_value"
        )

        # Check that all parameters were passed correctly
        mock_provider.create_speech.assert_called_once()
        args, kwargs = mock_provider.create_speech.call_args
        assert kwargs["response_format"] == "mp3"
        assert kwargs["speed"] == 1.5
        assert kwargs["custom_param"] == "test_value"

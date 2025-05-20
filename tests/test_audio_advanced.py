"""
Advanced tests for the audio capabilities module.

These tests focus on the audio transcription and translation functionality,
particularly the fallback mechanisms, synchronous wrappers, and error handling.
"""

import pytest
from unittest import mock
from io import BytesIO

from onellm import AudioTranscription, AudioTranslation
from onellm.utils.fallback import FallbackConfig
from onellm.errors import ServiceUnavailableError, RateLimitError


@pytest.mark.asyncio
async def test_audio_transcription_fallback_models():
    """Test AudioTranscription with fallback models."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_transcription = mock.AsyncMock(return_value={"text": "Transcribed text"})

    # Mock get_provider_with_fallbacks
    with mock.patch(
            "onellm.audio.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "whisper-1")

        # Test with fallback models
        result = await AudioTranscription.create(
            file=b"fake audio data",
            model="openai/whisper-1",
            fallback_models=["openai/whisper-1-hd", "anthropic/audio-model"],
            language="en"
        )

        # Check provider setup with fallbacks
        mock_get_provider_with_fallbacks.assert_called_with(
            primary_model="openai/whisper-1",
            fallback_models=["openai/whisper-1-hd", "anthropic/audio-model"],
            fallback_config=None
        )

        # Check provider method call
        mock_provider.create_transcription.assert_called_with(
            b"fake audio data", "whisper-1", language="en"
        )
        assert result == {"text": "Transcribed text"}


@pytest.mark.asyncio
async def test_audio_transcription_custom_fallback_config():
    """Test AudioTranscription with custom fallback configuration."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_transcription = mock.AsyncMock(return_value={"text": "Transcribed text"})

    # Mock get_provider_with_fallbacks
    with mock.patch(
            "onellm.audio.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "whisper-1")

        # Create a fallback config
        fallback_config = {
            "retriable_errors": [RateLimitError, ServiceUnavailableError],
            "max_fallbacks": 2,
            "log_fallbacks": True
        }

        # Call with fallback config
        await AudioTranscription.create(
            file=b"fake audio data",
            model="openai/whisper-1",
            fallback_models=["openai/whisper-1-hd"],
            fallback_config=fallback_config
        )

        # Verify fallback config was created and passed correctly
        mock_get_provider_with_fallbacks.assert_called_once()
        fb_config = mock_get_provider_with_fallbacks.call_args[1]["fallback_config"]
        assert isinstance(fb_config, FallbackConfig)
        assert RateLimitError in fb_config.retriable_errors
        assert ServiceUnavailableError in fb_config.retriable_errors
        assert fb_config.max_fallbacks == 2
        assert fb_config.log_fallbacks is True


def test_audio_transcription_sync_wrapper():
    """Test the synchronous wrapper for audio transcription."""
    # Mock the response data
    mock_result = {"text": "Synchronous transcription test"}

    # Mock the async create method to check parameters
    with mock.patch.object(
        AudioTranscription, "create", return_value=mock_result
    ) as mock_create, mock.patch("asyncio.run", side_effect=lambda x: x) as mock_run:

        # Call the sync method
        AudioTranscription.create_sync(
            file="test.mp3",
            model="openai/whisper-1",
            language="en",
            temperature=0.7
        )

        # Check that AudioTranscription.create was called with correct parameters,
        # including default parameters that are automatically added
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Check the explicitly passed parameters
        assert call_args.kwargs["file"] == "test.mp3"
        assert call_args.kwargs["model"] == "openai/whisper-1"
        assert call_args.kwargs["language"] == "en"
        assert call_args.kwargs["temperature"] == 0.7

        # Default parameters should also be present
        assert "fallback_models" in call_args.kwargs
        assert "fallback_config" in call_args.kwargs

        # Check that asyncio.run was called
        assert mock_run.called


@pytest.mark.asyncio
async def test_audio_translation_with_file_object():
    """Test AudioTranslation with a file-like object."""
    # Create a file-like object
    file_obj = BytesIO(b"fake audio data")
    file_obj.name = "test.mp3"  # Add a name attribute to simulate a real file

    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_translation = mock.AsyncMock(return_value={"text": "Translated text"})

    # Mock get_provider_with_fallbacks
    with mock.patch(
            "onellm.audio.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "whisper-1")

        # Call the method with a file object
        result = await AudioTranslation.create(
            file=file_obj,
            model="openai/whisper-1"
        )

        # Check that the file object was passed correctly
        mock_provider.create_translation.assert_called_with(file_obj, "whisper-1")
        assert result == {"text": "Translated text"}


def test_audio_translation_sync_wrapper():
    """Test the synchronous wrapper for audio translation."""
    # Mock the response data
    mock_result = {"text": "Synchronous translation test"}

    # Mock the async create method to check parameters
    with mock.patch.object(
        AudioTranslation, "create", return_value=mock_result
    ) as mock_create, mock.patch("asyncio.run", side_effect=lambda x: x) as mock_run:

        # Call the sync method with various optional parameters
        AudioTranslation.create_sync(
            file="foreign_speech.mp3",
            model="openai/whisper-1",
            response_format="srt",
            temperature=0.3,
            prompt="This is a recording about science"
        )

        # Check that AudioTranslation.create was called with correct parameters,
        # including default parameters that are automatically added
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Check the explicitly passed parameters
        assert call_args.kwargs["file"] == "foreign_speech.mp3"
        assert call_args.kwargs["model"] == "openai/whisper-1"
        assert call_args.kwargs["response_format"] == "srt"
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["prompt"] == "This is a recording about science"

        # Default parameters should also be present
        assert "fallback_models" in call_args.kwargs
        assert "fallback_config" in call_args.kwargs

        # Check that asyncio.run was called
        assert mock_run.called


@pytest.mark.asyncio
async def test_audio_translation_all_parameters():
    """Test AudioTranslation with all possible parameters."""
    # Mock provider and method
    mock_provider = mock.Mock()
    mock_provider.create_translation = mock.AsyncMock(
        return_value={"text": "Translated with all params"}
    )

    # Mock get_provider_with_fallbacks
    with mock.patch(
            "onellm.audio.get_provider_with_fallbacks") as mock_get_provider_with_fallbacks:
        mock_get_provider_with_fallbacks.return_value = (mock_provider, "whisper-1")

        # Call with all possible parameters
        result = await AudioTranslation.create(
            file=b"fake audio data",
            model="openai/whisper-1",
            fallback_models=["openai/whisper-1-hd"],
            fallback_config={"max_fallbacks": 3},
            prompt="Translation context",
            response_format="verbose_json",
            temperature=0.5,
            custom_param="test_value"
        )

        # Check that all parameters were passed through to the provider
        mock_provider.create_translation.assert_called_once()
        args, kwargs = mock_provider.create_translation.call_args
        assert kwargs["prompt"] == "Translation context"
        assert kwargs["response_format"] == "verbose_json"
        assert kwargs["temperature"] == 0.5
        assert kwargs["custom_param"] == "test_value"

        # Check result
        assert result == {"text": "Translated with all params"}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Additional test coverage for onellm/providers/openai.py to improve
overall test coverage with a focus on audio processing, MIME type detection,
speech generation, and error handling in raw requests.
"""

import io
import pytest
from unittest.mock import AsyncMock, patch, mock_open

from onellm.providers.openai import OpenAIProvider
from onellm.errors import APIError, InvalidRequestError


class TestOpenAIAudioProcessing:
    """Tests for audio file processing methods in OpenAI provider."""

    def setup_method(self):
        self.provider = OpenAIProvider(api_key="sk-test-key")

    def test_process_audio_file_with_filepath(self):
        """Test processing an audio file using a file path."""
        test_filepath = "/path/to/audio.mp3"
        test_content = b"test audio content"

        with patch("builtins.open", mock_open(read_data=test_content)) as mock_file:
            file_data, filename = self.provider._process_audio_file(test_filepath)

            mock_file.assert_called_once_with(test_filepath, "rb")
            assert file_data == test_content
            assert filename == "audio.mp3"  # Should use the filename from the path

    def test_process_audio_file_with_bytes(self):
        """Test processing an audio file using bytes data."""
        test_content = b"test audio content"

        file_data, filename = self.provider._process_audio_file(test_content)

        assert file_data == test_content
        assert filename == "audio.mp3"  # Should use default filename

    def test_process_audio_file_with_bytes_custom_filename(self):
        """Test processing an audio file using bytes data with custom filename."""
        test_content = b"test audio content"
        custom_filename = "custom.wav"

        file_data, filename = self.provider._process_audio_file(
            test_content, filename=custom_filename
        )

        assert file_data == test_content
        assert filename == custom_filename

    def test_process_audio_file_with_file_object(self):
        """Test processing an audio file using a file-like object."""
        test_content = b"test audio content"

        # Create a file-like object
        file_obj = io.BytesIO(test_content)
        file_obj.name = "file_obj.mp3"

        file_data, filename = self.provider._process_audio_file(file_obj)

        assert file_data == test_content
        assert filename == "file_obj.mp3"

    def test_process_audio_file_with_file_object_no_name(self):
        """Test processing a file-like object without a name attribute."""
        test_content = b"test audio content"

        # Create a file-like object without a name
        file_obj = io.BytesIO(test_content)

        file_data, filename = self.provider._process_audio_file(file_obj)

        assert file_data == test_content
        assert filename == "audio.mp3"  # Should use default filename

    def test_process_audio_file_invalid_type(self):
        """Test processing an audio file with an invalid type."""
        with pytest.raises(InvalidRequestError, match="Invalid file type"):
            self.provider._process_audio_file(123)  # An integer is not a valid file type

    def test_guess_audio_content_type(self):
        """Test guessing the content type based on file extension."""
        # Test common audio formats
        assert self.provider._guess_audio_content_type("audio.mp3") == "audio/mpeg"
        assert self.provider._guess_audio_content_type("audio.mp4") == "audio/mp4"
        assert self.provider._guess_audio_content_type("audio.wav") == "audio/wav"
        assert self.provider._guess_audio_content_type("audio.webm") == "audio/webm"
        assert self.provider._guess_audio_content_type("audio.m4a") == "audio/mp4"
        assert self.provider._guess_audio_content_type("audio.mpeg") == "audio/mpeg"
        assert self.provider._guess_audio_content_type("audio.mpga") == "audio/mpeg"

        # Test with uppercase extension
        assert self.provider._guess_audio_content_type("audio.MP3") == "audio/mpeg"

        # Test with unknown extension - should return default
        assert self.provider._guess_audio_content_type("audio.xyz") == "audio/mpeg"

        # Test with no extension
        assert self.provider._guess_audio_content_type("audionoext") == "audio/mpeg"


class TestOpenAISpeechGeneration:
    """Tests for speech generation validation in OpenAI provider."""

    def setup_method(self):
        self.provider = OpenAIProvider(api_key="sk-test-key")
        # Patch the _make_request_raw method to prevent actual API calls
        self.make_request_raw_patcher = patch.object(
            self.provider, "_make_request_raw", return_value=b"fake audio data"
        )
        self.mock_make_request_raw = self.make_request_raw_patcher.start()

    def teardown_method(self):
        self.make_request_raw_patcher.stop()

    @pytest.mark.asyncio
    async def test_create_speech_basic(self):
        """Test basic speech generation with default parameters."""
        result = await self.provider.create_speech("Test text")

        self.mock_make_request_raw.assert_called_once()
        # Check that the appropriate parameters were passed
        call_args = self.mock_make_request_raw.call_args[1]
        assert call_args["method"] == "POST"
        assert call_args["path"] == "/audio/speech"
        assert call_args["data"]["input"] == "Test text"
        assert call_args["data"]["model"] == "tts-1"
        assert call_args["data"]["voice"] == "alloy"

        assert result == b"fake audio data"

    @pytest.mark.asyncio
    async def test_create_speech_all_params(self):
        """Test speech generation with all parameters specified."""
        result = await self.provider.create_speech(
            "Test text",
            model="tts-1-hd",
            voice="nova",
            response_format="opus",
            speed=1.5
        )

        self.mock_make_request_raw.assert_called_once()
        call_args = self.mock_make_request_raw.call_args[1]
        assert call_args["data"]["input"] == "Test text"
        assert call_args["data"]["model"] == "tts-1-hd"
        assert call_args["data"]["voice"] == "nova"
        assert call_args["data"]["response_format"] == "opus"
        assert call_args["data"]["speed"] == 1.5

        assert result == b"fake audio data"

    @pytest.mark.asyncio
    async def test_create_speech_empty_input(self):
        """Test speech generation with empty input."""
        with pytest.raises(InvalidRequestError, match="Input text is required"):
            await self.provider.create_speech("")

    @pytest.mark.asyncio
    async def test_create_speech_invalid_input_type(self):
        """Test speech generation with invalid input type."""
        with pytest.raises(InvalidRequestError, match="Input text is required"):
            await self.provider.create_speech(123)  # Not a string

    @pytest.mark.asyncio
    async def test_create_speech_invalid_model(self):
        """Test speech generation with invalid model."""
        with pytest.raises(InvalidRequestError, match="not a supported TTS model"):
            await self.provider.create_speech("Test text", model="gpt-4")

    @pytest.mark.asyncio
    async def test_create_speech_invalid_voice(self):
        """Test speech generation with invalid voice."""
        with pytest.raises(InvalidRequestError, match="Voice 'invalid' is not supported"):
            await self.provider.create_speech("Test text", voice="invalid")

    @pytest.mark.asyncio
    async def test_create_speech_invalid_response_format(self):
        """Test speech generation with invalid response format."""
        with pytest.raises(InvalidRequestError,
                           match="Response format 'invalid' is not supported"):
            await self.provider.create_speech("Test text", response_format="invalid")

    @pytest.mark.asyncio
    async def test_create_speech_invalid_speed_too_low(self):
        """Test speech generation with speed too low."""
        with pytest.raises(InvalidRequestError,
                           match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech("Test text", speed=0.1)

    @pytest.mark.asyncio
    async def test_create_speech_invalid_speed_too_high(self):
        """Test speech generation with speed too high."""
        with pytest.raises(InvalidRequestError,
                           match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech("Test text", speed=5.0)

    @pytest.mark.asyncio
    async def test_create_speech_invalid_speed_type(self):
        """Test speech generation with invalid speed type."""
        with pytest.raises(InvalidRequestError,
                           match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech("Test text", speed="fast")  # Not a number


class TestOpenAIRawRequestHandling:
    """Tests for raw request handling in OpenAI provider."""

    def setup_method(self):
        self.provider = OpenAIProvider(api_key="sk-test-key")
        # Patch the retry_async function to simplify testing
        self.retry_patcher = patch("onellm.providers.openai.retry_async")
        self.mock_retry = self.retry_patcher.start()

    def teardown_method(self):
        self.retry_patcher.stop()

    @pytest.mark.asyncio
    async def test_make_request_raw_success(self):
        """Test successful raw request."""
        # Mock response for success case
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"test binary data")

        # Set up our async side effect to directly return the response data
        async def mock_retry_side_effect(func, config):
            return await mock_response.read()

        self.mock_retry.side_effect = mock_retry_side_effect

        # Execute the test
        result = await self.provider._make_request_raw(
            method="GET", path="/test/endpoint"
        )

        assert result == b"test binary data"
        self.mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_raw_error_json(self):
        """Test raw request with JSON error response."""
        # Set up the API error to be raised
        test_error = APIError("Test API error")

        # Set up our async side effect to raise the API error
        async def mock_retry_side_effect(func, config):
            raise test_error

        self.mock_retry.side_effect = mock_retry_side_effect

        # Execute the test
        with pytest.raises(APIError, match="Test API error"):
            await self.provider._make_request_raw(
                method="GET", path="/test/endpoint"
            )

        self.mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_raw_error_non_json(self):
        """Test raw request with non-JSON error response."""
        # Set up the specific API error with HTML content
        test_error = APIError(
            "OpenAI API error: <html>Error page</html> (status code: 500)",
            provider="openai",
            status_code=500
        )

        # Set up our async side effect to raise the API error
        async def mock_retry_side_effect(func, config):
            raise test_error

        self.mock_retry.side_effect = mock_retry_side_effect

        # Execute the test
        with pytest.raises(APIError) as excinfo:
            await self.provider._make_request_raw(
                method="GET", path="/test/endpoint"
            )

        # Verify the error message and status code
        assert "status code: 500" in str(excinfo.value)
        assert "<html>Error page</html>" in str(excinfo.value)
        self.mock_retry.assert_called_once()

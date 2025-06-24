import pytest
"""
Tests for the audio functionality in the OpenAI provider.

These tests verify that the OpenAI provider can correctly handle
audio transcription and translation requests.
"""

import unittest
from unittest import mock
import io

from onellm.providers.openai import OpenAIProvider
from onellm import AudioTranscription, AudioTranslation
from onellm.errors import InvalidRequestError


class TestAudioCapabilities(unittest.TestCase):
    """Tests for audio capabilities in the OpenAI provider."""

    def setUp(self):
        """Set up test environment."""
        # Mock API key for testing
        self.api_key = "test-api-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    def test_process_audio_file_path(self):
        """Test processing an audio file from a file path."""
        # Mock open to avoid actual file system access
        mock_open = mock.mock_open(read_data=b"fake audio data")
        with mock.patch("builtins.open", mock_open):
            file_data, filename = self.provider._process_audio_file("test.mp3")

            # Check that open was called with the correct args
            mock_open.assert_called_once_with("test.mp3", "rb")

            # Check the returned data
            self.assertEqual(file_data, b"fake audio data")
            self.assertEqual(filename, "test.mp3")

    def test_process_audio_file_bytes(self):
        """Test processing an audio file from bytes."""
        audio_bytes = b"fake audio data"
        file_data, filename = self.provider._process_audio_file(audio_bytes)

        self.assertEqual(file_data, audio_bytes)
        self.assertEqual(filename, "audio.mp3")  # Default filename

    def test_process_audio_file_file_object(self):
        """Test processing an audio file from a file-like object."""
        # Create a file-like object
        file_object = io.BytesIO(b"fake audio data")
        file_object.name = "test.mp3"

        file_data, filename = self.provider._process_audio_file(file_object)

        self.assertEqual(file_data, b"fake audio data")
        self.assertEqual(filename, "test.mp3")

    def test_process_audio_file_invalid(self):
        """Test that an invalid file type raises an error."""
        with self.assertRaises(InvalidRequestError):
            self.provider._process_audio_file(123)  # Invalid type

    def test_guess_audio_content_type(self):
        """Test guessing the content type from file extensions."""
        test_cases = [
            ("audio.mp3", "audio/mpeg"),
            ("audio.wav", "audio/wav"),
            ("audio.m4a", "audio/mp4"),
            ("audio.webm", "audio/webm"),
            ("audio.unknown", "audio/mpeg"),  # Default for unknown
            ("audio", "audio/mpeg")  # No extension
        ]

        for filename, expected_type in test_cases:
            content_type = self.provider._guess_audio_content_type(filename)
            self.assertEqual(content_type, expected_type)

    @pytest.mark.asyncio
    @mock.patch("onellm.providers.openai.OpenAIProvider._make_request")
    async def test_create_transcription(self, mock_make_request):
        """Test creating a transcription."""
        # Mock response from the API
        mock_response = {
            "text": "This is a test transcription.",
            "language": "en",
            "duration": 10.5
        }
        mock_make_request.return_value = mock_response

        # Create a test file
        test_file = io.BytesIO(b"fake audio data")
        test_file.name = "test.mp3"

        # Call the method
        result = await self.provider.create_transcription(
            file=test_file,
            model="whisper-1",
            language="en"
        )

        # Check the result
        self.assertEqual(result["text"], "This is a test transcription.")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["duration"], 10.5)

        # Check API call
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args

        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["path"], "/audio/transcriptions")
        self.assertEqual(kwargs["data"]["model"], "whisper-1")
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertEqual(kwargs["files"]["file"]["filename"], "test.mp3")
        self.assertEqual(kwargs["files"]["file"]["content_type"], "audio/mpeg")

    @pytest.mark.asyncio
    @mock.patch("onellm.providers.openai.OpenAIProvider._make_request")
    async def test_create_translation(self, mock_make_request):
        """Test creating a translation."""
        # Mock response from the API
        mock_response = {
            "text": "This is a test translation.",
            "duration": 10.5
        }
        mock_make_request.return_value = mock_response

        # Create a test file
        test_file = io.BytesIO(b"fake audio data")
        test_file.name = "test.mp3"

        # Call the method
        result = await self.provider.create_translation(
            file=test_file,
            model="whisper-1",
            prompt="Technical discussion"
        )

        # Check the result
        self.assertEqual(result["text"], "This is a test translation.")
        self.assertEqual(result["language"], "en")  # Always English for translations
        self.assertEqual(result["task"], "translation")
        self.assertEqual(result["duration"], 10.5)

        # Check API call
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args

        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["path"], "/audio/translations")
        self.assertEqual(kwargs["data"]["model"], "whisper-1")
        self.assertEqual(kwargs["data"]["prompt"], "Technical discussion")
        self.assertEqual(kwargs["files"]["file"]["filename"], "test.mp3")
        self.assertEqual(kwargs["files"]["file"]["content_type"], "audio/mpeg")

    @mock.patch("onellm.providers.get_provider")
    @pytest.mark.asyncio
    @mock.patch("onellm.audio.parse_model_name")
    async def test_audio_transcription_class(self, mock_parse_model_name, mock_get_provider):
        """Test the AudioTranscription class."""
        # Mock the provider and method
        mock_provider = mock.Mock()
        mock_result = {"text": "This is a test transcription."}
        mock_provider.create_transcription.return_value = mock_result

        # Set up the mocks
        mock_parse_model_name.return_value = ("openai", "whisper-1")
        mock_get_provider.return_value = mock_provider

        # Call the method
        result = await AudioTranscription.create(
            file="test.mp3",
            model="openai/whisper-1",
            language="en"
        )

        # Check the result
        self.assertEqual(result, mock_result)

        # Check that the correct provider method was called
        mock_parse_model_name.assert_called_with("openai/whisper-1")
        mock_get_provider.assert_called_with("openai")
        mock_provider.create_transcription.assert_called_with(
            "test.mp3", "whisper-1", language="en"
        )

    @mock.patch("onellm.providers.get_provider")
    @pytest.mark.asyncio
    @mock.patch("onellm.audio.parse_model_name")
    async def test_audio_translation_class(self, mock_parse_model_name, mock_get_provider):
        """Test the AudioTranslation class."""
        # Mock the provider and method
        mock_provider = mock.Mock()
        mock_result = {"text": "This is a test translation."}
        mock_provider.create_translation.return_value = mock_result

        # Set up the mocks
        mock_parse_model_name.return_value = ("openai", "whisper-1")
        mock_get_provider.return_value = mock_provider

        # Call the method
        result = await AudioTranslation.create(
            file="test.mp3",
            model="openai/whisper-1",
            prompt="Technical discussion"
        )

        # Check the result
        self.assertEqual(result, mock_result)

        # Check that the correct provider method was called
        mock_parse_model_name.assert_called_with("openai/whisper-1")
        mock_get_provider.assert_called_with("openai")
        mock_provider.create_translation.assert_called_with(
            "test.mp3", "whisper-1", prompt="Technical discussion"
        )


if __name__ == "__main__":
    unittest.main()

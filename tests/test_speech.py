"""
Tests for the text-to-speech functionality in the OpenAI provider.

These tests verify that the OpenAI provider can correctly handle
text-to-speech requests.
"""

import unittest
from unittest import mock
import asyncio

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm import Speech
from muxi_llm.errors import InvalidRequestError
from tests.test_utils import run_async


class TestSpeechCapabilities(unittest.TestCase):
    """Tests for text-to-speech capabilities in the OpenAI provider."""

    def setUp(self):
        """Set up test environment."""
        # Mock API key for testing
        self.api_key = "test-api-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    @run_async
    async def test_create_speech(self):
        """Test creating speech from text."""
        # Mock response from the API
        mock_patch = "muxi_llm.providers.openai.OpenAIProvider._make_request_raw"
        with mock.patch(mock_patch) as mock_make_request_raw:
            mock_audio_data = b"fake audio data"
            mock_make_request_raw.return_value = mock_audio_data

            # Call the method
            result = await self.provider.create_speech(
                input="Hello, this is a test",
                model="tts-1",
                voice="alloy",
                response_format="mp3",
                speed=1.0
            )

            # Check the result
            self.assertEqual(result, mock_audio_data)

            # Check API call
            mock_make_request_raw.assert_called_once()
            args, kwargs = mock_make_request_raw.call_args

            self.assertEqual(kwargs["method"], "POST")
            self.assertEqual(kwargs["path"], "/audio/speech")
            self.assertEqual(kwargs["data"]["input"], "Hello, this is a test")
            self.assertEqual(kwargs["data"]["model"], "tts-1")
            self.assertEqual(kwargs["data"]["voice"], "alloy")
            self.assertEqual(kwargs["data"]["response_format"], "mp3")
            self.assertEqual(kwargs["data"]["speed"], 1.0)

    def test_create_speech_invalid_input(self):
        """Test that invalid input raises an error."""
        # Test with empty input
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(input="", model="tts-1", voice="alloy"))

        # Test with non-string input
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(input=123, model="tts-1", voice="alloy"))

    def test_create_speech_invalid_model(self):
        """Test that invalid model raises an error."""
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(
                input="Test",
                model="invalid-model",
                voice="alloy"
            ))

    def test_create_speech_invalid_voice(self):
        """Test that invalid voice raises an error."""
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(
                input="Test",
                model="tts-1",
                voice="invalid-voice"
            ))

    def test_create_speech_invalid_format(self):
        """Test that invalid response format raises an error."""
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(
                input="Test",
                model="tts-1",
                voice="alloy",
                response_format="invalid-format"
            ))

    def test_create_speech_invalid_speed(self):
        """Test that invalid speed raises an error."""
        with self.assertRaises(InvalidRequestError):
            asyncio_run(self.provider.create_speech(
                input="Test",
                model="tts-1",
                voice="alloy",
                speed=5.0  # Maximum is 4.0
            ))

    def test_speech_class(self):
        """Test the Speech class synchronously."""
        # Data for testing
        test_input = "Hello, this is a test"
        test_model = "openai/tts-1"
        test_voice = "alloy"
        test_format = "mp3"
        test_speed = 1.0
        mock_audio_data = b"fake audio data"

        # Mock Speech.create_sync to avoid actual API calls
        with mock.patch.object(Speech, 'create_sync', return_value=mock_audio_data) as mock_create:
            # Call the method synchronously
            result = Speech.create_sync(
                input=test_input,
                model=test_model,
                voice=test_voice,
                response_format=test_format,
                speed=test_speed
            )

            # Check the result
            self.assertEqual(result, mock_audio_data)

            # Verify the correct parameters were passed
            mock_create.assert_called_once_with(
                input=test_input,
                model=test_model,
                voice=test_voice,
                response_format=test_format,
                speed=test_speed
            )

    def test_speech_class_with_output_file(self):
        """Test the Speech class with output file synchronously."""
        # Test data
        test_input = "Hello, this is a test"
        test_model = "openai/tts-1"
        test_voice = "alloy"
        test_output_file = "test_output.mp3"
        mock_audio_data = b"fake audio data"

        # Mock both Speech.create_sync and the open function
        create_patch = mock.patch.object(Speech, 'create_sync', return_value=mock_audio_data)
        with create_patch as mock_create:

            # Call the method
            result = Speech.create_sync(
                input=test_input,
                model=test_model,
                voice=test_voice,
                output_file=test_output_file
            )

            # Check the result
            self.assertEqual(result, mock_audio_data)

            # Verify the correct parameters were passed
            mock_create.assert_called_once_with(
                input=test_input,
                model=test_model,
                voice=test_voice,
                output_file=test_output_file
            )


def asyncio_run(coro):
    """Helper function to run coroutines in tests."""
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()

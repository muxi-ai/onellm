"""
Tests for the text-to-speech functionality in the OpenAI provider.

These tests verify that the OpenAI provider can correctly handle
text-to-speech requests.
"""

import unittest
from unittest import mock

from muxi.llm.providers.openai import OpenAIProvider
from muxi.llm import Speech
from muxi.llm.errors import InvalidRequestError


class TestSpeechCapabilities(unittest.TestCase):
    """Tests for text-to-speech capabilities in the OpenAI provider."""

    def setUp(self):
        """Set up test environment."""
        # Mock API key for testing
        self.api_key = "test-api-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    @mock.patch("muxi.llm.providers.openai.OpenAIProvider._make_request_raw")
    async def test_create_speech(self, mock_make_request_raw):
        """Test creating speech from text."""
        # Mock response from the API
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

    @mock.patch("muxi.llm.providers.get_provider")
    @mock.patch("muxi.llm.speech.parse_model_name")
    async def test_speech_class(self, mock_parse_model_name, mock_get_provider):
        """Test the Speech class."""
        # Mock the provider and method
        mock_provider = mock.Mock()
        mock_audio_data = b"fake audio data"
        mock_provider.create_speech.return_value = mock_audio_data

        # Set up the mocks
        mock_parse_model_name.return_value = ("openai", "tts-1")
        mock_get_provider.return_value = mock_provider

        # Call the method
        result = await Speech.create(
            input="Hello, this is a test",
            model="openai/tts-1",
            voice="alloy",
            response_format="mp3",
            speed=1.0
        )

        # Check the result
        self.assertEqual(result, mock_audio_data)

        # Check that the correct provider method was called
        mock_parse_model_name.assert_called_with("openai/tts-1")
        mock_get_provider.assert_called_with("openai")
        mock_provider.create_speech.assert_called_with(
            "Hello, this is a test",
            "tts-1",
            "alloy",
            response_format="mp3",
            speed=1.0
        )

    @mock.patch("muxi.llm.providers.get_provider")
    @mock.patch("muxi.llm.speech.parse_model_name")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    async def test_speech_class_with_output_file(
        self, mock_open, mock_parse_model_name, mock_get_provider
    ):
        """Test the Speech class with output file."""
        # Mock the provider and method
        mock_provider = mock.Mock()
        mock_audio_data = b"fake audio data"
        mock_provider.create_speech.return_value = mock_audio_data

        # Set up the mocks
        mock_parse_model_name.return_value = ("openai", "tts-1")
        mock_get_provider.return_value = mock_provider

        # Call the method with output_file
        result = await Speech.create(
            input="Hello, this is a test",
            model="openai/tts-1",
            voice="alloy",
            output_file="test_output.mp3"
        )

        # Check the result
        self.assertEqual(result, mock_audio_data)

        # Check that the file was opened and written to
        mock_open.assert_called_with("test_output.mp3", "wb")
        mock_open().write.assert_called_with(mock_audio_data)


def asyncio_run(coro):
    """Helper function to run coroutines in tests."""
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()

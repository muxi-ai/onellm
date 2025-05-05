"""
Advanced tests for the OpenAI provider implementation.

This module tests the OpenAI provider's API interactions, response handling,
and error conditions that were previously uncovered.
"""

import pytest
from unittest import mock
from unittest.mock import patch
from io import BytesIO
import json

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.errors import AuthenticationError
from muxi_llm.types.common import ImageGenerationResult, TranscriptionResult


class TestOpenAIProviderAdvanced:
    """Advanced tests for OpenAI provider functionality."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        # Set up provider with test API key
        self.provider = OpenAIProvider(api_key="sk-test-key")

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_create_image(self, mock_make_request):
        """Test image generation with DALL-E models."""
        # Mock response from the API
        mock_response = {
            "created": 1677858242,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A test image"
                }
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.create_image(
            prompt="A test image",
            model="dall-e-3",
            size="1024x1024",
            quality="standard",
            style="vivid"
        )

        # Check the result - should be properly converted to ImageGenerationResult
        assert isinstance(result, ImageGenerationResult)
        assert result.created == 1677858242
        assert len(result.data) == 1
        assert result.data[0]["url"] == "https://example.com/image.png"

        # Verify that the API was called correctly
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        request_data = call_args.kwargs["data"]
        assert request_data["prompt"] == "A test image"
        assert request_data["model"] == "dall-e-3"
        assert request_data["size"] == "1024x1024"
        assert request_data["quality"] == "standard"
        assert request_data["style"] == "vivid"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_create_image_with_options(self, mock_make_request):
        """Test image generation with different options."""
        # Mock response from the API
        mock_response = {
            "created": 1677858242,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"}
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method with multiple images for DALL-E 2
        result = await self.provider.create_image(
            prompt="A test image",
            model="dall-e-2",
            n=2,
            size="512x512",
            response_format="url"
        )

        # Check the result - should be converted to ImageGenerationResult
        assert isinstance(result, ImageGenerationResult)
        assert len(result.data) == 2
        assert result.data[0]["url"] == "https://example.com/image1.png"
        assert result.data[1]["url"] == "https://example.com/image2.png"

        # Verify API call parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        request_data = call_args.kwargs["data"]
        assert request_data["prompt"] == "A test image"
        assert request_data["model"] == "dall-e-2"
        assert request_data["n"] == 2
        assert request_data["size"] == "512x512"
        assert request_data["response_format"] == "url"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_transcription(self, mock_make_request):
        """Test audio transcription functionality."""
        # Mock response
        mock_response = {
            "text": "This is a transcription test."
        }
        mock_make_request.return_value = mock_response

        # Mock file handling
        test_audio = b"fake audio data"

        # Patch open to return our test audio
        with patch("builtins.open", mock.mock_open(read_data=test_audio)):
            # Test with file path
            result = await self.provider.create_transcription(
                file="test.mp3",
                model="whisper-1",
                language="en"
            )

            # Check result - should be converted to TranscriptionResult
            assert isinstance(result, TranscriptionResult)
            assert result.text == "This is a transcription test."

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.kwargs["files"] is not None
        assert call_args.kwargs["data"] is not None
        assert call_args.kwargs["data"]["model"] == "whisper-1"
        assert call_args.kwargs["data"]["language"] == "en"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_transcription_with_options(self, mock_make_request):
        """Test audio transcription with additional options."""
        # Mock response
        mock_response = {
            "text": "This is a transcription test.",
            "language": "en"
        }
        mock_make_request.return_value = mock_response

        # Test with bytes
        test_audio = b"fake audio data"
        result = await self.provider.create_transcription(
            file=test_audio,
            model="whisper-1",
            prompt="This is a prompt",
            response_format="json",
            temperature=0.7
        )

        # Check result - should be converted to TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a transcription test."
        assert result.language == "en"  # Extra field comes through

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.kwargs["files"] is not None
        assert call_args.kwargs["data"] is not None
        assert call_args.kwargs["data"]["model"] == "whisper-1"
        assert call_args.kwargs["data"]["prompt"] == "This is a prompt"
        assert call_args.kwargs["data"]["response_format"] == "json"
        assert call_args.kwargs["data"]["temperature"] == 0.7

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_translation(self, mock_make_request):
        """Test audio translation functionality."""
        # Mock response
        mock_response = {
            "text": "This is a translation test."
        }
        mock_make_request.return_value = mock_response

        # Mock file handling
        test_audio = b"fake audio data"

        # Test with bytes
        result = await self.provider.create_translation(
            file=test_audio,
            model="whisper-1"
        )

        # Check result - should be converted to TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a translation test."

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.kwargs["files"] is not None
        assert call_args.kwargs["data"] is not None
        assert call_args.kwargs["data"]["model"] == "whisper-1"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_translation_with_options(self, mock_make_request):
        """Test audio translation with additional options."""
        # Mock response
        mock_response = {
            "text": "This is a translation test."
        }
        mock_make_request.return_value = mock_response

        # Create a BytesIO object
        file_obj = BytesIO(b"fake audio data")

        # Test with file-like object
        result = await self.provider.create_translation(
            file=file_obj,
            model="whisper-1",
            prompt="Translation prompt",
            response_format="text",
            temperature=0.5
        )

        # Check result - should be converted to TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a translation test."

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.kwargs["files"] is not None
        assert call_args.kwargs["data"] is not None
        assert call_args.kwargs["data"]["model"] == "whisper-1"
        assert call_args.kwargs["data"]["prompt"] == "Translation prompt"
        assert call_args.kwargs["data"]["response_format"] == "text"
        assert call_args.kwargs["data"]["temperature"] == 0.5

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request_raw')
    async def test_download_file(self, mock_make_request_raw):
        """Test file download functionality."""
        # Mock response
        mock_file_data = b'{"prompt": "test", "completion": "result"}'
        mock_make_request_raw.return_value = mock_file_data

        # Set up an actual API key for successful authentication
        provider = OpenAIProvider(api_key="sk-actual-test-key")

        # Mock _handle_error_response to prevent actual API calls
        with patch.object(OpenAIProvider, '_handle_error_response'):
            # Call the method
            result = await provider.download_file(
                file_id="file-test123"
            )

            # Check result
            assert result == mock_file_data

        # Verify request parameters
        mock_make_request_raw.assert_called_once()
        call_args = mock_make_request_raw.call_args
        assert call_args.args[0] == "GET"
        assert "files/file-test123/content" in call_args.args[1]

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_list_files(self, mock_make_request):
        """Test file listing functionality."""
        # Mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "file-test123",
                    "object": "file",
                    "purpose": "fine-tune",
                    "filename": "test.jsonl",
                    "bytes": 1024,
                    "created_at": 1677858242,
                    "status": "processed"
                }
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.list_files(purpose="fine-tune")

        # Check result
        assert result == mock_response

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.args[0] == "GET"
        assert "files" in call_args.args[1]
        assert call_args.kwargs["data"] == {"purpose": "fine-tune"}

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_delete_file(self, mock_make_request):
        """Test file deletion functionality."""
        # Mock response
        mock_response = {
            "id": "file-test123",
            "object": "file",
            "deleted": True
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.delete_file(file_id="file-test123")

        # Check result
        assert result == mock_response

        # Verify request parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args.args[0] == "DELETE"
        assert "files/file-test123" in call_args.args[1]

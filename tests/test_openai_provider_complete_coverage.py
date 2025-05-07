#!/usr/bin/env python

"""
Tests for complete coverage of the OpenAI provider.

This file targets specific uncovered sections in the OpenAI provider implementation:
- Lines 162-216: File upload with multipart/form-data
- Lines 233-273: Response handling
- Lines 533-538, 546-556: Raw request handling
- Lines 679-698: Download file error handling
- Lines 759-788: Audio transcription
- Lines 816-845: Audio translation
- Lines 945-969: Raw request error handling
- Lines 1064-1146: Image generation (DALL-E models)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from muxi_llm.errors import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
)
from muxi_llm.providers.openai import OpenAIProvider


# Helper function to check if a dictionary has required keys
def has_keys(obj, keys):
    """Check if an object has all the specified keys."""
    if not isinstance(obj, dict):
        return False
    return all(key in obj for key in keys)


class MockResponse:
    """Mock aiohttp.ClientResponse for testing."""

    def __init__(self, data=None, status=200, content_type="application/json"):
        """Initialize mock response."""
        self.status = status
        self._content_type = content_type
        self._data = data

        # For raw data
        if isinstance(data, bytes):
            self._content = [data]
        elif isinstance(data, str):
            self._content = [data.encode('utf-8')]
        elif isinstance(data, dict):
            self._content = [json.dumps(data).encode('utf-8')]
        else:
            self._content = []

    async def json(self):
        """Get response as JSON."""
        if isinstance(self._data, dict):
            return self._data
        if isinstance(self._data, str):
            return json.loads(self._data)
        if isinstance(self._data, bytes):
            return json.loads(self._data.decode('utf-8'))
        return {}

    async def text(self):
        """Get response as text."""
        if isinstance(self._data, str):
            return self._data
        if isinstance(self._data, bytes):
            return self._data.decode('utf-8')
        if isinstance(self._data, dict):
            return json.dumps(self._data)
        return ""

    async def read(self):
        """Get response as bytes."""
        if isinstance(self._data, bytes):
            return self._data
        if isinstance(self._data, str):
            return self._data.encode('utf-8')
        if isinstance(self._data, dict):
            return json.dumps(self._data).encode('utf-8')
        return b''

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    @property
    def content(self):
        """Content property that returns self for async iteration."""
        return self

    async def __aiter__(self):
        """Support async iteration for streaming."""
        for chunk in self._content:
            yield chunk


class TestOpenAIProviderCompleteCoverage:
    """Tests targeting uncovered lines in the OpenAI provider."""

    def setup_method(self):
        """Set up test environment."""
        self.provider = OpenAIProvider(api_key="test-api-key")
        # Patch aiohttp.ClientSession to avoid actual API calls
        self.session_patch = patch('aiohttp.ClientSession')
        self.mock_session = self.session_patch.start()
        self.mock_session_instance = MagicMock()
        self.mock_session.return_value = self.mock_session_instance
        self.mock_session_instance.__aenter__.return_value = self.mock_session_instance
        self.mock_session_instance.__aexit__.return_value = None

    def teardown_method(self):
        """Clean up after tests."""
        self.session_patch.stop()

    @pytest.mark.asyncio
    async def test_make_request_with_files(self):
        """Test _make_request with file uploads (lines 162-216)."""
        # Setup file data
        files = {
            "file": {
                "data": b"test file content",
                "filename": "test.txt",
                "content_type": "text/plain"
            }
        }

        # Additional form data
        data = {
            "purpose": "assistants",
            "metadata": {"key": "value"}
        }

        # Mock response
        mock_response = MockResponse({
            "id": "file-123",
            "object": "file",
            "bytes": 16,
            "created_at": 1677858242,
            "filename": "test.txt",
            "purpose": "assistants"
        })

        # Set up the mock session to return our mock response
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Call the method
        result = await self.provider._make_request(
            method="POST",
            path="/files",
            data=data,
            files=files
        )

        # Verify request was made correctly
        self.mock_session_instance.request.assert_called_once()
        # Content-Type header should be removed for multipart uploads
        assert "Content-Type" not in self.mock_session_instance.request.call_args[1]["headers"]
        # Verify correct path
        assert "/files" in self.mock_session_instance.request.call_args[1]["url"]

        # Verify result
        assert result["id"] == "file-123"
        assert result["purpose"] == "assistants"

    @pytest.mark.asyncio
    async def test_make_request_raw_success(self):
        """Test _make_request_raw successful call (lines 917-944)."""
        expected_data = b"raw binary data"
        mock_response = MockResponse(expected_data)
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Call the method
        result = await self.provider._make_request_raw(
            method="GET",
            path="/raw-endpoint"
        )

        # Verify request was made correctly
        self.mock_session_instance.request.assert_called_once()
        # Verify result
        assert result == expected_data

    @pytest.mark.asyncio
    async def test_make_request_raw_error_json(self):
        """Test _make_request_raw with error response as JSON (lines 945-969)."""
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }
        mock_response = MockResponse(error_response, status=401)
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Call the method and expect an error
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider._make_request_raw(
                method="GET",
                path="/raw-endpoint"
            )

        # Verify error message
        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_request_raw_error_non_json(self):
        """Test _make_request_raw with non-JSON error response (lines 957-965)."""
        error_text = "Internal Server Error"
        mock_response = MockResponse(error_text, status=500)
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Call the method and expect an error
        with pytest.raises(APIError) as exc_info:
            await self.provider._make_request_raw(
                method="GET",
                path="/raw-endpoint"
            )

        # Verify error message
        assert "Internal Server Error" in str(exc_info.value)
        assert "status code: 500" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Cannot easily mock aiohttp.ClientSession inside download_file")
    async def test_download_file_error_handling(self):
        """Test download_file error handling (lines 679-698)."""
        # Since we can't control the aiohttp.ClientSession inside download_file
        # This test is marked as expected to fail
        with pytest.raises(
                InvalidRequestError) as exc_info:
            await self.provider.download_file(file_id="non-existent-file")

        # Verify error message if the test somehow passes
        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_speech_validation(self):
        """Test create_speech parameter validation (lines 974-1037)."""
        # Test with invalid model
        with pytest.raises(
                InvalidRequestError, match="not a supported TTS model"):
            await self.provider.create_speech(
                input="Hello world",
                model="unsupported-model"
            )

        # Test with invalid voice
        with pytest.raises(
                InvalidRequestError, match="Voice 'invalid' is not supported"):
            await self.provider.create_speech(
                input="Hello world",
                voice="invalid"
            )

        # Test with invalid response format
        with pytest.raises(
                InvalidRequestError, match="Response format 'invalid' is not supported"):
            await self.provider.create_speech(
                input="Hello world",
                response_format="invalid",
            )

        # Test with invalid speed (too low)
        with pytest.raises(
                InvalidRequestError, match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech(
                input="Hello world",
                speed=0.1
            )

        # Test with invalid speed (too high)
        with pytest.raises(
                InvalidRequestError, match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech(
                input="Hello world",
                speed=5.0
            )

        # Test with invalid speed (wrong type)
        with pytest.raises(
                InvalidRequestError, match="Speed must be a number between 0.25 and 4.0"):
            await self.provider.create_speech(
                input="Hello world",
                speed="fast"
            )

    @pytest.mark.asyncio
    async def test_create_image_validation(self):
        """Test create_image parameter validation (lines 1038-1146)."""
        # Set up a successful response for valid calls
        image_response = {
            "created": 1677858242,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "b64_json": None,
                    "revised_prompt": "A beautiful sunset"
                }
            ]
        }

        # Test with invalid model
        with pytest.raises(
                InvalidRequestError, match="not a supported image generation model"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="unsupported-model"
            )

        # Test with invalid size for DALL-E 3
        with pytest.raises(
                InvalidRequestError, match="Size '256x256' is not supported for dall-e-3"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="dall-e-3",
                size="256x256"
            )

        # Test with invalid size for DALL-E 2
        with pytest.raises(
                InvalidRequestError, match="Size '1792x1024' is not supported for dall-e-2"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="dall-e-2",
                size="1792x1024"
            )

        # Test with multiple images for DALL-E 3 (not supported)
        with pytest.raises(
                InvalidRequestError, match="DALL-E 3 only supports generating one image at a time"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="dall-e-3",
                n=2
            )

        # Test with invalid quality for DALL-E 3
        with pytest.raises(
                InvalidRequestError, match="Quality 'ultra' is not supported"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="dall-e-3",
                quality="ultra"
            )

        # Test with invalid style for DALL-E 3
        with pytest.raises(
                InvalidRequestError, match="Style 'abstract' is not supported"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                model="dall-e-3",
                style="abstract"
            )

        # Test with invalid response format
        with pytest.raises(
                InvalidRequestError, match="Response format 'png' is not supported"):
            await self.provider.create_image(
                prompt="A beautiful sunset",
                response_format="png"
            )

        # Test successful call with valid parameters
        mock_response = MockResponse(image_response)
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        result = await self.provider.create_image(
            prompt="A beautiful sunset",
            model="dall-e-3",
            size="1024x1024",
            quality="hd",
            style="vivid"
        )

        # Verify result by checking structure instead of using isinstance
        assert has_keys(result, ["created", "data"])
        assert len(result["data"]) == 1
        assert result["data"][0]["url"] == "https://example.com/image.png"
        assert result["data"][0]["revised_prompt"] == "A beautiful sunset"

    @pytest.mark.asyncio
    async def test_create_transcription(self):
        """Test audio transcription (lines 739-796)."""
        # Mock response for successful transcription
        mock_response = MockResponse({
            "text": "This is a transcription of audio content.",
            "language": "en"
        })
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Mock the _process_audio_file method to avoid actual file handling
        with patch.object(
            self.provider, '_process_audio_file',
            return_value=(b"mock audio data", "audio.mp3")
        ):
            # Call the method with minimum parameters
            result = await self.provider.create_transcription(
                file=b"mock audio content",
                model="whisper-1"
            )

            # Verify result using dictionary access instead of isinstance
            assert "text" in result
            assert result["text"] == "This is a transcription of audio content."
            assert "language" in result
            assert result["language"] == "en"

            # Call with all parameters to cover more paths
            result = await self.provider.create_transcription(
                file="path/to/audio.mp3",
                model="whisper-1",
                prompt="This is a test.",
                response_format="text",
                temperature=0.5,
                language="en"
            )

            # Verify API was called with correct parameters
            call_args = self.mock_session_instance.request.call_args[1]
            assert call_args["method"] == "POST"
            assert "/audio/transcriptions" in call_args["url"]

    @pytest.mark.asyncio
    async def test_create_translation(self):
        """Test audio translation (lines 797-853)."""
        # Mock response for successful translation
        mock_response = MockResponse({
            "text": "This is a translation of audio content."
        })
        self.mock_session_instance.request.return_value.__aenter__.return_value = mock_response

        # Mock the _process_audio_file method to avoid actual file handling
        with patch.object(
            self.provider, '_process_audio_file',
            return_value=(b"mock audio data", "audio.mp3")
        ):
            # Call the method with minimum parameters
            result = await self.provider.create_translation(
                file=b"mock audio content",
                model="whisper-1"
            )

            # Verify result using dictionary access instead of isinstance
            assert "text" in result
            assert result["text"] == "This is a translation of audio content."

            # Call with all parameters to cover more paths
            result = await self.provider.create_translation(
                file="path/to/audio.mp3",
                model="whisper-1",
                prompt="This is a test.",
                response_format="text",
                temperature=0.5
            )

            # Verify API was called with correct parameters
            call_args = self.mock_session_instance.request.call_args[1]
            assert call_args["method"] == "POST"
            assert "/audio/translations" in call_args["url"]

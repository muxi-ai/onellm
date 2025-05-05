"""
Tests for additional OpenAI provider functionality.

These tests cover more advanced features like image generation, transcription,
file handling, and vision model support.
"""

import os
import pytest
import mock
from unittest.mock import AsyncMock, patch, MagicMock, mock_open
from io import BytesIO

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.errors import InvalidRequestError, AuthenticationError


class TestOpenAIProviderAdvanced:
    """Advanced tests for the OpenAI provider."""

    def setUp(self):
        """Set up test environment with mocked API key."""
        self.api_key = "sk-test-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    @pytest.fixture(autouse=True)
    def setup_provider(self):
        """Set up provider instance before each test."""
        self.api_key = "sk-test-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

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

        # Check results
        assert result.created == 1677858242
        assert len(result.data) == 1
        assert result.data[0]["url"] == "https://example.com/image.png"

        # Check API call
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args

        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "/images/generations"
        assert kwargs["data"]["prompt"] == "A test image"
        assert kwargs["data"]["model"] == "dall-e-3"
        assert kwargs["data"]["size"] == "1024x1024"
        assert kwargs["data"]["quality"] == "standard"
        assert kwargs["data"]["style"] == "vivid"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_create_image_validation(self, mock_make_request):
        """Test validation of image generation parameters."""
        # Test invalid model
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="invalid-model"
            )

        # Test invalid size for DALL-E 3
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                size="256x256"  # Not valid for DALL-E 3
            )

        # Test n>1 for DALL-E 3
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                n=2  # DALL-E 3 only supports n=1
            )

        # Test invalid quality
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                quality="ultra"  # Not a valid quality
            )

        # Test invalid style
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                style="artistic"  # Not a valid style
            )

        # Test invalid response format
        with pytest.raises(InvalidRequestError):
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                response_format="png"  # Not a valid format
            )

        # Make sure mock_make_request was not called
        mock_make_request.assert_not_called()

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
        with patch("builtins.open", mock_open(read_data=test_audio)):
            # Test with file path
            result = await self.provider.create_transcription(
                file="test.mp3",
                model="whisper-1",
                language="en"
            )

            # Check result
            assert result.text == "This is a transcription test."

            # Check API call
            mock_make_request.assert_called_once()
            _, kwargs = mock_make_request.call_args
            assert kwargs["method"] == "POST"
            assert kwargs["path"] == "/audio/transcriptions"
            assert kwargs["data"]["model"] == "whisper-1"
            assert kwargs["data"]["language"] == "en"
            assert "file" in kwargs["files"]

            # Reset mock
            mock_make_request.reset_mock()

            # Test with bytes
            result = await self.provider.create_transcription(
                file=test_audio,
                model="whisper-1"
            )

            # Check result
            assert result.text == "This is a transcription test."

            # Reset mock
            mock_make_request.reset_mock()

            # Test with file-like object
            file_obj = BytesIO(test_audio)
            file_obj.name = "test.mp3"

            result = await self.provider.create_transcription(
                file=file_obj,
                model="whisper-1"
            )

            # Check result
            assert result.text == "This is a transcription test."

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

        # Check result
        assert result.text == "This is a translation test."

        # Check API call
        mock_make_request.assert_called_once()
        _, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "/audio/translations"
        assert kwargs["data"]["model"] == "whisper-1"
        assert "file" in kwargs["files"]

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_file_upload(self, mock_make_request):
        """Test file upload functionality."""
        # Mock response
        mock_response = {
            "id": "file-test123",
            "object": "file",
            "bytes": 1024,
            "created_at": 1677858242,
            "filename": "test.jsonl",
            "purpose": "fine-tune",
            "status": "uploaded",
            "status_details": None
        }
        mock_make_request.return_value = mock_response

        # Mock file data
        test_file_data = b'{"prompt": "test", "completion": "result"}'

        # Test with file path
        with patch("builtins.open", mock_open(read_data=test_file_data)):
            result = await self.provider.upload_file(
                file="test.jsonl",
                purpose="fine-tune"
            )

            # Check result
            assert result.id == "file-test123"
            assert result.filename == "test.jsonl"
            assert result.purpose == "fine-tune"

            # Check API call
            mock_make_request.assert_called_once()
            _, kwargs = mock_make_request.call_args
            assert kwargs["method"] == "POST"
            assert kwargs["path"] == "/files"
            assert kwargs["data"]["purpose"] == "fine-tune"
            assert "file" in kwargs["files"]

            # Reset mock
            mock_make_request.reset_mock()

            # Test with bytes
            result = await self.provider.upload_file(
                file=test_file_data,
                purpose="fine-tune",
                filename="test2.jsonl"
            )

            # Check filename handling
            _, kwargs = mock_make_request.call_args
            assert kwargs["files"]["file"]["filename"] == "test2.jsonl"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request_raw')
    async def test_download_file(self, mock_make_request_raw):
        """Test file download functionality."""
        # Mock response
        mock_file_data = b'{"prompt": "test", "completion": "result"}'
        mock_make_request_raw.return_value = mock_file_data

        # Call the method
        result = await self.provider.download_file(
            file_id="file-test123"
        )

        # Check result
        assert result == mock_file_data

        # Check that the coroutine was awaited
        mock_make_request_raw.assert_awaited_once()

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_vision_model_processing(self, mock_make_request):
        """Test vision model message processing."""
        # Mock successful response
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4-vision-preview",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I see an image of a cat."
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120
            }
        }
        mock_make_request.return_value = mock_response

        # Vision model with image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # Test valid vision model
        await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-4-vision-preview"
        )

        # Check messages were processed correctly
        mock_make_request.assert_called_once()
        _, kwargs = mock_make_request.call_args
        processed_messages = kwargs["data"]["messages"]

        # Verify image url is preserved
        assert processed_messages[0]["content"][1]["type"] == "image_url"
        assert processed_messages[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"

        # Reset mock
        mock_make_request.reset_mock()

        # Test sending image to non-vision model
        with pytest.raises(InvalidRequestError):
            await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-3.5-turbo"  # Not a vision model
            )

        # Check method wasn't called
        mock_make_request.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request_raw')
    async def test_create_speech(self, mock_make_request_raw):
        """Test text-to-speech capability."""
        # Mock response
        mock_audio_data = b"fake audio data"
        mock_make_request_raw.return_value = mock_audio_data

        # Call the method
        result = await self.provider.create_speech(
            input="Hello, this is a test speech",
            model="tts-1",
            voice="nova",
            response_format="mp3",
            speed=1.2
        )

        # Check result
        assert result == mock_audio_data

        # Check API call
        mock_make_request_raw.assert_called_once()
        _, kwargs = mock_make_request_raw.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "/audio/speech"
        assert kwargs["data"]["model"] == "tts-1"
        assert kwargs["data"]["voice"] == "nova"
        assert kwargs["data"]["response_format"] == "mp3"
        assert kwargs["data"]["speed"] == 1.2

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_streaming_chat_completion(self, mock_make_request):
        """Test streaming chat completion."""
        # Define stream chunks
        stream_chunks = [
            {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1691813142, "model": "gpt-4",
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1691813142, "model": "gpt-4",
             "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1691813142, "model": "gpt-4",
             "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}]}
        ]

        # Set up mock to return an async generator
        async def mock_stream():
            for chunk in stream_chunks:
                yield chunk

        mock_make_request.return_value = mock_stream()

        # Call with streaming=True
        response_stream = await self.provider.create_chat_completion(
            messages=[{"role": "user", "content": "Say hello"}],
            model="gpt-4",
            stream=True
        )

        # Collect chunks from the stream
        collected_chunks = []
        async for chunk in response_stream:
            collected_chunks.append(chunk)

        # Verify we got 3 chunks
        assert len(collected_chunks) == 3

        # Check first chunk has assistant role
        assert collected_chunks[0].choices[0].delta.role == "assistant"

        # Check second chunk has content
        assert collected_chunks[1].choices[0].delta.content == "Hello"

        # Check last chunk has finish reason
        assert collected_chunks[2].choices[0].finish_reason == "stop"

        # Verify API call
        mock_make_request.assert_called_once()
        _, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "chat/completions"
        assert kwargs["stream"] == True

    @pytest.mark.asyncio
    async def test_error_handling_responses(self):
        """Test error handling for various status codes."""
        # Setup various mocked error responses to test
        error_cases = [
            (401, AuthenticationError, "Authentication error"),
            (403, "PermissionError", "Permission denied"),
            (404, "ResourceNotFoundError", "Resource not found"),
            (429, "RateLimitError", "Rate limit exceeded"),
            (400, "InvalidRequestError", "Invalid request"),
            (500, "ServiceUnavailableError", "Server error"),
            (502, "BadGatewayError", "Bad gateway"),
            (504, "TimeoutError", "Request timed out"),
            (418, "APIError", "Unknown error")  # Test unhandled status code
        ]

        for status_code, error_type, error_msg in error_cases:
            # Create mock aiohttp response
            mock_response = MagicMock()
            mock_response.status = status_code
            mock_response.json.return_value = {"error": {"message": error_msg}}

            # Call the error handler
            with pytest.raises(Exception) as excinfo:
                self.provider._handle_error_response(status_code, {"error": {"message": error_msg}})

            # Check error type and message
            assert error_msg in str(excinfo.value)
            if isinstance(error_type, str):
                assert error_type in excinfo.value.__class__.__name__
            else:
                assert isinstance(excinfo.value, error_type)

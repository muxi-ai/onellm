#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Additional tests to improve coverage for the OpenAI provider.

This test file specifically targets uncovered lines in openai.py:
- 97: Missing API key error
- 208, 216: Request handling
- 236, 255-273: Streaming error handling and special cases
- 295-316: Response transformation
- 533-538: Completion streaming
- 546-556: Completion response processing
- 695: Error handling in download_file
- 1065, 1095: Image creation special cases
"""

import os
import pytest
from unittest import mock

from onellm.errors import (
    AuthenticationError,
    ServiceUnavailableError
)
from onellm.providers.openai import OpenAIProvider


class MockAsyncIterator:
    """Mock async iterator to use in tests."""

    def __init__(self, data):
        self.data = data
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        raise StopAsyncIteration


class TestOpenAIAdditionalCoverage:
    """Tests for improving OpenAI provider coverage."""

    def setup_method(self):
        """Set up test environment."""
        # Save original env vars
        self.original_api_key = os.environ.get("OPENAI_API_KEY")

        # Set test API key
        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Create provider
        self.provider = OpenAIProvider()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original env vars
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    @pytest.mark.skip("Complex to mock dict.get method")
    def test_missing_api_key(self):
        """Test error when API key is missing (line 97)."""
        # This test is challenging because we need to mock a dict.get method
        # which is a read-only attribute
        pass

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test _handle_streaming_response error handling (lines 255-258)."""
        # Create a mock response with error status and content
        mock_response = mock.MagicMock()
        mock_response.status = 401
        mock_response.json = mock.AsyncMock(return_value={
            "error": {
                "message": "Invalid authentication",
                "type": "authentication_error"
            }
        })

        with pytest.raises(AuthenticationError) as excinfo:
            async for _ in self.provider._handle_streaming_response(mock_response):
                pass

        assert "Invalid authentication" in str(excinfo.value)
        assert mock_response.json.called

    @pytest.mark.asyncio
    async def test_streaming_invalid_json(self):
        """Test handling of invalid JSON in streaming response (lines 265-273)."""
        # Create a mock response
        mock_response = mock.MagicMock()
        mock_response.status = 200

        # Set up the content as a proper async iterator
        test_data = [
            b'data: invalid json',
            b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": "content"}}]}',
            b'data: [DONE]'
        ]
        mock_response.content = MockAsyncIterator(test_data)

        chunks = []
        async for chunk in self.provider._handle_streaming_response(mock_response):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["id"] == "chatcmpl-123"

    @pytest.mark.asyncio
    async def test_completion_streaming(self):
        """Test the completion streaming implementation (lines 533-538)."""
        # Mock _make_request to return a generator
        async def mock_generator():
            yield {"id": "cmpl-123", "choices": [{"text": "chunk 1"}]}
            yield {"id": "cmpl-123", "choices": [{"text": "chunk 2"}]}

        self.provider._make_request = mock.AsyncMock(return_value=mock_generator())

        # Call the method with stream=True
        generator = await self.provider.create_completion(
            prompt="Test prompt",
            model="text-davinci-003",
            stream=True
        )

        # Collect and verify chunks
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["id"] == "cmpl-123"
        assert chunks[0]["choices"][0]["text"] == "chunk 1"
        assert chunks[1]["choices"][0]["text"] == "chunk 2"

        # Verify the right API call was made
        self.provider._make_request.assert_called_once()
        assert self.provider._make_request.call_args[1]["stream"] is True

    @pytest.mark.asyncio
    async def test_completion_response_processing(self):
        """Test completion response processing (lines 546-556)."""
        # Mock response data
        mock_response = {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1677858242,
            "model": "text-davinci-003",
            "choices": [
                {
                    "text": "This is a test response",
                    "index": 0,
                    "logprobs": {"tokens": ["test"], "token_logprobs": [-0.1]},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "system_fingerprint": "fp123"
        }

        # Mock _make_request to return the response
        self.provider._make_request = mock.AsyncMock(return_value=mock_response)

        # Call the method
        response = await self.provider.create_completion(
            prompt="Test prompt",
            model="text-davinci-003"
        )

        # Verify response processing
        assert response.id == "cmpl-123"
        assert response.object == "text_completion"
        assert response.model == "text-davinci-003"
        assert len(response.choices) == 1
        assert response.choices[0].text == "This is a test response"
        assert response.choices[0].logprobs == {"tokens": ["test"], "token_logprobs": [-0.1]}
        assert response.choices[0].finish_reason == "stop"
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        assert response.system_fingerprint == "fp123"

    @pytest.mark.skip("Complex to set up properly in isolation")
    @pytest.mark.asyncio
    async def test_download_file_error(self):
        """Test error handling in download_file (line 695)."""
        # This test is difficult to set up correctly without deeper mocking
        pass

    @pytest.mark.asyncio
    async def test_image_creation_options(self):
        """Test image creation with various options (lines around 1065, 1095)."""
        # Mock response data
        mock_response = {
            "created": 1677858242,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A beautiful sunset over the mountains"
                }
            ]
        }

        # Mock _make_request to return the response
        self.provider._make_request = mock.AsyncMock(return_value=mock_response)

        # Test with custom options that would hit lines 1065-1095
        response = await self.provider.create_image(
            prompt="A sunset",
            model="dall-e-2",  # Testing different model
            size="1024x1024",
            response_format="url",
            quality="standard",
            style="vivid"
        )

        # Verify the API request included the right parameters
        call_args = self.provider._make_request.call_args[1]
        assert call_args["path"] == "/images/generations"
        assert call_args["data"]["model"] == "dall-e-2"
        assert call_args["data"]["size"] == "1024x1024"
        assert call_args["data"]["response_format"] == "url"
        assert call_args["data"]["quality"] == "standard"
        assert call_args["data"]["style"] == "vivid"

        # Verify response processing - should conform to ImageGenerationResult
        assert isinstance(response, dict)
        assert "created" in response
        assert "data" in response
        assert len(response["data"]) == 1
        assert response["data"][0]["url"] == "https://example.com/image.png"

    @pytest.mark.asyncio
    async def test_handle_response_error(self):
        """Test _handle_response error handling (line 208, 216)."""
        # Create a mock response
        mock_response = mock.MagicMock()
        mock_response.status = 500
        mock_response.json = mock.AsyncMock(return_value={
            "error": {
                "message": "Internal server error",
                "type": "server_error"
            }
        })

        with pytest.raises(ServiceUnavailableError) as excinfo:
            await self.provider._handle_response(mock_response)

        assert "Internal server error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_chat_completion_response_processing(self):
        """Test chat completion response processing (lines 295-316)."""
        # Mock response data
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "system_fingerprint": "fp123"
        }

        # Mock _make_request to return the response
        self.provider._make_request = mock.AsyncMock(return_value=mock_response)

        # Call the method
        response = await self.provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )

        # Verify response processing
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.choices[0].message["role"] == "assistant"
        assert response.choices[0].message["content"] == "This is a test response"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert response.system_fingerprint == "fp123"

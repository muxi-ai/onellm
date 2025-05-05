"""
Tests for vision capabilities in the OpenAI provider.

These tests verify that the OpenAI provider can correctly handle multi-modal
content with images in chat completion requests.
"""

import unittest
from unittest import mock

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.errors import InvalidRequestError


class TestVisionCapabilities(unittest.TestCase):
    """Tests for vision capabilities in the OpenAI provider."""

    def setUp(self):
        """Set up test environment."""
        # Mock API key for testing
        self.api_key = "test-api-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    def test_process_messages_no_images(self):
        """Test that regular messages without images pass through unchanged."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        processed = self.provider._process_messages_for_vision(messages, "gpt-4")
        self.assertEqual(messages, processed)

    def test_process_messages_with_images_valid_model(self):
        """Test processing messages with images using a vision-capable model."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ]

        # This should not raise an error
        processed = self.provider._process_messages_for_vision(
            messages, "gpt-4-vision-preview"
        )
        self.assertEqual(messages, processed)

    def test_process_messages_with_images_invalid_model(self):
        """Test that using images with non-vision models raises an error."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ]

        # This should raise an InvalidRequestError
        with self.assertRaises(InvalidRequestError):
            self.provider._process_messages_for_vision(messages, "gpt-3.5-turbo")

    def test_process_messages_with_invalid_image_url(self):
        """Test that missing URL in image_url raises an error."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            # Missing url field
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # This should raise an InvalidRequestError
        with self.assertRaises(InvalidRequestError):
            self.provider._process_messages_for_vision(
                messages, "gpt-4-vision-preview"
            )

    def test_process_messages_with_invalid_detail(self):
        """Test that invalid detail value is corrected to 'auto'."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg",
                            "detail": "invalid-value"  # Invalid detail
                        }
                    }
                ]
            }
        ]

        processed = self.provider._process_messages_for_vision(
            messages, "gpt-4-vision-preview"
        )

        # Check that detail was corrected to 'auto'
        self.assertEqual(
            processed[1]["content"][1]["image_url"]["detail"],
            "auto"
        )

    @mock.patch("muxi_llm.providers.openai.OpenAIProvider._make_request")
    async def test_vision_request_formatting(self, mock_make_request):
        """Test that vision requests are formatted correctly for the API."""
        # Mock response from the API
        mock_response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1629988652,
            "model": "gpt-4-vision-preview",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I see an image of a landscape."
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

        # Create a message with an image
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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

        # Call the create_chat_completion method
        await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-4-vision-preview"
        )

        # Check that _make_request was called with the correct arguments
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args

        # Check method and path
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["path"], "chat/completions")

        # Check that our messages were passed correctly
        self.assertEqual(kwargs["data"]["messages"], messages)
        self.assertEqual(kwargs["data"]["model"], "gpt-4-vision-preview")


if __name__ == "__main__":
    unittest.main()

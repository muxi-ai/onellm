"""
Tests for the image generation functionality in the OpenAI provider.

These tests verify that the OpenAI provider can correctly handle
image generation requests.
"""

import unittest
from unittest import mock
import os
import tempfile
import base64

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm import Image
from muxi_llm.errors import InvalidRequestError


class TestImageCapabilities(unittest.TestCase):
    """Tests for image generation capabilities in the OpenAI provider."""

    def setUp(self):
        """Set up test environment."""
        # Mock API key for testing
        self.api_key = "test-api-key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    @mock.patch("muxi_llm.providers.openai.OpenAIProvider._make_request")
    async def test_create_image(self, mock_make_request):
        """Test creating images with OpenAI's DALL-E."""
        # Mock response from the API
        mock_response = {
            "created": 1589478378,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A detailed prompt with revisions"
                }
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.create_image(
            prompt="A beautiful sunset over mountains",
            model="dall-e-3",
            size="1024x1024"
        )

        # Verify the request was made correctly
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[1]["method"], "POST")
        self.assertEqual(call_args[1]["path"], "/images/generations")
        self.assertEqual(call_args[1]["data"]["prompt"], "A beautiful sunset over mountains")
        self.assertEqual(call_args[1]["data"]["model"], "dall-e-3")
        self.assertEqual(call_args[1]["data"]["size"], "1024x1024")

        # Verify the response was processed correctly
        self.assertEqual(result["created"], 1589478378)
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["url"], "https://example.com/image1.png")
        self.assertEqual(result["data"][0]["revised_prompt"], "A detailed prompt with revisions")

    @mock.patch("muxi_llm.providers.openai.OpenAIProvider._make_request")
    async def test_create_image_dall_e_2(self, mock_make_request):
        """Test creating images with DALL-E 2."""
        # Mock response
        mock_response = {
            "created": 1589478378,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"}
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.create_image(
            prompt="A beautiful sunset over mountains",
            model="dall-e-2",
            n=2,
            size="512x512"
        )

        # Verify the request
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[1]["data"]["n"], 2)
        self.assertEqual(call_args[1]["data"]["model"], "dall-e-2")
        self.assertEqual(call_args[1]["data"]["size"], "512x512")

        # Verify the response
        self.assertEqual(len(result["data"]), 2)

    @mock.patch("muxi_llm.providers.openai.OpenAIProvider._make_request")
    async def test_create_image_with_options(self, mock_make_request):
        """Test creating images with additional options."""
        # Mock response
        mock_response = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image1.png"}]
        }
        mock_make_request.return_value = mock_response

        # Call the method with additional options
        result = await self.provider.create_image(
            prompt="A beautiful sunset over mountains",
            model="dall-e-3",
            size="1024x1024",
            quality="hd",
            style="vivid",
            response_format="url",
            user="user123"
        )

        # Verify the request includes the additional options
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[1]["data"]["quality"], "hd")
        self.assertEqual(call_args[1]["data"]["style"], "vivid")
        self.assertEqual(call_args[1]["data"]["response_format"], "url")
        self.assertEqual(call_args[1]["data"]["user"], "user123")

    async def test_create_image_invalid_model(self):
        """Test that invalid models are rejected."""
        with self.assertRaises(InvalidRequestError) as context:
            await self.provider.create_image(
                prompt="A test image",
                model="invalid-model"
            )
        self.assertIn("not a supported image generation model", str(context.exception))

    async def test_create_image_invalid_size(self):
        """Test that invalid sizes are rejected."""
        with self.assertRaises(InvalidRequestError) as context:
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                size="invalid-size"
            )
        self.assertIn("not supported for dall-e-3", str(context.exception))

    async def test_dall_e_3_multiple_images(self):
        """Test that DALL-E 3 rejects multiple images."""
        with self.assertRaises(InvalidRequestError) as context:
            await self.provider.create_image(
                prompt="A test image",
                model="dall-e-3",
                n=2
            )
        self.assertIn("only supports generating one image", str(context.exception))

    @mock.patch("muxi_llm.providers.get_provider")
    @mock.patch("muxi_llm.providers.openai.OpenAIProvider.create_image")
    async def test_image_api_class(self, mock_create_image, mock_get_provider):
        """Test the Image API class."""
        # Set up mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_image.return_value = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image1.png"}]
        }
        mock_get_provider.return_value = mock_provider

        # Call the API class
        result = await Image.create(
            prompt="A beautiful sunset over mountains",
            model="openai/dall-e-3",
            size="1024x1024"
        )

        # Verify the provider was called correctly
        mock_get_provider.assert_called_with("openai")
        mock_provider.create_image.assert_called_once_with(
            "A beautiful sunset over mountains", "dall-e-3", n=1, size="1024x1024"
        )

        # Verify the result
        self.assertEqual(result["created"], 1589478378)
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["url"], "https://example.com/image1.png")

    @mock.patch("muxi_llm.providers.get_provider")
    @mock.patch("muxi_llm.image.Image._download_image")
    async def test_image_save_to_file(self, mock_download_image, mock_get_provider):
        """Test saving generated images to files."""
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up mocks
            mock_provider = mock.MagicMock()
            mock_provider.create_image.return_value = {
                "created": 1589478378,
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"b64_json": base64.b64encode(b"fake-image-data").decode('utf-8')}
                ]
            }
            mock_get_provider.return_value = mock_provider

            # Mock the download function
            mock_download_image.return_value = b"downloaded-image-data"

            # Call the API class with output_dir
            result = await Image.create(
                prompt="A beautiful sunset over mountains",
                model="openai/dall-e-3",
                n=1,
                size="1024x1024",
                output_dir=temp_dir
            )

            # Verify the images were saved
            self.assertTrue(os.path.exists(result["data"][0]["filepath"]))
            self.assertTrue(os.path.exists(result["data"][1]["filepath"]))

            # Verify the content of the files
            with open(result["data"][0]["filepath"], "rb") as f:
                self.assertEqual(f.read(), b"downloaded-image-data")

            with open(result["data"][1]["filepath"], "rb") as f:
                self.assertEqual(f.read(), b"fake-image-data")


if __name__ == "__main__":
    unittest.main()

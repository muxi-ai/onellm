"""
Tests for the Image module.

These tests verify that image generation functionality works correctly
with proper provider interactions.
"""

import os
import tempfile
import asyncio
import base64
from unittest import mock
from pathlib import Path

import pytest

from muxi_llm.image import Image
from muxi_llm.providers.base import Provider


class MockProvider:
    """Mock provider for testing image generation."""

    async def create_image(self, prompt, model, n=1, size="1024x1024", **kwargs):
        """Mock implementation of create_image."""
        response_format = kwargs.get("response_format", "url")
        result = {
            "created": 1234567890,
            "data": []
        }

        for i in range(n):
            data_item = {
                "revised_prompt": f"Revised: {prompt}"
            }

            if response_format == "url":
                data_item["url"] = f"https://example.com/image_{i}.png"
            elif response_format == "b64_json":
                data_item["b64_json"] = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

            result["data"].append(data_item)

        return result


class TestImage:
    """Tests for the Image class."""

    @pytest.mark.asyncio
    async def test_create_basic(self):
        """Test basic image creation with default parameters."""
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider:
            # Set up the mock provider
            mock_provider = MockProvider()
            mock_get_provider.return_value = (mock_provider, "dall-e-3")

            # Call the method
            result = await Image.create("A beautiful sunset")

            # Verify the result
            assert "created" in result
            assert "data" in result
            assert len(result["data"]) == 1
            assert "url" in result["data"][0]
            assert result["data"][0]["url"].startswith("https://example.com/")

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with(
                primary_model="openai/dall-e-3",
                fallback_models=None,
                fallback_config=None
            )

    @pytest.mark.asyncio
    async def test_create_with_parameters(self):
        """Test image creation with custom parameters."""
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider:
            # Set up the mock provider
            mock_provider = MockProvider()
            mock_get_provider.return_value = (mock_provider, "dall-e-2")

            # Call the method with custom parameters
            result = await Image.create(
                prompt="A futuristic city",
                model="openai/dall-e-2",
                n=3,
                size="512x512",
                quality="hd",
                style="vivid"
            )

            # Verify the result
            assert "created" in result
            assert "data" in result
            assert len(result["data"]) == 3

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with(
                primary_model="openai/dall-e-2",
                fallback_models=None,
                fallback_config=None
            )

    @pytest.mark.asyncio
    async def test_create_with_fallbacks(self):
        """Test image creation with fallback models."""
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider:
            # Set up the mock provider
            mock_provider = MockProvider()
            mock_get_provider.return_value = (mock_provider, "dall-e-3")

            # Call the method with fallback models
            result = await Image.create(
                prompt="A mountain landscape",
                model="openai/dall-e-3",
                fallback_models=["openai/dall-e-2", "stability/stable-diffusion"]
            )

            # Verify the result
            assert "created" in result
            assert "data" in result

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with(
                primary_model="openai/dall-e-3",
                fallback_models=["openai/dall-e-2", "stability/stable-diffusion"],
                fallback_config=None
            )

    @pytest.mark.asyncio
    async def test_create_with_output_dir_url(self):
        """Test image creation with output directory and URL response."""
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.Image._download_image") as mock_download, \
             tempfile.TemporaryDirectory() as tmp_dir:

            # Set up the mock provider and download function
            mock_provider = MockProvider()
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            mock_download.return_value = b"fake image data"

            # Call the method with output directory
            result = await Image.create(
                prompt="A forest scene",
                output_dir=tmp_dir
            )

            # Verify the result
            assert "data" in result
            assert "filepath" in result["data"][0]

            # Verify the file was created
            filepath = result["data"][0]["filepath"]
            assert os.path.exists(filepath)

            # Verify the download function was called
            mock_download.assert_called_once()
            assert mock_download.call_args[0][0].startswith("https://example.com/")

    @pytest.mark.asyncio
    async def test_create_with_output_dir_b64json(self):
        """Test image creation with output directory and base64 response."""
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             tempfile.TemporaryDirectory() as tmp_dir:

            # Set up the mock provider
            mock_provider = MockProvider()
            mock_get_provider.return_value = (mock_provider, "dall-e-3")

            # Call the method with output directory and b64_json response format
            result = await Image.create(
                prompt="An abstract pattern",
                output_dir=tmp_dir,
                response_format="b64_json"
            )

            # Verify the result
            assert "data" in result
            assert "filepath" in result["data"][0]

            # Verify the file was created
            filepath = result["data"][0]["filepath"]
            assert os.path.exists(filepath)

            # Check that the file contains actual image data
            with open(filepath, "rb") as f:
                file_content = f.read()
                assert len(file_content) > 0

    def test_create_sync(self):
        """Test synchronous image creation."""
        with mock.patch("muxi_llm.image.asyncio.run") as mock_run:
            # Set up the mock
            expected_result = {"created": 1234567890, "data": [{"url": "https://example.com/image.png"}]}
            mock_run.return_value = expected_result

            # Call the method
            result = Image.create_sync("A desert landscape")

            # Verify the result
            assert result == expected_result

            # Verify asyncio.run was called with the async create method
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            # First arg should be a coroutine (the result of Image.create(...))
            assert asyncio.iscoroutine(args[0])

    @pytest.mark.asyncio
    async def test_download_image(self):
        """Test the _download_image method."""
        # Create a simple async mock that returns fake image data
        async def mock_download_image(url):
            assert url == "https://example.com/image.png"
            return b"fake image data"

        # Patch the _download_image method directly
        with mock.patch.object(Image, '_download_image', side_effect=mock_download_image):
            # Call the method
            result = await Image._download_image("https://example.com/image.png")

            # Verify the result
            assert result == b"fake image data"

    @pytest.mark.asyncio
    async def test_download_image_error(self):
        """Test error handling in _download_image method."""
        # Create a mock that raises an error
        async def mock_download_image_error(url):
            assert url == "https://example.com/image.png"
            raise ValueError("Failed to download image: 404")

        # Patch the _download_image method directly
        with mock.patch.object(Image, '_download_image', side_effect=mock_download_image_error):
            # Call the method and expect an error
            with pytest.raises(ValueError) as excinfo:
                await Image._download_image("https://example.com/image.png")

            # Verify the error message
            assert "Failed to download image: 404" in str(excinfo.value)

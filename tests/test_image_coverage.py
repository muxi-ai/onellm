#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Additional tests for the image module, focused on coverage improvements.

These tests specifically target code paths that aren't covered by existing tests.
"""

import os
import pytest
import tempfile
from unittest import mock

from muxi_llm.image import Image
from muxi_llm.errors import APIError
from muxi_llm.providers.base import Provider


class MockImageProvider(Provider):
    """Mock provider for testing image generation."""

    def __init__(self, fail=False, response_type="url"):
        """Initialize the mock provider."""
        self.fail = fail
        self.response_type = response_type
        self.call_count = 0

    async def create_chat_completion(self, *args, **kwargs):
        pass

    async def create_completion(self, *args, **kwargs):
        pass

    async def create_embedding(self, *args, **kwargs):
        pass

    async def upload_file(self, *args, **kwargs):
        pass

    async def download_file(self, *args, **kwargs):
        pass

    async def create_image(self, prompt, model, **kwargs):
        """Mock image generation."""
        self.call_count += 1

        if self.fail:
            raise APIError("Image generation failed")

        # Create a timestamp to use in the response
        timestamp = 1234567890

        # Generate appropriate response based on configured type
        if self.response_type == "url":
            return {
                "created": timestamp,
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"url": "https://example.com/image2.png"}
                ]
            }
        elif self.response_type == "b64_json":
            # Simple 1x1 transparent PNG as base64
            b64_data = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8"
                "AAAAASUVORK5CYII="
            )
            return {
                "created": timestamp,
                "data": [
                    {"b64_json": b64_data}
                ]
            }
        else:
            return {"data": []}


class TestImageCoverage:
    """Tests specifically focused on improving coverage for the image module."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temp directory for output images
        self.temp_dir = tempfile.mkdtemp()

        # Patch the get_provider_with_fallbacks function
        self.mock_get_provider = mock.patch("muxi_llm.image.get_provider_with_fallbacks").start()

    def teardown_method(self):
        """Clean up test environment."""
        # Remove temp files
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, filename))
            os.rmdir(self.temp_dir)

        # Stop all mocks
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_image_download_error(self):
        """Test error handling during image download."""
        # Create a mock response with a failing status
        mock_response = mock.AsyncMock()
        mock_response.status = 404

        # Create a context manager that properly yields the mock response
        cm = mock.AsyncMock()
        cm.__aenter__.return_value = mock_response

        # Create a session with a get method that returns our context manager
        mock_session = mock.AsyncMock()
        mock_session.get.return_value = cm

        # Create a session class that yields our mock session
        mock_session_cm = mock.AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_class = mock.Mock(return_value=mock_session_cm)

        # Patch aiohttp.ClientSession
        with mock.patch("aiohttp.ClientSession", mock_session_class):
            # Call _download_image directly
            with pytest.raises(ValueError) as excinfo:
                await Image._download_image("https://example.com/image.png")

            # Verify error message
            assert "Failed to download image: 404" in str(excinfo.value)

            # Verify the session.get was called with the correct URL
            mock_session.get.assert_called_once_with("https://example.com/image.png")

    @pytest.mark.asyncio
    async def test_save_url_images_to_file(self):
        """Test saving URL-based images to files."""
        # Set up mock provider that returns URL images
        provider = MockImageProvider(response_type="url")
        self.mock_get_provider.return_value = (provider, "test-model")

        # Mock the _download_image method to return test image data
        test_image_data = b"mock image data"
        with mock.patch.object(Image, "_download_image", return_value=test_image_data):
            # Call create with output_dir
            result = await Image.create(
                prompt="test prompt",
                model="test/model",
                n=2,
                output_dir=self.temp_dir
            )

            # Verify the result
            assert len(result["data"]) == 2

            # Check that filepath was added to the results
            assert "filepath" in result["data"][0]
            assert "filepath" in result["data"][1]

            # Verify the files were created
            for item in result["data"]:
                filepath = item["filepath"]
                assert os.path.exists(filepath)

                # Verify correct content was written
                with open(filepath, "rb") as f:
                    assert f.read() == test_image_data

    @pytest.mark.asyncio
    async def test_save_base64_images_to_file(self):
        """Test saving base64-encoded images to files."""
        # Set up mock provider that returns base64 images
        provider = MockImageProvider(response_type="b64_json")
        self.mock_get_provider.return_value = (provider, "test-model")

        # Call create with output_dir
        result = await Image.create(
            prompt="test prompt",
            model="test/model",
            output_dir=self.temp_dir,
            output_format="jpg"  # Test custom format
        )

        # Verify the result
        assert len(result["data"]) == 1

        # Check that filepath was added to the results
        assert "filepath" in result["data"][0]

        # Verify the file was created with jpg extension
        filepath = result["data"][0]["filepath"]
        assert os.path.exists(filepath)
        assert filepath.endswith(".jpg")

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self):
        """Test creating images with custom configuration options."""
        # Set up mock provider
        provider = MockImageProvider()
        self.mock_get_provider.return_value = (provider, "dalle-3")

        # Call create with custom parameters
        await Image.create(
            prompt="test prompt",
            model="openai/dall-e-3",
            size="1024x1024",
            n=1,
            quality="hd",
            style="vivid",
            response_format="b64_json",
            user="user-123"
        )

        # Verify the provider was called with the correct parameters
        self.mock_get_provider.assert_called_once()
        # Directly check first arg (primary_model)
        assert self.mock_get_provider.call_args[0][0] == "openai/dall-e-3"

        # Create was called once
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_missing_image_data(self):
        """Test handling of responses with missing image data."""
        # Set up mock provider that returns empty data
        provider = MockImageProvider(response_type="empty")
        self.mock_get_provider.return_value = (provider, "test-model")

        # Call create with output_dir
        result = await Image.create(
            prompt="test prompt",
            model="test/model",
            output_dir=self.temp_dir
        )

        # Verify the result has data but no files were created
        assert "data" in result
        assert len(result["data"]) == 0

        # Verify the directory is empty
        assert len(os.listdir(self.temp_dir)) == 0

    def test_create_sync(self):
        """Test the synchronous version of create."""
        # Set up mock asyncio.run to capture the arguments
        mock_run = mock.Mock()

        with mock.patch("asyncio.run", mock_run):
            # Call create_sync
            Image.create_sync(
                prompt="test prompt",
                model="test/model",
                n=2,
                size="512x512"
            )

            # Verify asyncio.run was called with create
            assert mock_run.call_count == 1
            # Get the coroutine that was passed to asyncio.run
            coro = mock_run.call_args[0][0]
            # Verify it's a create coroutine with the correct arguments
            assert coro.cr_frame.f_locals["prompt"] == "test prompt"
            assert coro.cr_frame.f_locals["model"] == "test/model"
            assert coro.cr_frame.f_locals["n"] == 2
            assert coro.cr_frame.f_locals["size"] == "512x512"

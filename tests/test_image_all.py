"""
Comprehensive test file to achieve 100% coverage for image.py in muxi-llm.
"""

import pytest
import base64
import os
import tempfile
from unittest import mock

from muxi_llm import Image


class TestImageComprehensive:
    """Tests to achieve 100% coverage of image.py."""

    @pytest.mark.asyncio
    async def test_create_with_fallback_config(self):
        """Test creating images with fallback configuration."""
        # Create a mock provider
        mock_provider = mock.MagicMock()
        async def create_image(*args, **kwargs):
            return {
                "created": 1685867264,
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"b64_json": base64.b64encode(b"fake-image-data").decode('utf-8')}
                ]
            }
        mock_provider.create_image = create_image

        # Create a mock fallback config
        mock_fallback_config = mock.MagicMock()

        # Set up mocks
        with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                        return_value=(mock_provider, "dall-e-3")):
            with mock.patch('muxi_llm.image.FallbackConfig', return_value=mock_fallback_config):
                # Call image creation with fallback config
                result = await Image.create(
                    prompt="A beautiful sunset",
                    model="openai/dall-e-3",
                    fallback_models=["anthropic/claude-3-haiku"],
                    fallback_config={"max_fallbacks": 3}
                )

                # Verify the result
                assert "data" in result
                assert len(result["data"]) == 2

    @pytest.mark.asyncio
    async def test_create_with_output_dir(self):
        """Test creating images and saving them to a directory."""
        # Create a mock provider
        mock_provider = mock.MagicMock()
        async def create_image(*args, **kwargs):
            return {
                "created": 1685867264,
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"b64_json": base64.b64encode(b"fake-image-data").decode('utf-8')}
                ]
            }
        mock_provider.create_image = create_image

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the download_image method
            with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                           return_value=(mock_provider, "dall-e-3")):
                with mock.patch.object(Image, '_download_image',
                                       new_callable=mock.AsyncMock,
                                       return_value=b"downloaded-image-data"):
                    with mock.patch('os.makedirs') as makedirs_mock:
                        with mock.patch('builtins.open', mock.mock_open()) as open_mock:
                            # Call the create method with output_dir
                            result = await Image.create(
                                prompt="A beautiful sunset",
                                model="openai/dall-e-3",
                                output_dir=temp_dir
                            )

                            # Verify makedirs was called
                            makedirs_mock.assert_called_once_with(temp_dir, exist_ok=True)

                            # Verify open was called twice (once for each image)
                            assert open_mock.call_count == 2

                            # Verify filepaths were added to the result
                            assert "filepath" in result["data"][0]
                            assert "filepath" in result["data"][1]

    @pytest.mark.asyncio
    async def test_create_without_output_dir(self):
        """Test creating images without an output directory."""
        # Create a mock provider
        mock_provider = mock.MagicMock()
        async def create_image(*args, **kwargs):
            return {
                "created": 1685867264,
                "data": [{"url": "https://example.com/image.png"}]
            }
        mock_provider.create_image = create_image

        # Set up mocks
        with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                       return_value=(mock_provider, "dall-e-3")):
            # Call create without output_dir
            result = await Image.create(
                prompt="A beautiful sunset",
                model="openai/dall-e-3"
            )

            # Verify result has data but no filepath
            assert "data" in result
            assert "filepath" not in result["data"][0]

    def test_create_sync(self):
        """Test the synchronous wrapper for image creation."""
        # Create a mock coroutine to be returned by Image.create
        mock_create_result = {"data": [{"url": "https://example.com/image.png"}]}

        # Set up mocks
        with mock.patch('muxi_llm.image.Image.create',
                       new_callable=mock.AsyncMock,
                       return_value=mock_create_result):
            with mock.patch('muxi_llm.image.asyncio.run',
                           return_value=mock_create_result) as mock_run:
                # Call create_sync
                result = Image.create_sync(
                    prompt="A beautiful sunset",
                    model="openai/dall-e-3"
                )

                # Verify asyncio.run was called
                mock_run.assert_called_once()

                # Verify result
                assert result == mock_create_result

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """Test the _download_image method successfully downloads an image."""
        # Mock response data
        image_data = b"image_data"

        # Create a mock read method that returns our image data
        read_mock = mock.AsyncMock(return_value=image_data)

        # Create mock response
        response_mock = mock.MagicMock()
        response_mock.status = 200
        response_mock.read = read_mock

        # Set up context managers for response and session
        response_cm = mock.MagicMock()
        response_cm.__aenter__ = mock.AsyncMock(return_value=response_mock)
        response_cm.__aexit__ = mock.AsyncMock(return_value=None)

        session_mock = mock.MagicMock()
        session_mock.get = mock.MagicMock(return_value=response_cm)

        session_cm = mock.MagicMock()
        session_cm.__aenter__ = mock.AsyncMock(return_value=session_mock)
        session_cm.__aexit__ = mock.AsyncMock(return_value=None)

        # Create patch for aiohttp.ClientSession
        with mock.patch('aiohttp.ClientSession', return_value=session_cm):
            # Call _download_image and capture the result
            result = await Image._download_image("https://example.com/image.png")

            # Verify result matches our mock data
            assert result == image_data

            # Verify session.get was called with the correct URL
            session_mock.get.assert_called_once_with("https://example.com/image.png")

            # Verify response.read was called
            read_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_image_error(self):
        """Test _download_image handles error responses correctly."""
        # Create mock response with error status
        response_mock = mock.MagicMock()
        response_mock.status = 404

        # Set up context managers
        response_cm = mock.MagicMock()
        response_cm.__aenter__ = mock.AsyncMock(return_value=response_mock)
        response_cm.__aexit__ = mock.AsyncMock(return_value=None)

        session_mock = mock.MagicMock()
        session_mock.get = mock.MagicMock(return_value=response_cm)

        session_cm = mock.MagicMock()
        session_cm.__aenter__ = mock.AsyncMock(return_value=session_mock)
        session_cm.__aexit__ = mock.AsyncMock(return_value=None)

        # Create patch for aiohttp.ClientSession
        with mock.patch('aiohttp.ClientSession', return_value=session_cm):
            # Call _download_image and expect a ValueError
            with pytest.raises(ValueError, match="Failed to download image: 404"):
                await Image._download_image("https://example.com/image.png")

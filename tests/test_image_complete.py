"""
Tests for complete coverage of image.py in muxi-llm.

These tests target full coverage of the image generation functionality.
"""

import pytest
import os
import tempfile
import asyncio
from unittest import mock
import base64

from muxi_llm import Image


class TestImageComplete:
    """Tests for achieving complete coverage of image.py."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = mock.MagicMock()

        # Set up the provider's create_image method
        async def create_image_mock(*args, **kwargs):
            return {
                "created": 1685867264,
                "data": [
                    {"url": "https://example.com/image1.png"},
                    {"b64_json": base64.b64encode(b"fake-image-data").decode('utf-8')}
                ]
            }

        provider.create_image = create_image_mock
        return provider

    @pytest.mark.asyncio
    async def test_create_with_output_dir_url(self, mock_provider):
        """Test image creation with output directory and URL-based results."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up mocks
            with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                            return_value=(mock_provider, "dall-e-3")):

                # Create a proper async mock for _download_image
                async def mock_download_impl(url):
                    return b"downloaded-image-data"

                # Use the async mock function
                with mock.patch.object(Image, '_download_image',
                                       side_effect=mock_download_impl) as mock_download:

                    # Call the image creation method with output_dir
                    result = await Image.create(
                        prompt="A beautiful sunset",
                        output_dir=temp_dir
                    )

                    # Verify _download_image was called for URL image
                    mock_download.assert_called_once_with("https://example.com/image1.png")

                    # Verify files were created
                    assert len(result["data"]) == 2
                    assert "filepath" in result["data"][0]
                    assert "filepath" in result["data"][1]

                    # Verify filepath points to a real file that contains our mock data
                    assert os.path.exists(result["data"][0]["filepath"])
                    with open(result["data"][0]["filepath"], "rb") as f:
                        assert f.read() == b"downloaded-image-data"

                    assert os.path.exists(result["data"][1]["filepath"])
                    with open(result["data"][1]["filepath"], "rb") as f:
                        assert f.read() == b"fake-image-data"

    @pytest.mark.asyncio
    async def test_create_with_output_dir_invalid_data(self, mock_provider):
        """Test image creation with output directory and invalid data in response."""
        # Set up a mock provider that returns invalid image data
        invalid_provider = mock.MagicMock()

        async def create_image_invalid(*args, **kwargs):
            return {
                "created": 1685867264,
                "data": [
                    {"no_url_or_b64": "invalid"}  # Missing url or b64_json
                ]
            }

        invalid_provider.create_image = create_image_invalid

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up mocks
            with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                            return_value=(invalid_provider, "dall-e-3")):
                # Call the image creation method with output_dir
                result = await Image.create(
                    prompt="A beautiful sunset",
                    output_dir=temp_dir
                )

                # Verify no filepath was added since the data was invalid
                assert "filepath" not in result["data"][0]

                # No files should be created
                files = os.listdir(temp_dir)
                assert len(files) == 0

    @pytest.mark.asyncio
    async def test_download_image_error(self):
        """Test error handling in the _download_image method."""
        # Mock aiohttp.ClientSession
        mock_session = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.status = 404  # Simulate HTTP error

        # Configure the session to return our mocked response
        mock_session.__aenter__.return_value = mock_session
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Use mock_session in _download_image
        with mock.patch('aiohttp.ClientSession', return_value=mock_session):
            # The method should raise ValueError because status != 200
            with pytest.raises(ValueError, match="Failed to download image: 404"):
                await Image._download_image("https://example.com/nonexistent.png")

    def test_create_sync(self, mock_provider):
        """Test the synchronous wrapper for image creation."""
        # Set up mocks
        with mock.patch('muxi_llm.image.get_provider_with_fallbacks',
                        return_value=(mock_provider, "dall-e-3")):
            with mock.patch('asyncio.run',
                            side_effect=lambda coro:
                                asyncio.get_event_loop().run_until_complete(coro)):
                # Call the synchronous method
                result = Image.create_sync(
                    prompt="A beautiful sunset"
                )

                # Verify it returned the expected result
                assert "data" in result
                assert len(result["data"]) == 2
                assert "url" in result["data"][0]
                assert "b64_json" in result["data"][1]

"""
Test file for handling errors in the _download_image method of image.py
"""

import pytest
from unittest import mock

from muxi_llm import Image


class TestImageErrorHandling:
    """Tests for error handling in image.py module."""

    @pytest.mark.asyncio
    async def test_download_image_error_handling(self):
        """Test that _download_image properly handles HTTP error responses."""
        # Create a mock response with error status
        mock_response = mock.MagicMock()
        mock_response.status = 404  # Not Found

        # Create a mock context manager for the response
        mock_response_cm = mock.MagicMock()
        mock_response_cm.__aenter__ = mock.AsyncMock(return_value=mock_response)
        mock_response_cm.__aexit__ = mock.AsyncMock(return_value=None)

        # Create a mock session with a get method
        mock_session = mock.MagicMock()
        mock_session.get = mock.MagicMock(return_value=mock_response_cm)

        # Create a mock context manager for the session
        mock_session_cm = mock.MagicMock()
        mock_session_cm.__aenter__ = mock.AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = mock.AsyncMock(return_value=None)

        # Replace aiohttp.ClientSession with our mock
        with mock.patch('aiohttp.ClientSession', return_value=mock_session_cm):
            # Call _download_image with a URL and expect it to raise a ValueError
            with pytest.raises(ValueError, match=r"Failed to download image: 404"):
                await Image._download_image("https://example.com/image.png")

            # Verify that session.get was called with the correct URL
            mock_session.get.assert_called_once_with("https://example.com/image.png")

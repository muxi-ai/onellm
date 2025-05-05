"""
Test file to achieve 100% coverage for image.py in muxi-llm.
This file focuses on testing line 175 specifically.
"""

import pytest
from unittest import mock

from muxi_llm import Image


class TestImageFinalAwait:
    """Tests to achieve 100% coverage of image.py by testing line 175."""

    @pytest.mark.asyncio
    async def test_download_image_coroutine_awaited(self):
        """Test that the coroutine returned by response.read() is properly awaited."""
        # Create mock response object
        mock_response = mock.AsyncMock()
        mock_response.status = 200

        # Use a custom async function for read() that we know will be awaited
        read_result = b"image_data"

        async def mock_read():
            """Mock function for read() that returns our test data."""
            # When this coroutine is awaited, it will return read_result
            return read_result

        # Assign our mock read function to the response
        mock_response.read = mock_read

        # Create mock context managers for both response and session
        response_context = mock.AsyncMock()
        response_context.__aenter__.return_value = mock_response
        response_context.__aexit__.return_value = None

        mock_session = mock.AsyncMock()
        mock_session.get.return_value = response_context

        session_context = mock.AsyncMock()
        session_context.__aenter__.return_value = mock_session
        session_context.__aexit__.return_value = None

        # Create a direct implementation of Image._download_image that we can test
        # that doesn't rely on patching the original method
        async def test_implementation():
            """Test implementation of the _download_image method."""
            with mock.patch('aiohttp.ClientSession', return_value=session_context):
                result = await Image._download_image("https://example.com/image.png")
                return result

        # Run the test implementation and check the result
        result = await test_implementation()

        # Verify that our read coroutine was called and awaited
        assert result == read_result

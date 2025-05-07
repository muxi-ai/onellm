"""
Tests for file upload functionality in the OpenAI provider.

This module tests the file upload functionality implemented in the OpenAI provider.
"""

import pytest
from unittest import mock
import io
from unittest.mock import AsyncMock

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.models import FileObject
from muxi_llm.errors import InvalidRequestError


# Create a mock provider for successful file operations testing
class MockFileProvider:
    """Mock provider for file upload tests."""

    def __init__(self, raise_invalid_file=False):
        """Initialize with mock methods."""
        self.upload_file = AsyncMock()

        if raise_invalid_file:
            self.upload_file.side_effect = InvalidRequestError(
                "Invalid file type. Expected file path, bytes, or file-like object."
            )
        else:
            self.upload_file.return_value = FileObject(
                id="file-abc123",
                object="file",
                bytes=1024,
                created_at=1677858242,
                filename="test.txt",
                purpose="assistants",
                status="processed"
            )


class TestOpenAIFileUpload:
    """Test targeting file upload functionality in the OpenAI provider (lines 591-649)."""

    def setup_method(self):
        """Set up test environment."""
        self.provider = OpenAIProvider(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_upload_file_with_bytes(self):
        """Test uploading a file using bytes (lines 591-625)."""
        # Create test file content
        file_content = b"This is a test file content."

        # Set up mock response
        with mock.patch.object(
                self.provider, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = {
                "id": "file-abc123",
                "object": "file",
                "bytes": len(file_content),
                "created_at": 1677858242,
                "filename": "test.txt",
                "purpose": "assistants",
                "status": "processed"
            }

            # Call the method with bytes
            result = await self.provider.upload_file(
                file=file_content,
                purpose="assistants",
                filename="test.txt"
            )

            # Verify API was called correctly
            mock_make_request.assert_called_once()
            call_args = mock_make_request.call_args[1]
            assert call_args["method"] == "POST"
            assert call_args["path"] == "/files"
            assert "files" in call_args
            assert call_args["files"]["file"]["data"] == file_content
            assert call_args["files"]["file"]["filename"] == "test.txt"
            assert call_args["data"]["purpose"] == "assistants"

            # Verify result is correct
            assert isinstance(result, FileObject)
            assert result.id == "file-abc123"
            assert result.bytes == len(file_content)
            assert result.filename == "test.txt"
            assert result.purpose == "assistants"
            assert result.status == "processed"

    @pytest.mark.asyncio
    async def test_upload_file_with_path(self):
        """Test uploading a file using a file path (lines 595-625)."""
        # Mock file content
        mock_file_content = b"This is test content from a file path."

        # Set up mock response
        with mock.patch.object(
                self.provider, '_make_request', new_callable=AsyncMock) as mock_make_request, \
             mock.patch('builtins.open', mock.mock_open(read_data=mock_file_content)):

            mock_make_request.return_value = {
                "id": "file-xyz456",
                "object": "file",
                "bytes": len(mock_file_content),
                "created_at": 1677858242,
                "filename": "test_file.txt",
                "purpose": "assistants",
                "status": "processed"
            }

            # Call the method with a file path string
            result = await self.provider.upload_file(
                file="path/to/test_file.txt",
                purpose="assistants"
            )

            # Verify API was called correctly
            mock_make_request.assert_called_once()
            call_args = mock_make_request.call_args[1]
            assert call_args["method"] == "POST"
            assert call_args["path"] == "/files"
            assert "files" in call_args
            assert call_args["data"]["purpose"] == "assistants"

            # Verify result is correct
            assert isinstance(result, FileObject)
            assert result.id == "file-xyz456"
            assert result.filename == "test_file.txt"
            assert result.purpose == "assistants"

    @pytest.mark.asyncio
    async def test_upload_file_with_file_object(self):
        """Test uploading a file using a file-like object (lines 595-625)."""
        # Create a file-like object
        file_content = b"This is test content from a file-like object."
        file_obj = io.BytesIO(file_content)

        # Set up mock response
        with mock.patch.object(
                self.provider, '_make_request', new_callable=AsyncMock) as mock_make_request:
            mock_make_request.return_value = {
                "id": "file-io789",
                "object": "file",
                "bytes": len(file_content),
                "created_at": 1677858242,
                "filename": "uploaded_file.txt",
                "purpose": "assistants",
                "status": "processed"
            }

            # Call the method with a file-like object
            result = await self.provider.upload_file(
                file=file_obj,
                purpose="assistants",
                filename="uploaded_file.txt"
            )

            # Verify API was called correctly
            mock_make_request.assert_called_once()
            call_args = mock_make_request.call_args[1]
            assert call_args["method"] == "POST"
            assert call_args["path"] == "/files"
            assert "files" in call_args
            assert call_args["data"]["purpose"] == "assistants"

            # Verify result is correct
            assert isinstance(result, FileObject)
            assert result.id == "file-io789"
            assert result.filename == "uploaded_file.txt"
            assert result.purpose == "assistants"

    @pytest.mark.asyncio
    async def test_upload_file_with_invalid_file(self):
        """Test uploading with an invalid file type (line 647-649)."""
        # Create a custom class that doesn't have a read method and isn't a string/bytes
        class InvalidFileType:
            """Invalid file type for testing."""
            def __str__(self):
                return "InvalidFileObject"

        invalid_file = InvalidFileType()

        # We don't need to mock _make_request since the validation happens before that
        # The real implementation should raise the error directly
        with pytest.raises(InvalidRequestError) as exc_info:
            await self.provider.upload_file(
                file=invalid_file,
                purpose="assistants"
            )

        # Verify error message
        assert "Invalid file type" in str(exc_info.value)

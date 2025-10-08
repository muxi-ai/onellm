"""
Tests for file operations functionality in the OpenAI provider.

This module tests file upload, listing, deletion, and download functionality.
"""

import os
import io
import pytest
from unittest import mock
from unittest.mock import AsyncMock, patch, mock_open

from onellm.providers.openai import OpenAIProvider
from onellm.models import FileObject
from onellm.errors import InvalidRequestError


class MockResponse:
    """Mock for aiohttp response."""

    def __init__(self, status: int, data: dict):
        self.status = status
        self._data = data

    async def json(self):
        return self._data


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
                status="processed",
            )


class TestOpenAIFileUpload:
    """Test file upload functionality in the OpenAI provider."""

    def setup_method(self):
        """Set up test environment."""
        self.provider = OpenAIProvider(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_upload_file_with_bytes(self):
        """Test uploading a file using bytes."""
        # Create test file content
        file_content = b"This is a test file content."

        # Set up mock response
        with mock.patch.object(
            self.provider, "_make_request", new_callable=AsyncMock
        ) as mock_make_request:
            mock_make_request.return_value = {
                "id": "file-abc123",
                "object": "file",
                "bytes": len(file_content),
                "created_at": 1677858242,
                "filename": "test.txt",
                "purpose": "assistants",
                "status": "processed",
            }

            # Call the method with bytes
            result = await self.provider.upload_file(
                file=file_content, purpose="assistants", filename="test.txt"
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
        """Test uploading a file using a file path."""
        # Mock file content
        mock_file_content = b"This is test content from a file path."

        # Set up mock response
        with (
            mock.patch.object(
                self.provider, "_make_request", new_callable=AsyncMock
            ) as mock_make_request,
            mock.patch("builtins.open", mock_open(read_data=mock_file_content)),
        ):

            mock_make_request.return_value = {
                "id": "file-xyz456",
                "object": "file",
                "bytes": len(mock_file_content),
                "created_at": 1677858242,
                "filename": "test_file.txt",
                "purpose": "assistants",
                "status": "processed",
            }

            # Call the method with a file path string
            result = await self.provider.upload_file(
                file="path/to/test_file.txt", purpose="assistants"
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
        """Test uploading a file using a file-like object."""
        # Create a file-like object
        file_content = b"This is test content from a file-like object."
        file_obj = io.BytesIO(file_content)

        # Set up mock response
        with mock.patch.object(
            self.provider, "_make_request", new_callable=AsyncMock
        ) as mock_make_request:
            mock_make_request.return_value = {
                "id": "file-io789",
                "object": "file",
                "bytes": len(file_content),
                "created_at": 1677858242,
                "filename": "uploaded_file.txt",
                "purpose": "assistants",
                "status": "processed",
            }

            # Call the method with a file-like object
            result = await self.provider.upload_file(
                file=file_obj, purpose="assistants", filename="uploaded_file.txt"
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
        """Test uploading with an invalid file type."""

        # Create a custom class that doesn't have a read method and isn't a string/bytes
        class InvalidFileType:
            """Invalid file type for testing."""

            def __str__(self):
                return "InvalidFileObject"

        invalid_file = InvalidFileType()

        # We don't need to mock _make_request since the validation happens before that
        # The real implementation should raise the error directly
        with pytest.raises(InvalidRequestError) as exc_info:
            await self.provider.upload_file(file=invalid_file, purpose="assistants")

        # Verify error message
        assert "Invalid file type" in str(exc_info.value)


class TestOpenAIFileOperations:
    """Tests for OpenAI provider file operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = OpenAIProvider(api_key="sk-test-key")

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_list_files_basic(self, mock_make_request):
        """Test basic list_files functionality."""
        # Mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "file-123",
                    "object": "file",
                    "purpose": "assistants",
                    "filename": "test.txt",
                    "bytes": 1024,
                    "created_at": 1677858242,
                    "status": "processed",
                }
            ],
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.list_files()

        # Check response
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "file-123"

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/files"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_list_files_with_purpose(self, mock_make_request):
        """Test list_files with purpose filter."""
        # Mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "file-123",
                    "object": "file",
                    "purpose": "fine-tune",
                    "filename": "data.jsonl",
                    "bytes": 10240,
                    "created_at": 1677858242,
                    "status": "processed",
                }
            ],
        }
        mock_make_request.return_value = mock_response

        # Call the method with purpose
        result = await self.provider.list_files(purpose="fine-tune")

        # Check response
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["purpose"] == "fine-tune"

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/files"
        assert kwargs["data"]["purpose"] == "fine-tune"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_delete_file_success(self, mock_make_request):
        """Test successful file deletion."""
        # Mock response
        mock_response = {"id": "file-123", "object": "file", "deleted": True}
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.delete_file(file_id="file-123")

        # Check response
        assert result["id"] == "file-123"
        assert result["deleted"] is True

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["path"] == "/files/file-123"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_delete_file_with_additional_params(self, mock_make_request):
        """Test delete_file with additional parameters."""
        # Mock response
        mock_response = {"id": "file-456", "object": "file", "deleted": True}
        mock_make_request.return_value = mock_response

        # Call the method with additional params (should be passed through)
        result = await self.provider.delete_file(file_id="file-456", additional_param="test-value")

        # Check response
        assert result["id"] == "file-456"
        assert result["deleted"] is True

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["path"] == "/files/file-456"
        # Additional params are not used in this implementation but should be accepted

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_list_files_error_handling(self, mock_make_request):
        """Test error handling during list_files."""
        # Configure mock to raise an exception
        mock_make_request.side_effect = Exception("Invalid API key")

        # Call method and expect error
        with pytest.raises(Exception) as excinfo:
            await self.provider.list_files()

        # Verify error message
        assert "Invalid API key" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_make_request")
    async def test_delete_file_error_handling(self, mock_make_request):
        """Test error handling during delete_file."""
        # Configure mock to raise an exception
        mock_make_request.side_effect = Exception("File not found")

        # Call method and expect error
        with pytest.raises(Exception) as excinfo:
            await self.provider.delete_file(file_id="nonexistent-file")

        # Verify error message
        assert "File not found" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_execute_download_request")
    async def test_download_file_success(self, mock_execute_download):
        """Test successful file download."""
        # Mock successful file download
        expected_content = b"This is the file content from the server"
        mock_execute_download.return_value = expected_content

        # Call the method
        result = await self.provider.download_file(file_id="file-123")

        # Verify result
        assert result == expected_content

        # Verify the mock was called
        mock_execute_download.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, "_execute_download_request")
    async def test_download_file_error_handling(self, mock_execute_download):
        """Test download_file error handling."""
        # Configure mock to raise an error (simulating 404 response)
        from onellm.errors import InvalidRequestError
        
        mock_execute_download.side_effect = InvalidRequestError(
            "File not found", status_code=404, provider="openai"
        )

        # Call method and expect error
        with pytest.raises(InvalidRequestError) as exc_info:
            await self.provider.download_file(file_id="non-existent-file")

        # Verify error message
        assert "File not found" in str(exc_info.value)


# Integration test for cross-platform file upload
class TestOpenAIFilePathFix:
    """Test OpenAI provider file upload with cross-platform path fix."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key provided")
    @pytest.mark.asyncio
    async def test_file_upload_with_path_separator(self):
        """Test OpenAI file upload with os.path.basename fix."""
        import tempfile
        from onellm import OpenAI

        client = OpenAI()

        # Create a temporary file with a path that includes directory separators
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for file upload")
            temp_path = f.name

        try:
            # Test file upload using async API
            file_obj = await client.files.acreate(file=temp_path, purpose="assistants")

            assert file_obj.id
            assert file_obj.filename == os.path.basename(temp_path)

            # Clean up - delete the uploaded file
            await client.files.adelete(file_obj.id)

        finally:
            # Clean up temp file
            os.unlink(temp_path)

import pytest
import json
from unittest import mock
import io
from pathlib import Path

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.models import FileObject
from muxi_llm.errors import InvalidRequestError


@pytest.mark.asyncio
class TestOpenAIFileUpload:
    """Test targeting file upload functionality in the OpenAI provider (lines 591-649)."""

    def setup_method(self):
        """Set up the test environment."""
        # Use a patcher to completely mock get_provider_config
        self.config_patcher = mock.patch('muxi_llm.config.get_provider_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {"api_key": "test-api-key"}

        # Create provider instance with mocked config
        self.provider = OpenAIProvider()

        # Override the api_key directly to ensure consistency in tests
        self.provider.api_key = "test-api-key"

    def teardown_method(self):
        """Clean up patchers."""
        self.config_patcher.stop()

    async def test_upload_file_with_bytes(self):
        """Test uploading a file using bytes (lines 591-625)."""
        # Create test file content
        file_content = b"This is a test file content."

        # Create mock response
        mock_response = {
            "id": "file-abc123",
            "object": "file",
            "bytes": len(file_content),
            "created_at": 1677858242,
            "filename": "test.txt",
            "purpose": "assistants",
            "status": "processed"
        }

        # Mock the _make_request method
        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with bytes
            result = await self.provider.upload_file(
                file=file_content,
                purpose="assistants",
                filename="test.txt"  # Additional parameter
            )

            # Verify the request was made correctly
            called_args = mock_request.call_args[1]
            assert called_args["method"] == "POST"
            assert called_args["path"] == "/files"
            assert "files" in called_args
            assert called_args["files"]["file"]["data"] == file_content
            assert called_args["files"]["file"]["filename"] == "test.txt"
            assert called_args["data"]["purpose"] == "assistants"

            # Verify result is a FileObject
            assert isinstance(result, FileObject)
            assert result.id == "file-abc123"
            assert result.bytes == len(file_content)
            assert result.filename == "test.txt"
            assert result.purpose == "assistants"
            assert result.status == "processed"

    async def test_upload_file_with_path(self):
        """Test uploading a file using a file path (lines 595-625)."""
        # Mock Path.read_bytes
        mock_file_content = b"This is test content from a file path."

        # Mock open function to return a file-like object with the content
        mock_file = mock.mock_open(read_data=mock_file_content)

        # Mock response from API
        mock_response = {
            "id": "file-xyz456",
            "object": "file",
            "bytes": len(mock_file_content),
            "created_at": 1677858242,
            "filename": "test_file.txt",
            "purpose": "assistants",
            "status": "processed"
        }

        # Mock the _make_request method and open function
        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request, mock.patch(
            'builtins.open', mock_file
        ):
            # Call the method with a file path string
            result = await self.provider.upload_file(
                file="path/to/test_file.txt",
                purpose="assistants"
            )

            # Verify the request was made correctly
            called_args = mock_request.call_args[1]
            assert called_args["method"] == "POST"
            assert called_args["path"] == "/files"
            assert "files" in called_args
            # Verify the file data was passed correctly
            assert called_args["data"]["purpose"] == "assistants"

            # Verify result
            assert isinstance(result, FileObject)
            assert result.id == "file-xyz456"
            assert result.filename == "test_file.txt"

    async def test_upload_file_with_file_object(self):
        """Test uploading a file using a file-like object (lines 595-625)."""
        # Create a file-like object
        file_content = b"This is test content from a file-like object."
        file_obj = io.BytesIO(file_content)

        # Mock response from API
        mock_response = {
            "id": "file-io789",
            "object": "file",
            "bytes": len(file_content),
            "created_at": 1677858242,
            "filename": "uploaded_file.txt",
            "purpose": "assistants",
            "status": "processed"
        }

        # Mock the _make_request method
        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with a file-like object
            result = await self.provider.upload_file(
                file=file_obj,
                purpose="assistants",
                filename="uploaded_file.txt"  # Provide a filename
            )

            # Verify the request was made correctly
            called_args = mock_request.call_args[1]
            assert called_args["method"] == "POST"
            assert called_args["path"] == "/files"
            assert "files" in called_args
            # The file data should be read from the file object
            assert called_args["data"]["purpose"] == "assistants"

            # Verify result
            assert isinstance(result, FileObject)
            assert result.id == "file-io789"
            assert result.filename == "uploaded_file.txt"
            assert result.purpose == "assistants"

    async def test_upload_file_with_invalid_file(self):
        """Test uploading with an invalid file type (line 647-649)."""
        # Try with an invalid file type (a custom class that doesn't have read method)
        class InvalidFileType:
            pass

        invalid_file = InvalidFileType()

        # Expect an InvalidRequestError
        with pytest.raises(InvalidRequestError) as exc_info:
            await self.provider.upload_file(
                file=invalid_file,
                purpose="assistants"
            )

        # Verify error message
        assert "Invalid file type" in str(exc_info.value)

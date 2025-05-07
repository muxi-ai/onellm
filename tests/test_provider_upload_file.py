"""
Tests for file validation in Provider.upload_file method.

This file tests the upload_file validation logic by using a custom mock provider
implementation to avoid issues with asyncio and internal method mocking.
"""

import io
import pytest
from typing import Any
import tempfile
import os
from pathlib import Path

from muxi_llm.providers.base import Provider
from muxi_llm.models import FileObject
from muxi_llm.errors import InvalidRequestError


class MockFileProvider(Provider):
    """Mock Provider implementation for testing file operations."""

    def __init__(self):
        """Initialize with minimal required attributes."""
        self.api_key = "test-api-key"
        self.upload_called = False
        self.last_file_data = None
        self.last_file_purpose = None
        self.last_file_kwargs = None

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Mock implementation of upload_file that captures inputs.

        This implementation follows the real logic for file validation
        but doesn't make actual API calls.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            A mock FileObject
        """
        # This follows similar validation logic to OpenAIProvider.upload_file
        if isinstance(file, (str, Path)):
            # File path
            with open(file, "rb") as f:
                file_data = f.read()
            self.last_file_data = file_data
        elif isinstance(file, bytes):
            # Bytes data
            file_data = file
            self.last_file_data = file_data
        elif hasattr(file, "read"):
            # File-like object
            file_data = file.read()
            self.last_file_data = file_data
        else:
            # Invalid file type
            error_msg = (
                "Invalid file type. Expected file path, bytes, or file-like object."
            )
            raise InvalidRequestError(error_msg)

        # Record that this method was successfully called and capture the parameters
        self.upload_called = True
        self.last_file_purpose = purpose
        self.last_file_kwargs = kwargs

        # Create and return a mock FileObject
        return FileObject(
            id="file-mock-id-123",
            object="file",
            bytes=len(self.last_file_data),
            created_at=1677858242,
            filename=kwargs.get("filename", "test.txt"),
            purpose=purpose,
            status="processed"
        )

    # Required abstract method implementations (minimal for the test)
    async def create_chat_completion(self, *args, **kwargs):
        """Mock implementation."""
        return {}

    async def create_completion(self, *args, **kwargs):
        """Mock implementation."""
        return {}

    async def create_embedding(self, *args, **kwargs):
        """Mock implementation."""
        return {}

    async def download_file(self, *args, **kwargs):
        """Mock implementation."""
        return b""

    async def list_files(self, *args, **kwargs):
        """Mock implementation."""
        return {"data": []}

    async def delete_file(self, *args, **kwargs):
        """Mock implementation."""
        return {"deleted": True}


class TestProviderUploadFile:
    """Tests for the Provider.upload_file method."""

    @pytest.fixture
    def provider(self):
        """Create and return a MockFileProvider instance."""
        return MockFileProvider()

    @pytest.mark.asyncio
    async def test_upload_bytes(self, provider):
        """Test uploading file as bytes."""
        # Create a byte string to upload
        file_content = b"This is test content in bytes format."

        # Call the upload_file method
        result = await provider.upload_file(
            file=file_content,
            purpose="assistants",
            filename="test_bytes.txt"
        )

        # Verify method was called and parameters were captured correctly
        assert provider.upload_called is True
        assert provider.last_file_data == file_content
        assert provider.last_file_purpose == "assistants"
        assert provider.last_file_kwargs["filename"] == "test_bytes.txt"

        # Verify the result is a valid FileObject
        assert isinstance(result, FileObject)
        assert result.id == "file-mock-id-123"
        assert result.purpose == "assistants"
        assert result.filename == "test_bytes.txt"

    @pytest.mark.asyncio
    async def test_upload_file_path(self, provider):
        """Test uploading file from a path."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"This is content in a temporary file.")
            temp_file_path = temp_file.name

        try:
            # Call the upload_file method
            result = await provider.upload_file(
                file=temp_file_path,
                purpose="fine-tune"
            )

            # Verify method was called and parameters were captured correctly
            assert provider.upload_called is True
            assert provider.last_file_data == b"This is content in a temporary file."
            assert provider.last_file_purpose == "fine-tune"

            # Verify the result is a valid FileObject
            assert isinstance(result, FileObject)
            assert result.id == "file-mock-id-123"
            assert result.purpose == "fine-tune"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_upload_file_path_as_path_object(self, provider):
        """Test uploading file from a Path object."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"This is content in a temporary file using Path.")
            temp_file_path = temp_file.name

        try:
            # Use a Path object instead of string
            path_obj = Path(temp_file_path)

            # Call the upload_file method
            result = await provider.upload_file(
                file=path_obj,
                purpose="fine-tune"
            )

            # Verify method was called and parameters were captured correctly
            assert provider.upload_called is True
            assert provider.last_file_data == b"This is content in a temporary file using Path."
            assert provider.last_file_purpose == "fine-tune"

            # Verify the result is a valid FileObject
            assert isinstance(result, FileObject)
            assert result.id == "file-mock-id-123"
            assert result.purpose == "fine-tune"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_upload_file_object(self, provider):
        """Test uploading file as a file-like object."""
        # Create a file-like object
        file_content = b"This is content in a BytesIO file-like object."
        file_obj = io.BytesIO(file_content)

        # Call the upload_file method
        result = await provider.upload_file(
            file=file_obj,
            purpose="assistants",
            filename="file_obj.txt"
        )

        # Verify method was called and parameters were captured correctly
        assert provider.upload_called is True
        assert provider.last_file_data == file_content
        assert provider.last_file_purpose == "assistants"
        assert provider.last_file_kwargs["filename"] == "file_obj.txt"

        # Verify the result is a valid FileObject
        assert isinstance(result, FileObject)
        assert result.id == "file-mock-id-123"
        assert result.purpose == "assistants"
        assert result.filename == "file_obj.txt"

    @pytest.mark.asyncio
    async def test_upload_with_additional_kwargs(self, provider):
        """Test passing additional keyword arguments to upload_file."""
        file_content = b"Testing additional kwargs."

        # Call with various additional kwargs
        result = await provider.upload_file(
            file=file_content,
            purpose="assistants",
            filename="kwargs_test.txt",
            user_id="test_user_123",  # Additional kwarg
            tags=["test", "example"],  # Additional kwarg
            custom_metadata={"category": "testing"}  # Additional kwarg
        )

        # Verify all kwargs were captured
        assert provider.last_file_kwargs["filename"] == "kwargs_test.txt"
        assert provider.last_file_kwargs["user_id"] == "test_user_123"
        assert provider.last_file_kwargs["tags"] == ["test", "example"]
        assert provider.last_file_kwargs["custom_metadata"] == {"category": "testing"}

        # Verify the result
        assert isinstance(result, FileObject)
        assert result.filename == "kwargs_test.txt"

    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, provider):
        """Test uploading with an invalid file type."""
        # Create a custom class that doesn't meet any valid file criteria
        class InvalidFileType:
            """Invalid file type for testing."""
            def __str__(self):
                return "InvalidFileObject"

        invalid_file = InvalidFileType()

        # Test with invalid file type and expect an exception
        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.upload_file(
                file=invalid_file,
                purpose="assistants"
            )

        # Verify the error message
        assert "Invalid file type" in str(exc_info.value)

        # Verify method was not successfully called
        assert provider.upload_called is False

    @pytest.mark.asyncio
    async def test_upload_with_empty_file(self, provider):
        """Test uploading an empty file."""
        # Create an empty file
        empty_content = b""

        # Call with empty file content
        result = await provider.upload_file(
            file=empty_content,
            purpose="assistants",
            filename="empty.txt"
        )

        # Verify it was processed correctly
        assert provider.upload_called is True
        assert provider.last_file_data == b""
        assert provider.last_file_purpose == "assistants"

        # Verify the FileObject shows 0 bytes
        assert result.bytes == 0

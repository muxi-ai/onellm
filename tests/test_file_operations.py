"""
Tests for the File module.

These tests verify that file operations work correctly with provider interactions.
"""

import tempfile
import asyncio
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

from muxi_llm.files import File
from muxi_llm.models import FileObject


class MockProvider:
    """Mock provider for testing file operations."""

    async def upload_file(self, file, purpose="assistants", **kwargs):
        """Mock implementation of upload_file."""
        # Simulate provider returning a file object
        return FileObject(
            id="file-123456",
            object="file",
            purpose=purpose,
            filename="test.txt" if isinstance(file, (bytes, BytesIO)) else Path(file).name,
            bytes=1024,
            created_at=1234567890,
            status="processed"
        )

    async def download_file(self, file_id, **kwargs):
        """Mock implementation of download_file."""
        # Return fake file content
        return b"Test file content"


class TestFile:
    """Tests for the File class."""

    def test_upload_with_file_path(self):
        """Test uploading a file using a file path."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("asyncio.run") as mock_run, \
             tempfile.NamedTemporaryFile() as temp_file:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Set up asyncio.run to pass through to the coroutine
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            # Write some data to the temp file
            temp_file.write(b"Test content")
            temp_file.flush()

            # Call the method
            result = File.upload(temp_file.name, purpose="fine-tune", provider="openai")

            # Verify the result
            assert isinstance(result, FileObject)
            assert result.id == "file-123456"
            assert result.purpose == "fine-tune"
            assert result.filename == Path(temp_file.name).name

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

    def test_upload_with_bytes(self):
        """Test uploading a file using bytes."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("asyncio.run") as mock_run:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Set up asyncio.run to pass through to the coroutine
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            # Call the method with bytes
            result = File.upload(b"Test content", provider="openai")

            # Verify the result
            assert isinstance(result, FileObject)
            assert result.id == "file-123456"

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

    def test_upload_with_file_object(self):
        """Test uploading a file using a file-like object."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("asyncio.run") as mock_run:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Set up asyncio.run to pass through to the coroutine
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            # Create a BytesIO object
            file_obj = BytesIO(b"Test content")

            # Call the method with file object
            result = File.upload(file_obj, provider="openai")

            # Verify the result
            assert isinstance(result, FileObject)
            assert result.id == "file-123456"

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_aupload(self):
        """Test uploading a file asynchronously."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             tempfile.NamedTemporaryFile() as temp_file:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Write some data to the temp file
            temp_file.write(b"Test content")
            temp_file.flush()

            # Call the method
            result = await File.aupload(temp_file.name, purpose="assistants", provider="anthropic")

            # Verify the result
            assert isinstance(result, FileObject)
            assert result.id == "file-123456"
            assert result.purpose == "assistants"

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("anthropic")

    def test_download_without_destination(self):
        """Test downloading a file without specifying a destination."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("asyncio.run") as mock_run:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Set up asyncio.run to pass through to the coroutine
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            # Call the method
            result = File.download("file-123456", provider="openai")

            # Verify the result is the bytes content
            assert result == b"Test file content"

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

    def test_download_with_destination(self):
        """Test downloading a file with a destination path."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("asyncio.run") as mock_run, \
             mock.patch("builtins.open", mock.mock_open()) as mock_file, \
             tempfile.TemporaryDirectory() as temp_dir:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Set up asyncio.run to pass through to the coroutine
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            # Create a destination path
            dest_path = Path(temp_dir) / "downloaded_file.txt"

            # Call the method
            result = File.download("file-123456", destination=dest_path, provider="openai")

            # Verify the result is the file path
            assert result == str(dest_path)

            # Verify the file was written correctly
            mock_file.assert_called_once_with(dest_path, "wb")
            mock_file().write.assert_called_once_with(b"Test file content")

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_adownload_without_destination(self):
        """Test downloading a file asynchronously without a destination."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Call the method
            result = await File.adownload("file-123456", provider="anthropic")

            # Verify the result is the bytes content
            assert result == b"Test file content"

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("anthropic")

    @pytest.mark.asyncio
    async def test_adownload_with_destination(self):
        """Test downloading a file asynchronously with a destination path."""
        with mock.patch("muxi_llm.files.get_provider") as mock_get_provider, \
             mock.patch("builtins.open", mock.mock_open()) as mock_file, \
             tempfile.TemporaryDirectory() as temp_dir:

            # Set up mocks
            mock_provider = MockProvider()
            mock_get_provider.return_value = mock_provider

            # Create a destination path
            dest_path = Path(temp_dir) / "downloaded_file.txt"

            # Call the method
            result = await File.adownload("file-123456", destination=dest_path, provider="anthropic")

            # Verify the result is the file path
            assert result == str(dest_path)

            # Verify the file was written correctly
            mock_file.assert_called_once_with(dest_path, "wb")
            mock_file().write.assert_called_once_with(b"Test file content")

            # Verify the provider was called correctly
            mock_get_provider.assert_called_once_with("anthropic")

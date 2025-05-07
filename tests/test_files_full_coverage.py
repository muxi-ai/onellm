#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for files.py to achieve 100% test coverage.

This module contains tests for the file handling functionality in muxi-llm
with a focus on achieving full coverage of all lines.
"""

import pytest
from unittest import mock
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from muxi_llm.files import File
from muxi_llm.models import FileObject
from muxi_llm.errors import InvalidRequestError


# Create a comprehensive mock provider for file operations
class MockFileProvider:
    """Mock provider for file operations tests."""

    def __init__(self):
        """Initialize with mock methods."""
        # Mock synchronous file operations
        self.upload_file = AsyncMock()
        self.download_file = AsyncMock(return_value=b"test file content")
        self.list_files = AsyncMock(return_value={"data": [
            {"id": "file-123", "filename": "test.txt"},
            {"id": "file-456", "filename": "data.json"}
        ]})
        self.delete_file = AsyncMock(return_value={"id": "file-123", "deleted": True})

        # Set up upload_file to return a FileObject
        self.upload_file.return_value = FileObject(
            id="file-abc123",
            object="file",
            bytes=1024,
            created_at=1677858242,
            filename="test.txt",
            purpose="assistants",
            status="processed"
        )


class TestFilesFullCoverage:
    """Tests aimed at achieving 100% test coverage of files.py."""

    def test_upload_with_path_string(self):
        """Test uploading a file with a path string."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock), \
             mock.patch("builtins.open", mock.mock_open(read_data=b"test content")):
            # Call the method with a string path
            result = File.upload(
                file="path/to/test.txt",
                purpose="assistants",
                provider="openai"
            )

            # Verify the provider method was called with correct args
            provider_mock.upload_file.assert_called_once()

            # Verify the result matches the mocked response
            assert result.id == "file-abc123"
            assert result.purpose == "assistants"

    def test_upload_with_path_object(self):
        """Test uploading a file with a Path object."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock), \
             mock.patch("builtins.open", mock.mock_open(read_data=b"test content")):
            # Call the method with a Path object
            file_path = Path("path/to/test.txt")
            result = File.upload(
                file=file_path,
                purpose="assistants",
                provider="openai"
            )

            # Verify the provider method was called
            provider_mock.upload_file.assert_called_once()

            # Verify the result
            assert result.id == "file-abc123"
            assert result.purpose == "assistants"

    def test_upload_with_file_object(self):
        """Test uploading a file with a file-like object."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Create a file-like object
            file_obj = io.BytesIO(b"test content")

            # Call the method with file object
            result = File.upload(
                file=file_obj,
                purpose="fine-tune",
                provider="openai"
            )

            # Verify the provider method was called
            provider_mock.upload_file.assert_called_once()

            # Verify the result
            assert result.id == "file-abc123"
            assert result.purpose == "assistants"

    def test_upload_with_bytes(self):
        """Test uploading a file with bytes."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the method with bytes
            result = File.upload(
                file=b"test content bytes",
                purpose="assistants",
                provider="openai"
            )

            # Verify the provider method was called
            provider_mock.upload_file.assert_called_once()

            # Verify the result
            assert result.id == "file-abc123"
            assert result.purpose == "assistants"

    def test_download_without_destination(self):
        """Test downloading a file without specifying a destination."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the method
            result = File.download(
                file_id="file-123",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.download_file.assert_called_once_with(file_id="file-123")

            # Verify result is the file content
            assert result == b"test file content"

    def test_download_with_destination(self):
        """Test downloading a file to a specific destination."""
        provider_mock = MockFileProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "download.txt")

            with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
                # Call the method with destination
                result = File.download(
                    file_id="file-456",
                    destination=dest_path,
                    provider="openai"
                )

                # Verify provider method was called
                provider_mock.download_file.assert_called_once_with(file_id="file-456")

                # Verify file was written to destination
                assert os.path.exists(dest_path)
                with open(dest_path, "rb") as f:
                    content = f.read()
                    assert content == b"test file content"

    def test_download_creating_parent_directories(self):
        """Test downloading a file creates parent directories if needed."""
        provider_mock = MockFileProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "dirs", "for", "test")
            dest_path = os.path.join(nested_dir, "download.txt")

            # Ensure directory doesn't exist
            assert not os.path.exists(nested_dir)

            with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
                # Call the method with nested path
                result = File.download(
                    file_id="file-789",
                    destination=dest_path,
                    provider="openai"
                )

                # Verify parent directories were created
                assert os.path.exists(nested_dir)
                assert os.path.exists(dest_path)

    def test_list_files(self):
        """Test listing files."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the method
            result = File.list(
                purpose="assistants",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.list_files.assert_called_once_with(purpose="assistants")

            # Verify result
            assert len(result["data"]) == 2
            assert result["data"][0]["id"] == "file-123"

    def test_delete_file(self):
        """Test deleting a file."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the method
            result = File.delete(
                file_id="file-123",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.delete_file.assert_called_once_with(file_id="file-123")

            # Verify result
            assert result["id"] == "file-123"
            assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_aupload_file(self):
        """Test asynchronous file upload."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the async method
            result = await File.aupload(
                file=b"async test content",
                purpose="fine-tune",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.upload_file.assert_called_once()

            # Verify result
            assert result.id == "file-abc123"

    @pytest.mark.asyncio
    async def test_adownload_file(self):
        """Test asynchronous file download."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the async method
            result = await File.adownload(
                file_id="file-123",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.download_file.assert_called_once_with(file_id="file-123")

            # Verify result
            assert result == b"test file content"

    @pytest.mark.asyncio
    async def test_adownload_with_destination(self):
        """Test async download with destination."""
        provider_mock = MockFileProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "async_download.txt")

            with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
                # Call the async method with destination
                result = await File.adownload(
                    file_id="file-123",
                    destination=dest_path,
                    provider="openai"
                )

                # Verify file was created
                assert os.path.exists(dest_path)

                # Verify content
                with open(dest_path, "rb") as f:
                    content = f.read()
                    assert content == b"test file content"

    @pytest.mark.asyncio
    async def test_alist_files(self):
        """Test asynchronous file listing."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the async method
            result = await File.alist(
                purpose="assistants",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.list_files.assert_called_once_with(purpose="assistants")

            # Verify result
            assert len(result["data"]) == 2

    @pytest.mark.asyncio
    async def test_adelete_file(self):
        """Test asynchronous file deletion."""
        provider_mock = MockFileProvider()

        with mock.patch("muxi_llm.files.get_provider", return_value=provider_mock):
            # Call the async method
            result = await File.adelete(
                file_id="file-456",
                provider="openai"
            )

            # Verify provider method was called
            provider_mock.delete_file.assert_called_once_with(file_id="file-456")

            # Verify result
            assert result["id"] == "file-123"
            assert result["deleted"] is True

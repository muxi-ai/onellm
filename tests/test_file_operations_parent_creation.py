import asyncio
import os
import pytest
from unittest import mock
from pathlib import Path
import tempfile
import shutil

from onellm.files import File

class TestFileParentDirectoryCreation:
    """Test file operations with specific focus on parent directory creation."""

    def setup_method(self):
        """Set up temporary directory for file tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_download_with_nested_directory_creation(self):
        """Test downloading a file with creation of nested parent directories."""
        # Mock provider's download_file method
        mock_provider = mock.AsyncMock()
        mock_provider.download_file.return_value = b"test file content"

        # Mock get_provider to return our mock provider
        with mock.patch("onellm.files.get_provider", return_value=mock_provider):
            # Create a destination path with nested directories that don't exist
            nested_path = os.path.join(self.temp_dir, "level1", "level2", "level3", "test_file.txt")

            # Use the async version to test the directory creation path
            result = await File.adownload(
                file_id="file123",
                destination=nested_path,
                provider="openai"
            )

            # Verify the parent directories were created
            assert os.path.exists(os.path.dirname(nested_path))
            # Verify the file was saved correctly
            assert os.path.exists(nested_path)
            # Verify the correct path was returned
            assert result == str(Path(nested_path))
            # Verify the file content
            with open(nested_path, "rb") as f:
                assert f.read() == b"test file content"

    def test_download_sync_with_nested_directory_creation(self):
        """Test downloading a file synchronously with creation of nested parent directories."""
        # Mock provider's download_file method
        mock_provider = mock.AsyncMock()
        mock_provider.download_file.return_value = b"test file content"

        # Mock get_provider to return our mock provider
        with mock.patch("onellm.files.get_provider", return_value=mock_provider):
            # Create a destination path with nested directories that don't exist
            nested_path = os.path.join(self.temp_dir, "sync_level1", "sync_level2", "test_file.txt")

            # Use the synchronous version which calls mkdir internally
            result = File.download(
                file_id="file123",
                destination=nested_path,
                provider="openai"
            )

            # Verify the parent directories were created
            assert os.path.exists(os.path.dirname(nested_path))
            # Verify the file was saved correctly
            assert os.path.exists(nested_path)
            # Verify the correct path was returned
            assert result == str(Path(nested_path))
            # Verify the file content
            with open(nested_path, "rb") as f:
                assert f.read() == b"test file content"

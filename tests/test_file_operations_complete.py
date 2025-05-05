"""
Tests for complete coverage of file operations in muxi-llm.

These tests target the specific uncovered lines in files.py to achieve 100% test coverage.
"""

import pytest
from unittest import mock
import asyncio
from pathlib import Path
import shutil  # Used in conditional block

from muxi_llm import File


class TestFileOperationsComplete:
    """Tests targeting uncovered lines in files.py."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file."""
        file_path = tmp_path / "test_file.txt"
        with open(file_path, "w") as f:
            f.write("Test content")
        return file_path

    def test_download_with_destination_parent_creation(self, tmp_path):
        """Test that File.download creates parent directories if needed."""
        # Setup
        provider_mock = mock.MagicMock()
        provider_mock.download_file.return_value = asyncio.Future()
        provider_mock.download_file.return_value.set_result(b"Test file content")

        # Create a nested path that doesn't exist yet
        nested_dir = tmp_path / "nested" / "directories" / "for" / "test"
        dest_path = nested_dir / "downloaded_file.txt"

        # Make sure the directory doesn't already exist
        if nested_dir.exists():
            shutil.rmtree(nested_dir)

        with mock.patch('muxi_llm.files.get_provider', return_value=provider_mock):
            with mock.patch('asyncio.run', side_effect=lambda x: x.result()):
                # Test download with destination in directory that doesn't exist
                result = File.download(
                    file_id="file-123",
                    destination=str(dest_path)
                )

                # Verify parent directories were created
                assert nested_dir.exists()

                # Verify file was created
                assert Path(result).exists()

                # Verify the file contains the expected content
                with open(result, "rb") as f:
                    content = f.read()
                    assert content == b"Test file content"

    def test_download_without_destination(self):
        """Test File.download when no destination is provided."""
        # Setup
        provider_mock = mock.MagicMock()
        provider_mock.download_file.return_value = asyncio.Future()
        provider_mock.download_file.return_value.set_result(b"Test file content")

        with mock.patch('muxi_llm.files.get_provider', return_value=provider_mock):
            with mock.patch('asyncio.run', side_effect=lambda x: x.result()):
                # Test download without destination
                result = File.download(file_id="file-456")

                # Should return bytes directly
                assert result == b"Test file content"

                # Check that the download_file was called with correct args
                provider_mock.download_file.assert_called_once_with(file_id="file-456")

    @pytest.mark.asyncio
    async def test_adownload_with_destination_parent_creation(self, tmp_path):
        """Test that File.adownload creates parent directories if needed."""
        # Setup - create a coroutine that can be awaited
        async def mock_download_file(*args, **kwargs):
            return b"Test file content"

        provider_mock = mock.MagicMock()
        provider_mock.download_file = mock_download_file

        # Create a nested path that doesn't exist yet
        nested_dir = tmp_path / "async_nested" / "directories" / "for" / "test"
        dest_path = nested_dir / "downloaded_file.txt"

        # Make sure the directory doesn't already exist
        if nested_dir.exists():
            shutil.rmtree(nested_dir)

        with mock.patch('muxi_llm.files.get_provider', return_value=provider_mock):
            # Test adownload with destination in directory that doesn't exist
            result = await File.adownload(
                file_id="file-123",
                destination=str(dest_path)
            )

            # Verify parent directories were created
            assert nested_dir.exists()

            # Verify file was created
            assert Path(result).exists()

            # Verify the file contains the expected content
            with open(result, "rb") as f:
                content = f.read()
                assert content == b"Test file content"

    @pytest.mark.asyncio
    async def test_adownload_without_destination(self):
        """Test File.adownload when no destination is provided."""
        # Setup - create a coroutine that can be awaited
        async def mock_download_file(*args, **kwargs):
            return b"Test file content"

        provider_mock = mock.MagicMock()
        provider_mock.download_file = mock_download_file

        with mock.patch('muxi_llm.files.get_provider', return_value=provider_mock):
            # Test adownload without destination (should return bytes directly)
            result = await File.adownload(
                file_id="file-789"
            )

            # Should return bytes directly
            assert result == b"Test file content"

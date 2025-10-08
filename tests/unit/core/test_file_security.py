#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for file security validation."""

import io
import os
import tempfile
from pathlib import Path

import pytest

from onellm.errors import InvalidRequestError
from onellm.files import _sanitize_filename, SizeLimitedFileWrapper
from onellm.utils.file_validator import FileValidator


class TestFilenameSanitization:
    """Test filename sanitization to prevent directory traversal."""

    def test_sanitize_directory_traversal(self):
        """Test that directory traversal attempts are sanitized."""
        assert _sanitize_filename("../../etc/passwd") == "passwd"
        assert _sanitize_filename("../../../config.txt") == "config.txt"
        assert _sanitize_filename("..\\..\\windows\\system32\\evil.dll") == "evil.dll"

    def test_sanitize_subdirectories(self):
        """Test that subdirectory components are removed."""
        assert _sanitize_filename("dir/subdir/file.txt") == "file.txt"
        assert _sanitize_filename("path/to/deep/file.pdf") == "file.pdf"
        assert _sanitize_filename("folder\\subfolder\\document.doc") == "document.doc"

    def test_sanitize_null_bytes(self):
        """Test that null bytes are removed."""
        assert _sanitize_filename("file\x00.exe") == "file.exe"
        assert _sanitize_filename("doc\x00ument.pdf") == "document.pdf"
        assert _sanitize_filename("\x00hidden.txt") == "hidden.txt"

    def test_sanitize_special_filenames(self):
        """Test that special filenames are handled."""
        assert _sanitize_filename(".") == "file.bin"
        assert _sanitize_filename("..") == "file.bin"
        assert _sanitize_filename("") == "file.bin"
        assert _sanitize_filename(None) == "file.bin"

    def test_sanitize_legitimate_filenames(self):
        """Test that legitimate filenames pass through."""
        assert _sanitize_filename("document.pdf") == "document.pdf"
        assert _sanitize_filename("my..file.txt") == "my..file.txt"
        assert _sanitize_filename("data..2024.csv") == "data..2024.csv"
        assert _sanitize_filename("file-name_123.doc") == "file-name_123.doc"

    def test_sanitize_custom_default(self):
        """Test custom default filename."""
        assert _sanitize_filename("", default="custom.bin") == "custom.bin"
        assert _sanitize_filename(".", default="default.dat") == "default.dat"


class TestFileValidator:
    """Test FileValidator security checks."""

    def test_directory_traversal_prevention(self):
        """Test that directory traversal is blocked."""
        with pytest.raises(InvalidRequestError, match="Directory traversal detected"):
            FileValidator.validate_file_path("../../../etc/passwd")

        with pytest.raises(InvalidRequestError, match="Directory traversal detected"):
            FileValidator.validate_file_path("..\\..\\windows\\system32\\config\\sam")

    def test_legitimate_filenames_allowed(self):
        """Test that legitimate filenames with '..' as substring are allowed."""
        # Create temporary file with '..' in name
        with tempfile.NamedTemporaryFile(suffix="my..file.txt", delete=False) as f:
            temp_file = f.name
            f.write(b"test content")

        try:
            # Should not raise - '..' is substring, not path component
            validated = FileValidator.validate_file_path(temp_file)
            assert validated.exists()
        finally:
            os.unlink(temp_file)

    def test_file_size_validation(self):
        """Test that file size limits are enforced."""
        # Create a file larger than limit
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b"x" * 1000)  # 1KB file

        try:
            # Should succeed with 10KB limit
            validated = FileValidator.validate_file_path(temp_file, max_size=10 * 1024)
            assert validated.exists()

            # Should fail with 500 byte limit
            with pytest.raises(InvalidRequestError, match="File too large"):
                FileValidator.validate_file_path(temp_file, max_size=500)
        finally:
            os.unlink(temp_file)

    def test_extension_validation(self):
        """Test that file extension validation works."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_file = f.name
            f.write(b"test")

        try:
            # Should succeed with .pdf allowed
            validated = FileValidator.validate_file_path(
                temp_file, allowed_extensions={".pdf", ".txt"}
            )
            assert validated.exists()

            # Should fail with only .txt allowed
            with pytest.raises(InvalidRequestError, match="File type not allowed"):
                FileValidator.validate_file_path(temp_file, allowed_extensions={".txt"})
        finally:
            os.unlink(temp_file)

    def test_bytes_size_validation(self):
        """Test bytes size validation."""
        data = b"x" * 1000

        # Should succeed
        FileValidator.validate_bytes_size(data, max_size=2000)

        # Should fail
        with pytest.raises(InvalidRequestError, match="too large"):
            FileValidator.validate_bytes_size(data, max_size=500)

    def test_filename_validation(self):
        """Test filename extension and MIME validation."""
        # Valid filename
        FileValidator.validate_filename("document.pdf", allowed_extensions={".pdf", ".txt"})

        # Invalid extension
        with pytest.raises(InvalidRequestError, match="File type not allowed"):
            FileValidator.validate_filename("file.exe", allowed_extensions={".pdf", ".txt"})

        # No extension
        with pytest.raises(InvalidRequestError, match="File has no extension"):
            FileValidator.validate_filename("file", allowed_extensions={".pdf"})


class TestSizeLimitedFileWrapper:
    """Test SizeLimitedFileWrapper for enforcing size limits on streams."""

    def test_within_limit(self):
        """Test reading within size limit."""
        data = b"Hello World"
        file_obj = io.BytesIO(data)
        wrapper = SizeLimitedFileWrapper(file_obj, max_size=100, name="test")

        # Read all data - should succeed
        result = wrapper.read()
        assert result == data

    def test_exceeds_limit(self):
        """Test that exceeding size limit raises error."""
        data = b"x" * 1000
        file_obj = io.BytesIO(data)
        wrapper = SizeLimitedFileWrapper(file_obj, max_size=500, name="test")

        # Read should raise when limit exceeded
        with pytest.raises(InvalidRequestError, match="test too large"):
            wrapper.read()

    def test_chunked_reading(self):
        """Test reading in chunks."""
        data = b"x" * 1000
        file_obj = io.BytesIO(data)
        wrapper = SizeLimitedFileWrapper(file_obj, max_size=1500, name="test")

        # Read in chunks - should succeed
        chunk1 = wrapper.read(500)
        assert len(chunk1) == 500

        chunk2 = wrapper.read(500)
        assert len(chunk2) == 500

        # Third chunk should fail (total would be 1500, but we read slightly over)
        with pytest.raises(InvalidRequestError, match="test too large"):
            wrapper.read(500)

    def test_attribute_delegation(self):
        """Test that other attributes are delegated to wrapped file."""
        file_obj = io.BytesIO(b"test")
        file_obj.custom_attr = "custom_value"
        wrapper = SizeLimitedFileWrapper(file_obj, max_size=100, name="test")

        # Should delegate attribute access
        assert wrapper.custom_attr == "custom_value"


class TestFileSanitizationIntegration:
    """Test integration of filename sanitization with File.upload."""

    @pytest.mark.asyncio
    async def test_upload_sanitizes_filename(self, monkeypatch):
        """Test that File.upload sanitizes filenames before sending to provider."""
        received_filename = None

        class MockProvider:
            async def upload_file(self, file, purpose, filename=None, **kwargs):
                nonlocal received_filename
                received_filename = filename
                # Return a minimal FileObject-like response
                return type('obj', (object,), {
                    'id': 'file-123',
                    'purpose': purpose,
                    'filename': filename
                })()

        def mock_get_provider(provider):
            return MockProvider()

        monkeypatch.setattr("onellm.files.get_provider", mock_get_provider)

        from onellm import File

        # Test directory traversal is sanitized
        data = b"test content"
        await File.aupload(data, purpose="test", filename="../../evil.exe")
        assert received_filename == "evil.exe"

        # Test subdirectories are sanitized
        await File.aupload(data, purpose="test", filename="path/to/file.txt")
        assert received_filename == "file.txt"

        # Test null bytes are sanitized
        await File.aupload(data, purpose="test", filename="file\x00.exe")
        assert received_filename == "file.exe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

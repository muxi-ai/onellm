#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File handling utilities for OneLLM.

This module provides a unified interface for working with files across different LLM providers,
including uploading, retrieving, and listing files.
"""

import os
from pathlib import Path
from typing import BinaryIO

from .errors import InvalidRequestError
from .models import FileObject
from .providers.base import get_provider
from .utils.async_helpers import run_async
from .utils.file_validator import FileValidator, DEFAULT_MAX_FILE_SIZE


def _sanitize_filename(filename: str | None, default: str = "file.bin") -> str:
    """
    Sanitize a filename by removing directory components and null bytes.
    
    This prevents directory traversal attacks and ensures the filename is safe.
    
    Args:
        filename: Raw filename that may contain directory components or null bytes
        default: Default filename to use if sanitized result is empty
        
    Returns:
        Sanitized filename safe for use
        
    Examples:
        >>> _sanitize_filename("../../etc/passwd")
        'passwd'
        >>> _sanitize_filename("dir/subdir/file.txt")
        'file.txt'
        >>> _sanitize_filename("file\\x00.exe")
        'file.exe'
        >>> _sanitize_filename("")
        'file.bin'
    """
    if not filename:
        return default

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Strip directory components (works for / and \ separators)
    filename = os.path.basename(filename)

    # If basename is empty after sanitization, use default
    if not filename or filename in ('.', '..'):
        return default

    return filename


class SizeLimitedFileWrapper:
    """
    Transparent wrapper for file-like objects that enforces size limits during reading.
    
    This is used for non-seekable streams where we can't check the size upfront.
    The wrapper tracks bytes read and raises an error if the limit is exceeded.
    
    All read methods (read, readline, readlines, readinto, readinto1, iteration)
    are explicitly wrapped to ensure size accounting cannot be bypassed.
    
    Note: This only protects against reading too much data. A race condition
    exists where the file could be modified between size check and read for
    seekable files. For untrusted sources, validate both before and after.
    """

    def __init__(self, file_obj: BinaryIO, max_size: int, name: str = "file"):
        self._file = file_obj
        self._max_size = max_size
        self._bytes_read = 0
        self._name = name

    def _check_size(self, data_len: int) -> None:
        """Check if adding data_len bytes would exceed the limit."""
        self._bytes_read += data_len
        if self._bytes_read > self._max_size:
            max_mb = self._max_size / (1024 * 1024)
            actual_mb = self._bytes_read / (1024 * 1024)
            raise InvalidRequestError(
                f"{self._name} too large: exceeds {max_mb:.2f}MB (read {actual_mb:.2f}MB so far)"
            )

    def read(self, size: int = -1) -> bytes:
        """Read from the underlying file while tracking size."""
        data = self._file.read(size)
        self._check_size(len(data))
        return data

    def readline(self, size: int = -1) -> bytes:
        """Read one line from the underlying file while tracking size."""
        data = self._file.readline(size)
        self._check_size(len(data))
        return data

    def readlines(self, hint: int = -1) -> list[bytes]:
        """Read all lines from the underlying file while tracking size."""
        lines = self._file.readlines(hint)
        total_size = sum(len(line) for line in lines)
        self._check_size(total_size)
        return lines

    def readinto(self, b: bytearray) -> int:
        """Read into a buffer while tracking size."""
        if hasattr(self._file, 'readinto'):
            n = self._file.readinto(b)
            if n is not None:
                self._check_size(n)
            return n
        else:
            # Fallback: use read() if readinto not available
            data = self._file.read(len(b))
            n = len(data)
            b[:n] = data
            self._check_size(n)
            return n

    def readinto1(self, b: bytearray) -> int:
        """Read into a buffer (single read call) while tracking size."""
        if hasattr(self._file, 'readinto1'):
            n = self._file.readinto1(b)
            if n is not None:
                self._check_size(n)
            return n
        else:
            # Fallback to readinto
            return self.readinto(b)

    def __iter__(self):
        """Iterate over lines while tracking size."""
        return self

    def __next__(self) -> bytes:
        """Get next line while tracking size."""
        line = next(self._file)
        self._check_size(len(line))
        return line

    def __getattr__(self, name: str):
        """
        Delegate all other attributes and methods to the wrapped file object.
        
        This makes the wrapper transparent to code expecting standard file-like
        objects, including provider implementations.
        
        Note: All common read methods are explicitly wrapped above to ensure
        size limits cannot be bypassed.
        """
        return getattr(self._file, name)


class File:
    """Interface for file operations across different providers."""

    @classmethod
    def upload(
        cls,
        file: str | Path | BinaryIO | bytes,
        purpose: str = "assistants",
        provider: str = "openai",  # Required but defaults for compatibility
        max_size: int | None = None,
        allowed_extensions: set[str] | None = None,
        validate_mime: bool = True,
        base_directory: Path | None = None,
        **kwargs
    ) -> FileObject:
        """
        Upload a file to the provider's API.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file (defaults to "assistants")
            provider: Provider to use (e.g., "openai")
            max_size: Maximum file size in bytes (default: 100MB)
            allowed_extensions: Set of allowed file extensions (default: common types)
            validate_mime: Whether to validate MIME type (default: True)
            base_directory: Optional base directory to restrict file access (for path validation)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: If file validation fails (size, type, security, etc.)

        Example:
            >>> file_obj = File.upload("path/to/file.pdf", purpose="fine-tune", provider="openai")
            >>> print(f"Uploaded file ID: {file_obj.id}")
            
            >>> # With custom size limit
            >>> file_obj = File.upload("large.mp3", purpose="transcription", max_size=200*1024*1024)
        """
        # Enforce default max_size if not specified (100MB)
        if max_size is None:
            max_size = DEFAULT_MAX_FILE_SIZE

        # Get provider instance
        provider_instance = get_provider(provider)

        # Validate file based on type, but preserve original object for provider
        # This allows providers to optimize (e.g., streaming) while ensuring security
        if isinstance(file, (str, Path)):
            # Validate file path for security (checks size, extension, MIME, traversal)
            validated_path = FileValidator.validate_file_path(
                str(file),
                max_size=max_size,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime,
                base_directory=base_directory,
            )

            # Pass the validated path string to provider (allows provider to stream)
            file_to_upload = str(validated_path)
            
            # Sanitize and validate the filename even for paths
            # This prevents smuggling disallowed extensions or path segments
            raw_filename = kwargs.get("filename") or validated_path.name
            sanitized_name = _sanitize_filename(raw_filename, "file.bin")
            
            # Validate the sanitized filename
            FileValidator.validate_filename(
                sanitized_name,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )
            
            # Always set the sanitized and validated filename
            kwargs["filename"] = sanitized_name

        elif isinstance(file, bytes):
            # Validate bytes size
            FileValidator.validate_bytes_size(
                file,
                max_size=max_size,
                name="file data"
            )

            # Get and sanitize filename from kwargs - required if allowed_extensions is set
            raw_filename = kwargs.get("filename")
            if raw_filename is None:
                if allowed_extensions:
                    raise InvalidRequestError(
                        "filename parameter is required when uploading bytes with allowed_extensions set"
                    )
                filename = "file.bin"
            else:
                # Sanitize filename to remove directory components and null bytes
                filename = _sanitize_filename(raw_filename, "file.bin")

            # Validate sanitized filename extension and MIME type
            FileValidator.validate_filename(
                filename,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )

            # Always update kwargs with sanitized filename to ensure provider receives safe value
            kwargs["filename"] = filename

            # Pass bytes as-is to provider
            file_to_upload = file

        elif hasattr(file, "read"):
            # File-like object - try to validate without reading entire file

            # Get filename - prefer user-provided kwargs, then file.name, then default
            # This ensures we validate the ACTUAL filename that will be sent to provider
            raw_filename = kwargs.get("filename") or getattr(file, "name", None)
            if raw_filename is None:
                if allowed_extensions:
                    raise InvalidRequestError(
                        "filename parameter is required when uploading file-like object without .name attribute with allowed_extensions set"
                    )
                filename = "file.bin"
            else:
                # Sanitize filename to remove directory components and null bytes
                filename = _sanitize_filename(raw_filename, "file.bin")

            # Validate sanitized filename extension and MIME type
            FileValidator.validate_filename(
                filename,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )

            # Always set the sanitized and validated filename in kwargs to ensure provider receives safe value
            kwargs["filename"] = filename

            # For size validation, try to get size without reading entire file
            # max_size is always set (defaults to 100MB if not specified)
            file_size = None

            # Try to get size from seekable file-like object for early validation
            # Note: Size can still change after this check (TOCTOU), but we wrap below
            if hasattr(file, 'seek') and hasattr(file, 'tell'):
                try:
                    current_pos = file.tell()
                    original_error = None
                    position_error = None
                    try:
                        file.seek(0, 2)  # Seek to end
                        file_size = file.tell()
                    except Exception as e:
                        # Capture any exception from size check
                        original_error = e
                    finally:
                        # Always try to restore position, even if an error occurs
                        try:
                            file.seek(current_pos)
                        except OSError as e:
                            # Store restoration error separately
                            position_error = e
                    
                    # Prioritize exceptions: original error takes precedence
                    if original_error is not None:
                        # Re-raise the original exception from size check
                        raise original_error
                    elif position_error is not None:
                        # Only raise position error if no original error
                        raise InvalidRequestError(
                            f"Failed to restore file position after size check: {str(position_error)}"
                        )
                except OSError:
                    # File is not seekable (e.g., stdin, pipe, socket)
                    # This is not an error, just means we can't pre-check size
                    pass

            # If we got the size, validate it (early check, wrapper enforces below)
            if file_size is not None:
                max_mb = max_size / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                if file_size > max_size:
                    raise InvalidRequestError(
                        f"File too large: {actual_mb:.2f}MB exceeds {max_mb:.2f}MB"
                    )

            # ALWAYS wrap with size-limiting wrapper to prevent TOCTOU attacks
            # For seekable files: prevents size changes after validation (TOCTOU protection)
            # For non-seekable streams: enforces size limits during reading
            # The wrapper is transparent and delegates all operations
            file = SizeLimitedFileWrapper(file, max_size, filename)

            # Pass the file-like object (wrapped if non-seekable) to provider
            # Filename already set in kwargs during validation
            file_to_upload = file

        else:
            raise InvalidRequestError(
                f"file must be a path (str/Path), bytes, or file-like object, "
                f"got {type(file).__name__}"
            )

        # Call the provider's upload_file method synchronously
        # Provider receives original file object (or bytes if already read) for optimal handling
        return run_async(
            provider_instance.upload_file(
                file=file_to_upload,
                purpose=purpose,
                **kwargs
            )
        )

    @classmethod
    async def aupload(
        cls,
        file: str | Path | BinaryIO | bytes,
        purpose: str = "assistants",
        provider: str = "openai",  # Required but defaults for compatibility
        max_size: int | None = None,
        allowed_extensions: set[str] | None = None,
        validate_mime: bool = True,
        base_directory: Path | None = None,
        **kwargs
    ) -> FileObject:
        """
        Upload a file to the provider's API asynchronously.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file (defaults to "assistants")
            provider: Provider to use (e.g., "openai")
            max_size: Maximum file size in bytes (default: 100MB)
            allowed_extensions: Set of allowed file extensions (default: common types)
            validate_mime: Whether to validate MIME type (default: True)
            base_directory: Optional base directory to restrict file access (for path validation)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: If file validation fails (size, type, security, etc.)

        Example:
            >>> file_obj = await File.aupload("file.pdf", purpose="fine-tune", provider="openai")
            >>> print(f"Uploaded file ID: {file_obj.id}")
        """
        # Enforce default max_size if not specified (100MB)
        if max_size is None:
            max_size = DEFAULT_MAX_FILE_SIZE

        # Get provider instance
        provider_instance = get_provider(provider)

        # Validate file based on type, but preserve original object for provider
        # This allows providers to optimize (e.g., streaming) while ensuring security
        if isinstance(file, (str, Path)):
            # Validate file path for security (checks size, extension, MIME, traversal)
            validated_path = FileValidator.validate_file_path(
                str(file),
                max_size=max_size,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime,
                base_directory=base_directory,
            )

            # Pass the validated path string to provider (allows provider to stream)
            file_to_upload = str(validated_path)
            
            # Sanitize and validate the filename even for paths
            # This prevents smuggling disallowed extensions or path segments
            raw_filename = kwargs.get("filename") or validated_path.name
            sanitized_name = _sanitize_filename(raw_filename, "file.bin")
            
            # Validate the sanitized filename
            FileValidator.validate_filename(
                sanitized_name,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )
            
            # Always set the sanitized and validated filename
            kwargs["filename"] = sanitized_name

        elif isinstance(file, bytes):
            # Validate bytes size
            FileValidator.validate_bytes_size(
                file,
                max_size=max_size,
                name="file data"
            )

            # Get and sanitize filename from kwargs - required if allowed_extensions is set
            raw_filename = kwargs.get("filename")
            if raw_filename is None:
                if allowed_extensions:
                    raise InvalidRequestError(
                        "filename parameter is required when uploading bytes with allowed_extensions set"
                    )
                filename = "file.bin"
            else:
                # Sanitize filename to remove directory components and null bytes
                filename = _sanitize_filename(raw_filename, "file.bin")

            # Validate sanitized filename extension and MIME type
            FileValidator.validate_filename(
                filename,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )

            # Always update kwargs with sanitized filename to ensure provider receives safe value
            kwargs["filename"] = filename

            # Pass bytes as-is to provider
            file_to_upload = file

        elif hasattr(file, "read"):
            # File-like object - try to validate without reading entire file

            # Get filename - prefer user-provided kwargs, then file.name, then default
            # This ensures we validate the ACTUAL filename that will be sent to provider
            raw_filename = kwargs.get("filename") or getattr(file, "name", None)
            if raw_filename is None:
                if allowed_extensions:
                    raise InvalidRequestError(
                        "filename parameter is required when uploading file-like object without .name attribute with allowed_extensions set"
                    )
                filename = "file.bin"
            else:
                # Sanitize filename to remove directory components and null bytes
                filename = _sanitize_filename(raw_filename, "file.bin")

            # Validate sanitized filename extension and MIME type
            FileValidator.validate_filename(
                filename,
                allowed_extensions=allowed_extensions,
                validate_mime=validate_mime
            )

            # Always set the sanitized and validated filename in kwargs to ensure provider receives safe value
            kwargs["filename"] = filename

            # For size validation, try to get size without reading entire file
            # max_size is always set (defaults to 100MB if not specified)
            file_size = None

            # Try to get size from seekable file-like object for early validation
            # Note: Size can still change after this check (TOCTOU), but we wrap below
            if hasattr(file, 'seek') and hasattr(file, 'tell'):
                try:
                    current_pos = file.tell()
                    original_error = None
                    position_error = None
                    try:
                        file.seek(0, 2)  # Seek to end
                        file_size = file.tell()
                    except Exception as e:
                        # Capture any exception from size check
                        original_error = e
                    finally:
                        # Always try to restore position, even if an error occurs
                        try:
                            file.seek(current_pos)
                        except OSError as e:
                            # Store restoration error separately
                            position_error = e
                    
                    # Prioritize exceptions: original error takes precedence
                    if original_error is not None:
                        # Re-raise the original exception from size check
                        raise original_error
                    elif position_error is not None:
                        # Only raise position error if no original error
                        raise InvalidRequestError(
                            f"Failed to restore file position after size check: {str(position_error)}"
                        )
                except OSError:
                    # File is not seekable (e.g., stdin, pipe, socket)
                    # This is not an error, just means we can't pre-check size
                    pass

            # If we got the size, validate it (early check, wrapper enforces below)
            if file_size is not None:
                max_mb = max_size / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                if file_size > max_size:
                    raise InvalidRequestError(
                        f"File too large: {actual_mb:.2f}MB exceeds {max_mb:.2f}MB"
                    )

            # ALWAYS wrap with size-limiting wrapper to prevent TOCTOU attacks
            # For seekable files: prevents size changes after validation (TOCTOU protection)
            # For non-seekable streams: enforces size limits during reading
            # The wrapper is transparent and delegates all operations
            file = SizeLimitedFileWrapper(file, max_size, filename)

            # Pass the file-like object (wrapped if non-seekable) to provider
            # Filename already set in kwargs during validation
            file_to_upload = file

        else:
            raise InvalidRequestError(
                f"file must be a path (str/Path), bytes, or file-like object, "
                f"got {type(file).__name__}"
            )

        # Call the provider's upload_file method
        # Provider receives original file object (or bytes if already read) for optimal handling
        return await provider_instance.upload_file(
            file=file_to_upload,
            purpose=purpose,
            **kwargs
        )

    @classmethod
    def download(
        cls,
        file_id: str,
        destination: str | Path | None = None,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> bytes | str:
        """
        Download a file from the provider's API.

        Args:
            file_id: ID of the file to download
            destination: Optional path where to save the file
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Bytes content of the file if destination is None, otherwise path to the saved file

        Example:
            >>> file_bytes = File.download("file-abc123", provider="openai")
            >>> # or
            >>> file_path = File.download("file-abc123", destination="file.txt", provider="openai")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's download_file method synchronously
        # We need to use our safe async runner to call the async method from a synchronous context
        file_bytes = run_async(
            provider_instance.download_file(file_id=file_id, **kwargs)
        )

        # Save to destination if provided
        # If a destination path is given, save the file and return the path
        # Otherwise, return the raw bytes
        if destination:
            dest_path = Path(destination)
            # Create parent directories if they don't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Write the file contents
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
            return str(dest_path)

        return file_bytes

    @classmethod
    async def adownload(
        cls,
        file_id: str,
        destination: str | Path | None = None,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> bytes | str:
        """
        Download a file from the provider's API asynchronously.

        Args:
            file_id: ID of the file to download
            destination: Optional path where to save the file
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Bytes content of the file if destination is None, otherwise path to the saved file

        Example:
            >>> file_bytes = await File.adownload("file", provider="openai")
            >>> # or
            >>> file_path = await File.adownload("file", destination="file.txt", provider="openai")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's download_file method
        # This is the async version, so we directly await the result
        file_bytes = await provider_instance.download_file(file_id=file_id, **kwargs)

        # Save to destination if provided
        # If a destination path is given, save the file and return the path
        # Otherwise, return the raw bytes
        if destination:
            dest_path = Path(destination)
            # Create parent directories if they don't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Write the file contents
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
            return str(dest_path)

        return file_bytes

    @classmethod
    def list(
        cls,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> dict:
        """
        List files from the provider's API.

        Args:
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider like 'purpose'

        Returns:
            Dictionary containing the list of files

        Example:
            >>> files = File.list(provider="openai", purpose="fine-tune")
            >>> for file in files["data"]:
            >>>     print(f"File: {file['filename']}, ID: {file['id']}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's list_files method synchronously
        # We need to use our safe async runner to call the async method from a synchronous context
        return run_async(provider_instance.list_files(**kwargs))

    @classmethod
    async def alist(
        cls,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> dict:
        """
        List files from the provider's API asynchronously.

        Args:
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider like 'purpose'

        Returns:
            Dictionary containing the list of files

        Example:
            >>> files = await File.alist(provider="openai", purpose="fine-tune")
            >>> for file in files["data"]:
            >>>     print(f"File: {file['filename']}, ID: {file['id']}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's list_files method
        # This is the async version, so we directly await the result
        return await provider_instance.list_files(**kwargs)

    @classmethod
    def delete(
        cls,
        file_id: str,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> dict:
        """
        Delete a file from the provider's API.

        Args:
            file_id: ID of the file to delete
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dictionary containing the deletion status

        Example:
            >>> result = File.delete("file-abc123", provider="openai")
            >>> print(f"Deleted: {result['deleted']}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's delete_file method synchronously
        # We need to use our safe async runner to call the async method from a synchronous context
        return run_async(
            provider_instance.delete_file(file_id=file_id, **kwargs)
        )

    @classmethod
    async def adelete(
        cls,
        file_id: str,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> dict:
        """
        Delete a file from the provider's API asynchronously.

        Args:
            file_id: ID of the file to delete
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dictionary containing the deletion status

        Example:
            >>> result = await File.adelete("file-abc123", provider="openai")
            >>> print(f"Deleted: {result['deleted']}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's delete_file method
        # This is the async version, so we directly await the result
        return await provider_instance.delete_file(file_id=file_id, **kwargs)

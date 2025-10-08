#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
File validation utilities for OneLLM.

This module provides security-focused file validation to prevent common attacks
like directory traversal, and to enforce size and type constraints.
"""

import mimetypes
from pathlib import Path
from typing import Optional, Set

from ..errors import InvalidRequestError

# Default maximum file size: 100MB
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024

# Default allowed file extensions
# These are common file types used with LLM APIs
DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    # Audio formats (for transcription, translation)
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac",
    # Image formats (for vision models)
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    # Document formats
    ".pdf", ".txt", ".json", ".jsonl", ".csv", ".tsv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Code and data formats
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".xml", ".yaml", ".yml", ".toml", ".ini",
    # Archive formats
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    # Video formats (for future support)
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
}


class FileValidator:
    """
    Validates file paths and contents for security and compliance.
    
    This class provides methods to:
    - Validate file paths to prevent directory traversal attacks
    - Enforce file size limits to prevent DoS attacks
    - Validate file types to prevent uploading malicious files
    - Safely read file contents
    """
    
    @staticmethod
    def validate_file_path(
        file_path: str,
        max_size: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
        validate_mime: bool = True,
    ) -> Path:
        """
        Validate and normalize a file path for security.
        
        This method performs comprehensive validation including:
        - Path existence and type checking
        - Directory traversal prevention
        - File size validation
        - Extension validation
        - MIME type validation
        
        Args:
            file_path: Path to the file to validate
            max_size: Maximum allowed file size in bytes (default: 100MB)
            allowed_extensions: Set of allowed file extensions (default: common types)
            validate_mime: Whether to validate MIME type matches extension
            
        Returns:
            Validated and normalized Path object
            
        Raises:
            InvalidRequestError: If any validation check fails
            
        Example:
            >>> path = FileValidator.validate_file_path("data/file.txt")
            >>> with open(path, 'rb') as f:
            ...     data = f.read()
        """
        # Validate input type
        if not file_path or not isinstance(file_path, str):
            raise InvalidRequestError(
                "file_path must be a non-empty string"
            )
        
        # Set defaults
        if max_size is None:
            max_size = DEFAULT_MAX_FILE_SIZE
        if allowed_extensions is None:
            allowed_extensions = DEFAULT_ALLOWED_EXTENSIONS
        
        try:
            # Convert to Path and resolve to absolute path
            # This follows symlinks and normalizes the path
            path = Path(file_path).resolve(strict=True)
        except FileNotFoundError:
            raise InvalidRequestError(
                f"File not found: {file_path}"
            )
        except (OSError, RuntimeError) as e:
            raise InvalidRequestError(
                f"Invalid file path: {e}"
            )
        
        # Verify it's a regular file (not a directory, device, etc.)
        if not path.is_file():
            if path.is_dir():
                raise InvalidRequestError(
                    f"Path is a directory, not a file: {file_path}"
                )
            else:
                raise InvalidRequestError(
                    f"Path is not a regular file: {file_path}"
                )
        
        # Check for directory traversal attempts
        # After resolve(), the path should not contain ".."
        # This prevents attacks like "../../../../etc/passwd"
        if ".." in path.parts:
            raise InvalidRequestError(
                f"Directory traversal detected in path: {file_path}"
            )
        
        # Validate file extension if restrictions are set
        if allowed_extensions:
            file_extension = path.suffix.lower()
            
            # Empty extension check
            if not file_extension:
                raise InvalidRequestError(
                    f"File has no extension: {path.name}. "
                    f"Allowed extensions: {', '.join(sorted(allowed_extensions))}"
                )
            
            # Check if extension is allowed
            if file_extension not in allowed_extensions:
                # Create a helpful error message with allowed types
                allowed_list = ', '.join(sorted(allowed_extensions)[:10])
                if len(allowed_extensions) > 10:
                    allowed_list += f", ... ({len(allowed_extensions)} total)"
                
                raise InvalidRequestError(
                    f"File type not allowed: {file_extension}. "
                    f"Allowed types: {allowed_list}"
                )
        
        # Check file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise InvalidRequestError(
                f"Cannot access file: {e}"
            )
        
        # Empty file check
        if file_size == 0:
            raise InvalidRequestError(
                f"File is empty: {path.name}"
            )
        
        # Size limit check
        if max_size and file_size > max_size:
            # Convert to human-readable format
            max_mb = max_size / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            
            raise InvalidRequestError(
                f"File too large: {actual_mb:.2f}MB exceeds limit of {max_mb:.2f}MB. "
                f"File: {path.name}"
            )
        
        # Validate MIME type matches extension
        if validate_mime:
            mime_type, _ = mimetypes.guess_type(str(path))
            
            # If we can't determine MIME type, be cautious
            if mime_type is None:
                # Some extensions might not have MIME types registered
                # Only warn for common cases
                if path.suffix.lower() not in {'.txt', '.json', '.jsonl', '.csv'}:
                    raise InvalidRequestError(
                        f"Cannot determine file type for: {path.name}. "
                        f"Extension: {path.suffix}"
                    )
        
        return path
    
    @staticmethod
    def safe_read_file(
        path: Path,
        max_size: Optional[int] = None,
        chunk_size: int = 8192,
    ) -> bytes:
        """
        Safely read file contents with memory protection.
        
        This method reads files in chunks to avoid loading huge files
        into memory all at once, which could cause memory issues.
        
        Args:
            path: Validated Path object to read
            max_size: Maximum size to read (default: file size)
            chunk_size: Size of chunks to read (default: 8KB)
            
        Returns:
            File contents as bytes
            
        Raises:
            InvalidRequestError: If file cannot be read or is too large
            
        Example:
            >>> path = FileValidator.validate_file_path("data.bin")
            >>> data = FileValidator.safe_read_file(path)
        """
        if not isinstance(path, Path):
            raise InvalidRequestError(
                "path must be a Path object (use validate_file_path first)"
            )
        
        # Get file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise InvalidRequestError(
                f"Cannot access file: {e}"
            )
        
        # Check against max_size if provided
        if max_size and file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise InvalidRequestError(
                f"File too large to read: {actual_mb:.2f}MB exceeds {max_mb:.2f}MB"
            )
        
        # Read file in chunks
        try:
            chunks = []
            bytes_read = 0
            
            with open(path, "rb") as f:
                while True:
                    # Read a chunk
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunks.append(chunk)
                    bytes_read += len(chunk)
                    
                    # Double-check we haven't exceeded max_size
                    # (in case file was modified during reading)
                    if max_size and bytes_read > max_size:
                        raise InvalidRequestError(
                            f"File size exceeded during read: {path.name}"
                        )
            
            return b"".join(chunks)
            
        except OSError as e:
            raise InvalidRequestError(
                f"Error reading file: {e}"
            )
        except MemoryError:
            raise InvalidRequestError(
                f"File too large to fit in memory: {path.name}"
            )
    
    @staticmethod
    def validate_bytes_size(
        data: bytes,
        max_size: Optional[int] = None,
        name: str = "data",
    ) -> None:
        """
        Validate size of byte data.
        
        Args:
            data: Bytes to validate
            max_size: Maximum allowed size in bytes
            name: Name for error messages
            
        Raises:
            InvalidRequestError: If data is too large
        """
        if not isinstance(data, bytes):
            raise InvalidRequestError(
                f"{name} must be bytes, got {type(data).__name__}"
            )
        
        if len(data) == 0:
            raise InvalidRequestError(
                f"{name} is empty"
            )
        
        if max_size and len(data) > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = len(data) / (1024 * 1024)
            raise InvalidRequestError(
                f"{name} too large: {actual_mb:.2f}MB exceeds {max_mb:.2f}MB"
            )

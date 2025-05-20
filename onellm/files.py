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
File handling utilities for OneLLM.

This module provides a unified interface for working with files across different LLM providers,
including uploading, retrieving, and listing files.
"""

from pathlib import Path
from typing import BinaryIO, Union

from .providers.base import get_provider
from .models import FileObject


class File:
    """Interface for file operations across different providers."""

    @classmethod
    def upload(
        cls,
        file: Union[str, Path, BinaryIO, bytes],
        purpose: str = "assistants",
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> FileObject:
        """
        Upload a file to the provider's API.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file (defaults to "assistants")
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            FileObject representing the uploaded file

        Example:
            >>> file_obj = File.upload("path/to/file.pdf", purpose="fine-tune", provider="openai")
            >>> print(f"Uploaded file ID: {file_obj.id}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's upload_file method synchronously
        # We need to use asyncio.run to call the async method from a synchronous context
        import asyncio

        return asyncio.run(
            provider_instance.upload_file(file=file, purpose=purpose, **kwargs)
        )

    @classmethod
    async def aupload(
        cls,
        file: Union[str, Path, BinaryIO, bytes],
        purpose: str = "assistants",
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> FileObject:
        """
        Upload a file to the provider's API asynchronously.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file (defaults to "assistants")
            provider: Provider to use (e.g., "openai")
            **kwargs: Additional parameters to pass to the provider

        Returns:
            FileObject representing the uploaded file

        Example:
            >>> file_obj = await File.aupload("file.pdf", purpose="fine-tune", provider="openai")
            >>> print(f"Uploaded file ID: {file_obj.id}")
        """
        # Get provider instance
        provider_instance = get_provider(provider)

        # Call the provider's upload_file method
        # This is the async version, so we directly await the result
        return await provider_instance.upload_file(file=file, purpose=purpose, **kwargs)

    @classmethod
    def download(
        cls,
        file_id: str,
        destination: Union[str, Path, None] = None,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> Union[bytes, str]:
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
        # We need to use asyncio.run to call the async method from a synchronous context
        import asyncio

        file_bytes = asyncio.run(
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
        destination: Union[str, Path, None] = None,
        provider: str = "openai",  # Required but defaults for compatibility
        **kwargs
    ) -> Union[bytes, str]:
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
        # We need to use asyncio.run to call the async method from a synchronous context
        import asyncio

        return asyncio.run(provider_instance.list_files(**kwargs))

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
        # We need to use asyncio.run to call the async method from a synchronous context
        import asyncio

        return asyncio.run(
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

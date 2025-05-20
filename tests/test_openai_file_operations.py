import asyncio
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for OpenAI provider file operations.

These tests specifically focus on the list_files and delete_file methods
to improve test coverage.
"""

import pytest
from unittest.mock import patch

from onellm.providers.openai import OpenAIProvider


class MockResponse:
    """Mock for aiohttp response."""

    def __init__(self, status: int, data: dict):
        self.status = status
        self._data = data

    async def json(self):
        return self._data


class TestOpenAIFileOperations:
    """Tests for OpenAI provider file operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = OpenAIProvider(api_key="sk-test-key")

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_list_files_basic(self, mock_make_request):
        """Test basic list_files functionality."""
        # Mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "file-123",
                    "object": "file",
                    "purpose": "assistants",
                    "filename": "test.txt",
                    "bytes": 1024,
                    "created_at": 1677858242,
                    "status": "processed"
                }
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.list_files()

        # Check response
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "file-123"

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/files"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_list_files_with_purpose(self, mock_make_request):
        """Test list_files with purpose filter."""
        # Mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "file-123",
                    "object": "file",
                    "purpose": "fine-tune",
                    "filename": "data.jsonl",
                    "bytes": 10240,
                    "created_at": 1677858242,
                    "status": "processed"
                }
            ]
        }
        mock_make_request.return_value = mock_response

        # Call the method with purpose
        result = await self.provider.list_files(purpose="fine-tune")

        # Check response
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["purpose"] == "fine-tune"

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/files"
        assert kwargs["data"]["purpose"] == "fine-tune"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_delete_file_success(self, mock_make_request):
        """Test successful file deletion."""
        # Mock response
        mock_response = {
            "id": "file-123",
            "object": "file",
            "deleted": True
        }
        mock_make_request.return_value = mock_response

        # Call the method
        result = await self.provider.delete_file(file_id="file-123")

        # Check response
        assert result["id"] == "file-123"
        assert result["deleted"] is True

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["path"] == "/files/file-123"

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_delete_file_with_additional_params(self, mock_make_request):
        """Test delete_file with additional parameters."""
        # Mock response
        mock_response = {
            "id": "file-456",
            "object": "file",
            "deleted": True
        }
        mock_make_request.return_value = mock_response

        # Call the method with additional params (should be passed through)
        result = await self.provider.delete_file(
            file_id="file-456",
            additional_param="test-value"
        )

        # Check response
        assert result["id"] == "file-456"
        assert result["deleted"] is True

        # Verify request parameters
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["path"] == "/files/file-456"
        # Additional params are not used in this implementation but should be accepted

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_list_files_error_handling(self, mock_make_request):
        """Test error handling during list_files."""
        # Configure mock to raise an exception
        mock_make_request.side_effect = Exception("Invalid API key")

        # Call method and expect error
        with pytest.raises(Exception) as excinfo:
            await self.provider.list_files()

        # Verify error message
        assert "Invalid API key" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch.object(OpenAIProvider, '_make_request')
    async def test_delete_file_error_handling(self, mock_make_request):
        """Test error handling during delete_file."""
        # Configure mock to raise an exception
        mock_make_request.side_effect = Exception("File not found")

        # Call method and expect error
        with pytest.raises(Exception) as excinfo:
            await self.provider.delete_file(file_id="nonexistent-file")

        # Verify error message
        assert "File not found" in str(excinfo.value)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper module to patch provider objects during testing to avoid real API calls.
This is used by the test runner script to ensure tests use mocks instead of real providers.
"""

import sys
import os
import asyncio
from unittest import mock
from pathlib import Path

# Add the parent directory to the path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from muxi_llm.models import FileObject
from muxi_llm.providers import get_provider, openai

# Create mock file objects for testing
MOCK_FILE_OBJECTS = {
    "file-123": FileObject(
        id="file-123",
        object="file",
        purpose="assistants",
        filename="test.txt",
        bytes=1024,
        created_at=1234567890,
        status="processed"
    ),
    "file-456": FileObject(
        id="file-456",
        object="file",
        purpose="assistants",
        filename="data.csv",
        bytes=2048,
        created_at=1234567890,
        status="processed"
    )
}

async def mock_upload_file(*args, **kwargs):
    """Mock file upload API call."""
    return MOCK_FILE_OBJECTS["file-123"]

async def mock_download_file(*args, **kwargs):
    """Mock file download API call."""
    return b"test file content"

async def mock_list_files(*args, **kwargs):
    """Mock list files API call."""
    return {"data": [{"id": f.id, "object": f.object, "filename": f.filename}
                    for f in MOCK_FILE_OBJECTS.values()]}

async def mock_delete_file(*args, **kwargs):
    """Mock delete file API call."""
    return {"id": "file-123", "deleted": True}

# Apply the patches to avoid real API calls
def apply_patches():
    """Apply all necessary patches to avoid real API calls."""
    patches = []

    # Patch OpenAI provider methods
    patches.append(mock.patch.object(openai.OpenAIProvider, 'upload_file', mock_upload_file))
    patches.append(mock.patch.object(openai.OpenAIProvider, 'download_file', mock_download_file))
    patches.append(mock.patch.object(openai.OpenAIProvider, 'list_files', mock_list_files))
    patches.append(mock.patch.object(openai.OpenAIProvider, 'delete_file', mock_delete_file))

    # Activate all patches
    for p in patches:
        p.start()

    return patches

# This will be imported and used in test runner
patches = apply_patches()

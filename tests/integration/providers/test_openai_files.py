#!/usr/bin/env python3
"""
Integration tests for OpenAI file operations.
"""

import os
import pytest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key provided")
class TestOpenAIFilePathFix:
    """Test OpenAI provider file upload with cross-platform path fix."""

    @pytest.mark.asyncio
    async def test_file_upload_with_path_separator(self):
        """Test OpenAI file upload with os.path.basename fix."""
        import tempfile
        from onellm import OpenAI

        client = OpenAI()

        # Create a temporary file with a path that includes directory separators
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for file upload")
            temp_path = f.name

        try:
            # Test file upload using async API
            file_obj = await client.files.acreate(file=temp_path, purpose="assistants")

            assert file_obj.id
            assert file_obj.filename == os.path.basename(temp_path)

            # Clean up - delete the uploaded file
            await client.files.adelete(file_obj.id)

        finally:
            # Clean up temp file
            os.unlink(temp_path)

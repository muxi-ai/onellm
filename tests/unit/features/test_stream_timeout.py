"""
Tests for timeout handling in streaming utilities.

This file focuses on testing the timeout functionality in a simpler, more direct way.
"""

import asyncio
import pytest
from unittest import mock
from typing import AsyncGenerator, Any, List

from onellm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)

# Helper to create a simple async generator for testing
async def async_generator(items: List[Any]) -> AsyncGenerator[Any, None]:
    """Helper to create a simple async generator for testing."""
    for item in items:
        yield item

class TestTimeoutHandling:
    """Test class focused on timeout handling across all streaming functions."""

    @pytest.mark.asyncio
    async def test_stream_generator_timeout(self):
        """Test that timeout works in stream_generator."""
        # Mock asyncio.wait_for to raise TimeoutError after first item
        original_wait_for = asyncio.wait_for
        call_count = 0

        async def mock_wait_for(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # Let the first item through, timeout on the second
                raise asyncio.TimeoutError("Test timeout")
            return await original_wait_for(*args, **kwargs)

        # Create a simple source generator with multiple items
        source = async_generator(["item1", "item2", "item3"])

        # Patch wait_for to timeout after first item
        with mock.patch("asyncio.wait_for", side_effect=mock_wait_for):
            # Collect results until timeout
            results = []
            with pytest.raises(StreamingError) as excinfo:
                async for item in stream_generator(source, timeout=0.1):
                    results.append(item)

            # Check that we got one item before timeout
            assert len(results) == 1
            assert results[0] == "item1"
            # Check the timeout error
            assert "Streaming response timed out after 0.1 seconds" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_json_stream_generator_timeout(self):
        """Test that timeout is passed through json_stream_generator."""
        # Mock wait_for to raise TimeoutError
        async def mock_wait_for(*args, **kwargs):
            raise asyncio.TimeoutError("Test timeout")

        # Create a source with JSON strings
        source = async_generator(['{"key": "value"}'])

        # First, check that the timeout parameter is passed to wait_for
        with mock.patch("asyncio.wait_for", side_effect=mock_wait_for):
            # This should raise a StreamingError
            with pytest.raises(StreamingError) as excinfo:
                async for _ in json_stream_generator(source, timeout=0.5):
                    pass

            # Check the timeout error
            assert "Streaming response timed out" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_line_stream_generator_timeout(self):
        """Test that timeout is passed through line_stream_generator."""
        # Mock wait_for to raise TimeoutError
        async def mock_wait_for(*args, **kwargs):
            raise asyncio.TimeoutError("Test timeout")

        # Create a source with lines
        source = async_generator(["line1", "line2"])

        # Check that the timeout parameter is passed to wait_for
        with mock.patch("asyncio.wait_for", side_effect=mock_wait_for):
            # This should raise a StreamingError
            with pytest.raises(StreamingError) as excinfo:
                async for _ in line_stream_generator(source, timeout=0.2):
                    pass

            # Check the timeout error
            assert "Streaming response timed out" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_empty_and_filtered_lines(self):
        """Test handling of empty lines and prefix filtering."""
        # Create source with empty lines and lines with/without prefix
        source = async_generator([
            "",                # Empty line should be filtered out
            "data: line1",     # With prefix - should be included
            "  ",              # Whitespace-only - should be filtered out
            "other: line2",    # Without prefix - should be filtered out
            "data: line3"      # With prefix - should be included
        ])

        # Process with prefix filtering
        results = []
        async for line in line_stream_generator(source, prefix="data: "):
            results.append(line)

        # Check that only the lines with the prefix are included, with prefix removed
        assert results == ["line1", "line3"]

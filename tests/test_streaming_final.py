"""
Final tests for achieving 100% coverage of streaming.py.

These tests focus on the specific missing lines (85-94, 124-139) to achieve full coverage.
"""

import pytest
from unittest import mock
from typing import AsyncGenerator, Any, List

from muxi_llm.utils.streaming import (
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


# Helper to create a simple async generator for testing
async def async_generator(items: List[Any]) -> AsyncGenerator[Any, None]:
    """Helper to create a simple async generator for testing."""
    for item in items:
        yield item


# Test Class for specific code paths in json_stream_generator
class TestJsonStreamGeneratorFinal:
    """Test class focused on achieving full coverage for json_stream_generator."""

    @pytest.mark.asyncio
    async def test_json_stream_generator_timeout(self):
        """Test handling of timeout in json_stream_generator (lines 85-94)."""
        # Test scenario: stream_generator raises a timeout error with the provided timeout
        timeout_error = StreamingError("Streaming response timed out after 10.0 seconds")

        # Create a mock generator function that just raises a timeout error
        async def mock_stream_gen(*args, **kwargs):
            # Check that the timeout parameter is passed through
            assert kwargs.get('timeout') == 10.0
            raise timeout_error

        # Patch stream_generator to raise our timeout error
        with mock.patch('muxi_llm.utils.streaming.stream_generator', side_effect=mock_stream_gen):
            # json_stream_generator should propagate this timeout error
            with pytest.raises(StreamingError) as excinfo:
                source = async_generator(['{"key": "value"}'])
                async for _ in json_stream_generator(source, timeout=10.0):
                    pass

            # Verify it's the same error (should be passed through unchanged)
            assert excinfo.value is timeout_error


# Test Class for specific code paths in line_stream_generator
class TestLineStreamGeneratorFinal:
    """Test class focused on achieving full coverage for line_stream_generator."""

    @pytest.mark.asyncio
    async def test_line_stream_generator_timeout(self):
        """Test handling of timeout in line_stream_generator (lines 124-139)."""
        # Test scenario: stream_generator raises a timeout error with the provided timeout
        timeout_error = StreamingError("Streaming response timed out after 5.0 seconds")

        # Create a mock generator function that just raises a timeout error
        async def mock_stream_gen(*args, **kwargs):
            # Check that the timeout parameter is passed through
            assert kwargs.get('timeout') == 5.0
            raise timeout_error

        # Patch stream_generator to raise our timeout error
        with mock.patch('muxi_llm.utils.streaming.stream_generator', side_effect=mock_stream_gen):
            # line_stream_generator should propagate this timeout error
            with pytest.raises(StreamingError) as excinfo:
                source = async_generator(["line1", "line2"])
                async for _ in line_stream_generator(source, timeout=5.0):
                    pass

            # Verify it's the same error (should be passed through unchanged)
            assert excinfo.value is timeout_error

    @pytest.mark.asyncio
    async def test_empty_prefix(self):
        """Test the line_stream_generator with an empty prefix."""
        # Create a source with one line
        source = async_generator(["line1"])

        # Mock stream_generator to apply the transform function and yield the result
        async def mock_stream_gen(gen, transform_func=None, **kwargs):
            async for item in gen:
                if transform_func:
                    result = transform_func(item)
                    if result is not None:
                        yield result

        # Patch stream_generator to use our mock
        with mock.patch('muxi_llm.utils.streaming.stream_generator', side_effect=mock_stream_gen):
            # Process with an empty prefix (should return the full string)
            results = []
            async for line in line_stream_generator(source, prefix=""):
                results.append(line)

            # The line should be returned unchanged since prefix is empty
            assert results == ["line1"]

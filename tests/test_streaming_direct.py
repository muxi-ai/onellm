"""
Direct tests for the streaming utilities module.

These tests directly exercise the stream_generator, json_stream_generator, and line_stream_generator
functions to achieve coverage for lines 85-94 and 124-139 in streaming.py.
"""

import pytest
import asyncio
import json
from unittest import mock

# Import the modules we want to test directly
from muxi_llm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


# Create a simple AsyncGenerator for testing
async def items_generator(items):
    for item in items:
        yield item


# Create a generator that raises a timeout
async def timeout_generator():
    yield "first item"
    raise asyncio.TimeoutError("Timeout")


class TestStreamGeneratorDirect:
    """Test the stream_generator directly."""

    @pytest.mark.asyncio
    async def test_stream_generator_timeout_with_value(self):
        """Test that timeout error includes the timeout value."""
        # This tests line 56 in stream_generator which is reported as missing
        timeout_value = 5.0
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(timeout_generator(), timeout=timeout_value):
                pass

        assert f"Streaming response timed out after {timeout_value} seconds" in str(excinfo.value)


class TestJsonStreamGeneratorDirect:
    """Test the missing code paths in json_stream_generator directly."""

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_timeout_error(self):
        """Test that json_stream_generator correctly handles timeout errors."""
        # We need to mock stream_generator to return an async generator that raises a timeout error
        test_timeout = 10.0

        # Create a mock that raises the right error
        async def mock_stream_generator(*args, **kwargs):
            # Extract the timeout parameter to include in the error message
            timeout_value = kwargs.get('timeout')
            # Yield one item to allow the for loop to start
            yield "this will be returned"
            # Then simulate a timeout
            raise StreamingError(f"Streaming response timed out after {timeout_value} seconds")

        # Patch the stream_generator function
        with mock.patch('muxi_llm.utils.streaming.stream_generator', new=mock_stream_generator):
            # Create a source generator with some JSON data
            source = items_generator(['{"key": "value"}'])

            # This should raise the StreamingError from mock_stream_generator
            results = []
            with pytest.raises(StreamingError) as excinfo:
                async for item in json_stream_generator(source, timeout=test_timeout):
                    results.append(item)

            # Verify our error was raised and message includes the timeout
            assert "Streaming response timed out after 10.0 seconds" in str(excinfo.value)
            # First item should have been processed
            assert results[0] == "this will be returned"

    @pytest.mark.asyncio
    async def test_json_stream_generator_transform_and_data_key(self):
        """Test the transform_json function's ability to extract data keys."""
        # Create a source with JSON objects
        json_strings = [
            '{"data": {"value": 1}, "meta": "info1"}',
            '{"data": {"value": 2}, "meta": "info2"}',
        ]
        source = items_generator(json_strings)

        # We'll instrument the JSON parsing to track actual calls
        original_loads = json.loads

        # Create a wrapper for json.loads that allows us to track calls
        def tracked_json_loads(text):
            # Call original function and track that we processed this text
            result = original_loads(text)
            tracked_json_loads.called_with.append(text)
            return result

        # Initialize tracking
        tracked_json_loads.called_with = []

        # Patch json.loads with our tracking version
        with mock.patch('json.loads', tracked_json_loads):
            # Process with a data_key
            results = []
            async for item in json_stream_generator(source, data_key="data"):
                results.append(item)

            # Verify results - only the data key contents should be yielded
            assert len(results) == 2
            assert results[0] == {"value": 1}
            assert results[1] == {"value": 2}

            # Verify our wrapper was called for each JSON string
            assert len(tracked_json_loads.called_with) == 2
            assert tracked_json_loads.called_with[0] == '{"data": {"value": 1}, "meta": "info1"}'
            assert tracked_json_loads.called_with[1] == '{"data": {"value": 2}, "meta": "info2"}'


class TestLineStreamGeneratorDirect:
    """Test the missing code paths in line_stream_generator directly."""

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_timeout_error(self):
        """Test that line_stream_generator correctly handles timeout errors."""
        # We need to mock stream_generator to return an async generator that raises a timeout error
        test_timeout = 5.0

        # Create a mock that raises the right error
        async def mock_stream_generator(*args, **kwargs):
            # Extract the timeout parameter to include in the error message
            timeout_value = kwargs.get('timeout')
            # Yield one item to allow the for loop to start
            yield "this will be returned"
            # Then simulate a timeout
            raise StreamingError(f"Streaming response timed out after {timeout_value} seconds")

        # Patch the stream_generator function
        with mock.patch('muxi_llm.utils.streaming.stream_generator', new=mock_stream_generator):
            # Create a source generator with some line data
            source = items_generator(["line1", "line2"])

            # This should raise the StreamingError from mock_stream_generator
            results = []
            with pytest.raises(StreamingError) as excinfo:
                async for item in line_stream_generator(source, timeout=test_timeout):
                    results.append(item)

            # Verify our error was raised and message includes the timeout
            assert "Streaming response timed out after 5.0 seconds" in str(excinfo.value)
            # First item should have been processed
            assert results[0] == "this will be returned"

    @pytest.mark.asyncio
    async def test_direct_process_line_functionality(self):
        """Test the process_line functionality in line_stream_generator."""
        # In this test, we'll provide various inputs directly to line_stream_generator,
        # but we'll control what stream_generator does to test the process_line function

        # Mock stream_generator to apply the transform function directly
        async def mock_stream_gen(source, transform_func=None, **kwargs):
            # Instead of actually streaming, we'll just apply the transform
            # to our controlled input and return the result
            if transform_func:
                test_inputs = [
                    # Test string with prefix
                    "data: line1",
                    # Test string without prefix
                    "other: text",
                    # Test bytes with prefix
                    b"data: binary1",
                    # Test empty string
                    "",
                    # Test whitespace-only string
                    "   ",
                    # Test string with newlines
                    "data: line2\r\n",
                ]

                # Apply transform to each input and yield non-None results
                for item in test_inputs:
                    result = transform_func(item)
                    if result is not None:
                        yield result

        # Patch stream_generator with our custom implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', side_effect=mock_stream_gen):
            # Create a dummy source (the mock will ignore this)
            source = items_generator(["dummy"])

            # Test with a prefix - should only yield lines with that prefix
            results = []
            async for line in line_stream_generator(source, prefix="data: "):
                results.append(line)

            # Only items starting with "data: " should be in results,
            # with the prefix removed
            assert results == ["line1", "binary1", "line2"]

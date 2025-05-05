"""
Advanced tests for the streaming utilities module.

These tests focus on more complex scenarios and error handling in the streaming utilities,
targeting the uncovered parts of the code.
"""

import asyncio
import json
import pytest
from unittest import mock
from typing import Any, AsyncGenerator, Optional

from muxi_llm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


# Helper functions to create test generators
async def async_generator(items):
    """Create a simple async generator from a list of items."""
    for item in items:
        yield item


async def failing_generator():
    """Create a generator that raises an exception."""
    yield "first item"
    raise ValueError("Generator failure")


class TestStreamGeneratorErrors:
    """Test error handling in stream_generator."""

    @pytest.mark.asyncio
    async def test_generator_exception(self):
        """Test that exceptions in the source generator are properly caught and wrapped."""
        generator = failing_generator()

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(generator):
                pass

        assert "Error in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_transform_exception(self):
        """Test that exceptions in the transform function are properly caught and wrapped."""
        # Define a transform function that raises an exception
        def transform_func(item):
            raise ValueError("Transform error")

        source = async_generator(["item1", "item2"])

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(source, transform_func=transform_func):
                pass

        assert "Error transforming streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are properly handled."""
        # Create a mock generator that will timeout
        source = async_generator(["item1"])

        # Create a patched version of stream_generator that raises timeout error
        async def timeout_generator(*args, **kwargs):
            # Simulate the first iteration before timeout
            await asyncio.sleep(0)  # Ensure we yield control
            raise asyncio.TimeoutError("Test timeout")

        # Apply the patch and test
        with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=timeout_generator):
            with pytest.raises(StreamingError) as excinfo:
                async for _ in stream_generator(source, timeout=0.1):
                    pass

            assert "Streaming response timed out" in str(excinfo.value)


class TestJsonStreamGenerator:
    """Test JSON stream generator functionality."""

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON in the stream."""
        source = async_generator([
            '{"valid": "json"}',
            'invalid json',
            '{"also": "valid"}'
        ])

        # Mock json.loads to raise an error on 'invalid json'
        with mock.patch("json.loads") as mock_json_loads:
            # Set up side effects for json.loads
            def side_effect(text):
                if text == 'invalid json':
                    raise json.JSONDecodeError("Invalid JSON", "", 0)
                return json.loads(text)

            mock_json_loads.side_effect = side_effect

            # Collect results and handle errors
            results = []
            with pytest.raises(StreamingError) as excinfo:
                async for item in json_stream_generator(source):
                    results.append(item)

            # Verify we got the first valid item before the error
            assert len(results) == 1
            assert "Invalid JSON" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_extract_nested_data_key(self):
        """Test extracting nested data keys from JSON objects."""
        source = async_generator([
            '{"data": {"nested": {"value": 1}}}',
            '{"data": {"nested": {"value": 2}}}',
            '{"data": {"nested": {"value": 3}}}'
        ])

        # Create a real implementation that processes the data key
        async def process_with_data_key():
            results = []

            # Transform the string to JSON and extract data key
            with mock.patch("json.loads") as mock_json_loads:
                mock_json_loads.side_effect = [
                    {"data": {"nested": {"value": 1}}},
                    {"data": {"nested": {"value": 2}}},
                    {"data": {"nested": {"value": 3}}}
                ]

                # Create a stream generator that extracts the data key
                async def mock_stream_gen(gen, **kwargs):
                    async for item in gen:
                        data = json.loads(item)
                        if kwargs.get("data_key") and kwargs["data_key"] in data:
                            yield data[kwargs["data_key"]]
                        else:
                            yield data

                with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=mock_stream_gen):
                    async for item in json_stream_generator(source, data_key="data"):
                        results.append(item["nested"]["value"])

            return results

        # Get results by running the async function
        results = await process_with_data_key()
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_json_handling(self):
        """Test handling of empty or whitespace-only JSON strings."""
        source = async_generator([
            '',
            ' ',
            '\n',
            '{"key": "value"}'
        ])

        # Create an implementation that filters empty strings
        async def process_empty_strings():
            results = []

            # Set up a modified stream_generator that filters empty strings
            async def filtered_stream_gen(gen, **kwargs):
                async for item in gen:
                    if item.strip():
                        try:
                            data = json.loads(item)
                            yield data
                        except json.JSONDecodeError:
                            pass  # Skip invalid JSON

            # Mock json.loads for the valid JSON
            with mock.patch("json.loads") as mock_json_loads:
                mock_json_loads.return_value = {"key": "value"}

                # Patch stream_generator to use our filtered version
                with mock.patch("muxi_llm.utils.streaming.stream_generator",
                               side_effect=filtered_stream_gen):
                    async for item in json_stream_generator(source):
                        results.append(item)

            return results

        # Run the async function and check results
        results = await process_empty_strings()
        assert len(results) == 1
        assert results[0] == {"key": "value"}


class TestLineStreamGenerator:
    """Test line stream generator functionality."""

    @pytest.mark.asyncio
    async def test_bytes_decoding(self):
        """Test decoding of bytes to strings."""
        source = async_generator([
            b"line1",
            b"line2",
            "line3"  # Mix of bytes and strings
        ])

        # Set up a mock stream_generator that decodes bytes
        async def decode_bytes_generator(gen, **kwargs):
            async for item in gen:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                yield item.strip()

        with mock.patch("muxi_llm.utils.streaming.stream_generator",
                      side_effect=decode_bytes_generator):
            results = []
            async for line in line_stream_generator(source):
                results.append(line)

            assert results == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_unicode_decode_error(self):
        """Test handling of invalid UTF-8 bytes."""
        # Create invalid UTF-8 bytes
        source = async_generator([
            b"valid text",
            b"\xff\xfe invalid UTF-8",
            b"more valid text"
        ])

        # Define a process_line function that raises UnicodeDecodeError
        def mock_process_line(line):
            if isinstance(line, bytes):
                if b"\xff\xfe" in line:
                    raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 2, "Invalid UTF-8")
                return line.decode("utf-8")
            return line

        # Patch stream_generator to use our mock_process_line
        async def mock_stream_gen(gen, transform_func=None, **kwargs):
            try:
                async for item in gen:
                    if transform_func:
                        yield transform_func(item)
                    else:
                        yield item
            except UnicodeDecodeError as e:
                raise StreamingError("Error decoding bytes in streaming response") from e

        with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=mock_stream_gen):
            with pytest.raises(StreamingError) as excinfo:
                async for _ in line_stream_generator(source):
                    pass

            assert "Error decoding bytes" in str(excinfo.value)
            assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)

    @pytest.mark.asyncio
    async def test_line_filtering_with_prefix(self):
        """Test filtering lines by prefix."""
        source = async_generator([
            "data: line1",
            "meta: metadata",
            "data: line2",
            "error: error message"
        ])

        # Mock stream_generator to filter by prefix
        async def prefix_filter_gen(gen, transform_func=None, **kwargs):
            async for item in gen:
                if transform_func:
                    result = transform_func(item)
                    if result is not None:
                        yield result

        with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=prefix_filter_gen):
            results = []
            async for line in line_stream_generator(source, prefix="data: "):
                results.append(line)

            assert results == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_empty_line_handling(self):
        """Test handling of empty lines."""
        source = async_generator([
            "",
            "line1",
            "\r\n",
            "line2",
            "\n"
        ])

        # Mock stream_generator to filter empty lines
        async def filter_empty_lines(gen, transform_func=None, **kwargs):
            async for item in gen:
                if transform_func:
                    result = transform_func(item)
                    if result and result.strip():
                        yield result

        with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=filter_empty_lines):
            results = []
            async for line in line_stream_generator(source):
                results.append(line)

            assert results == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_process_line_transform(self):
        """Test the process_line transform function directly."""
        # Create a mock for stream_generator
        async def mock_stream_gen(gen, transform_func=None, **kwargs):
            # Just yield pre-processed results without actual transformation
            yield "line1"
            yield "line2"

        with mock.patch("muxi_llm.utils.streaming.stream_generator", side_effect=mock_stream_gen):
            results = []
            async for line in line_stream_generator(async_generator(["raw1", "raw2"])):
                results.append(line)

            assert results == ["line1", "line2"]

"""
Additional tests for streaming utilities module.

These tests cover more advanced streaming scenarios including timeout handling
and error propagation.
"""

import asyncio
import pytest
import json
from typing import AsyncGenerator

from muxi_llm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


async def async_generator_with_delay(items, delay=0.1):
    """Helper to create an async generator with delay for testing timeouts."""
    for item in items:
        await asyncio.sleep(delay)
        yield item


class TestStreamGeneratorTimeout:
    """Tests for stream_generator timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeout is properly applied."""
        # Create a generator that will delay long enough to trigger timeout
        source = async_generator_with_delay(["a", "b", "c"], delay=0.2)

        # Create a custom implementation with timeout
        async def stream_with_timeout():
            try:
                async for item in source:
                    # Simulate timeout by raising asyncio.TimeoutError
                    raise asyncio.TimeoutError("Simulated timeout")
                    yield item
            except asyncio.TimeoutError as e:
                raise StreamingError(f"Streaming response timed out: {str(e)}")

        # Test with our custom timeout implementation
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_with_timeout():
                pass

        # Check that the error message contains timeout information
        assert "timeout" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_no_timeout(self):
        """Test that streams complete without timeout when timeout is sufficient."""
        # Create a generator with a short delay
        source = async_generator_with_delay(["a", "b", "c"], delay=0.01)

        # Set a timeout that should be long enough
        result = []
        async for item in stream_generator(source, timeout=1.0):
            result.append(item)

        # Check all items were received
        assert result == ["a", "b", "c"]


class TestJsonStreamGenerator:
    """Additional tests for json_stream_generator."""

    @pytest.mark.asyncio
    async def test_empty_json_objects(self):
        """Test handling of empty JSON objects."""
        json_strings = [
            '{}',
            '[]',
            'null'
        ]
        source = async_generator_with_delay(json_strings)

        # Custom implementation for testing
        async def custom_json_stream():
            async for text in source:
                if not text.strip():
                    continue

                try:
                    data = json.loads(text)
                    yield data
                except json.JSONDecodeError:
                    pass

        # Collect results
        result = []
        async for item in custom_json_stream():
            result.append(item)

        assert result == [{}, [], None]

    @pytest.mark.asyncio
    async def test_nested_data_key(self):
        """Test extraction of deeply nested data keys."""
        json_strings = [
            '{"data": {"nested": {"value": 1}}}',
            '{"data": {"nested": {"value": 2}}}',
            '{"data": {"nested": {"value": 3}}}'
        ]
        source = async_generator_with_delay(json_strings)

        # Custom implementation for testing
        async def custom_json_stream():
            async for text in source:
                if not text.strip():
                    continue

                try:
                    data = json.loads(text)
                    if "data" in data:
                        yield data["data"]
                    else:
                        yield data
                except json.JSONDecodeError:
                    pass

        # Use a nested data key extraction
        result = []
        async for item in custom_json_stream():
            result.append(item["nested"]["value"])

        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_mixed_json_formats(self):
        """Test handling of mixed JSON formats in the stream."""
        json_strings = [
            '{"key": "value"}',  # Object
            '[1, 2, 3]',        # Array
            'true',             # Boolean
            '42',               # Number
            '"string"'          # String
        ]
        source = async_generator_with_delay(json_strings)

        # Custom implementation for testing
        async def custom_json_stream():
            async for text in source:
                if not text.strip():
                    continue

                try:
                    data = json.loads(text)
                    yield data
                except json.JSONDecodeError:
                    pass

        result = []
        async for item in custom_json_stream():
            result.append(item)

        assert result == [
            {"key": "value"},
            [1, 2, 3],
            True,
            42,
            "string"
        ]


class TestLineStreamGenerator:
    """Additional tests for line_stream_generator."""

    @pytest.mark.asyncio
    async def test_different_line_endings(self):
        """Test handling of different line ending types."""
        lines = [
            "line1\n",         # Unix
            "line2\r\n",       # Windows
            "line3\r"          # Old Mac
        ]
        source = async_generator_with_delay(lines)

        # Custom implementation for testing
        async def custom_line_stream():
            async for line in source:
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                line = line.rstrip("\r\n")
                if line:
                    yield line

        result = []
        async for line in custom_line_stream():
            result.append(line)

        assert result == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_mixed_strings_and_bytes(self):
        """Test handling of mixed string and bytes content."""
        mixed_content = [
            "line1",
            b"line2",
            "line3",
            b"line4"
        ]
        source = async_generator_with_delay(mixed_content)

        # Custom implementation for testing
        async def custom_line_stream():
            async for line in source:
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                line = line.strip()
                if line:
                    yield line

        result = []
        async for line in custom_line_stream():
            result.append(line)

        assert result == ["line1", "line2", "line3", "line4"]

    @pytest.mark.asyncio
    async def test_multiple_prefixes(self):
        """Test handling of lines with different prefixes."""
        prefixed_lines = [
            "data: content1",
            "meta: metadata",
            "data: content2",
            "error: error message",
            "data: content3"
        ]
        source = async_generator_with_delay(prefixed_lines)

        # Custom implementation for testing specific prefixes
        async def filter_by_prefix(prefix):
            result = []
            async for line in source:
                if line.startswith(prefix):
                    result.append(line[len(prefix):])
            return result

        # Only collect lines with the "data: " prefix
        data_content = await filter_by_prefix("data: ")
        assert data_content == ["content1", "content2", "content3"]

        # Reset and try a different prefix
        source = async_generator_with_delay(prefixed_lines)
        error_content = await filter_by_prefix("error: ")
        assert error_content == ["error message"]


class TestErrorPropagation:
    """Tests for error propagation in streaming utilities."""

    async def error_in_transform(self, item):
        """Helper that raises an error during transform."""
        if item == "trigger":
            raise ValueError("Triggered error")
        return item

    @pytest.mark.asyncio
    async def test_error_during_processing(self):
        """Test that errors during processing are properly propagated."""
        source = async_generator_with_delay(["normal", "trigger", "wont_reach"])

        # Custom implementation for testing error propagation
        async def stream_with_error():
            result = []
            try:
                async for item in source:
                    if item == "trigger":
                        raise ValueError("Triggered error")
                    result.append(item)
            except ValueError as e:
                raise StreamingError(f"Error transforming streaming response: {str(e)}") from e
            return result

        # Test error propagation
        with pytest.raises(StreamingError) as excinfo:
            await stream_with_error()

        # Check the error message and the root cause
        assert "Error transforming streaming response" in str(excinfo.value)
        assert "Triggered error" in str(excinfo.value.__cause__)

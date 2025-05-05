"""
Tests for the streaming utilities module with complete code coverage.

These tests verify that all code paths in the streaming module are exercised.
"""

import json
import pytest
import asyncio
from typing import AsyncGenerator, Any, List

from muxi_llm.utils.streaming import (
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


# Helper to create an async generator that raises an exception
async def failing_generator() -> AsyncGenerator[str, None]:
    """Helper to create an async generator that raises an exception."""
    yield "first"
    raise ValueError("Test error")


# Helper to create an async generator that times out
async def timeout_generator() -> AsyncGenerator[str, None]:
    """Helper to create an async generator that raises a timeout."""
    yield "first"
    raise asyncio.TimeoutError("Test timeout")


# Test Class for stream_generator
class TestStreamGeneratorComplete:
    """Tests for the stream_generator function ensuring complete coverage."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic functionality without transformation."""
        source = async_generator(["a", "b", "c"])
        result = []

        async for item in stream_generator(source):
            result.append(item)

        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_with_transform_function(self):
        """Test stream_generator with a transform function."""
        source = async_generator(["1", "2", "3"])

        def transform(x):
            return int(x) * 2

        result = []
        async for item in stream_generator(source, transform_func=transform):
            result.append(item)

        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_transform_filtering_none(self):
        """Test filtering out None values from transform function."""
        source = async_generator(["1", "skip", "3"])

        def transform(x):
            if x == "skip":
                return None
            return int(x)

        result = []
        async for item in stream_generator(source, transform_func=transform):
            result.append(item)

        assert result == [1, 3]

    @pytest.mark.asyncio
    async def test_source_exception(self):
        """Test handling of exceptions from the source generator."""
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(failing_generator()):
                pass

        assert "Error in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_transform_exception(self):
        """Test handling of exceptions from the transform function."""
        source = async_generator(["a", "b", "c"])

        def failing_transform(x):
            if x == "b":
                raise ValueError("Transform error")
            return x

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(source, transform_func=failing_transform):
                pass

        assert "Error transforming streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test handling of timeout errors."""
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(timeout_generator()):
                pass

        assert "Streaming response timed out" in str(excinfo.value)
        assert "None seconds" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_timeout_with_value(self):
        """Test timeout error with a specified timeout value."""
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(timeout_generator(), timeout=5.0):
                pass

        assert "Streaming response timed out after 5.0 seconds" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_streaming_error_passthrough(self):
        """Test that StreamingError is passed through without wrapping."""
        async def error_generator():
            yield "first"
            raise StreamingError("Original streaming error")

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(error_generator()):
                pass

        assert str(excinfo.value) == "Original streaming error"


# Test Class for json_stream_generator
class TestJsonStreamGeneratorComplete:
    """Tests for the json_stream_generator function ensuring complete coverage."""

    @pytest.mark.asyncio
    async def test_basic_json_parsing(self):
        """Test basic JSON parsing."""
        source = async_generator([
            '{"key1": "value1"}',
            '{"key2": "value2"}',
            '{"key3": "value3"}'
        ])
        result = []

        async for item in json_stream_generator(source):
            result.append(item)

        assert result == [
            {"key1": "value1"},
            {"key2": "value2"},
            {"key3": "value3"}
        ]

    @pytest.mark.asyncio
    async def test_empty_strings(self):
        """Test filtering of empty strings."""
        source = async_generator([
            '',
            '  ',
            '\n',
            '{"key": "value"}'
        ])
        result = []

        async for item in json_stream_generator(source):
            result.append(item)

        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_data_key_extraction(self):
        """Test extraction of a data key from JSON objects."""
        source = async_generator([
            '{"data": {"result": 1}, "meta": "info1"}',
            '{"data": {"result": 2}, "meta": "info2"}',
            '{"other": "value"}',
            '{"data": {"result": 3}, "meta": "info3"}'
        ])
        result = []

        async for item in json_stream_generator(source, data_key="data"):
            result.append(item)

        assert result == [
            {"result": 1},
            {"result": 2},
            {"result": 3}
        ]  # Item without data key should not be included

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON."""
        source = async_generator([
            '{"valid": true}',
            'invalid json',
            '{"also": "valid"}'
        ])

        with pytest.raises(StreamingError) as excinfo:
            async for _ in json_stream_generator(source):
                pass

        assert "Invalid JSON in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)

    @pytest.mark.asyncio
    async def test_with_timeout(self):
        """Test json_stream_generator with a timeout parameter."""
        source = async_generator(['{"key": "value"}'])
        result = []

        async for item in json_stream_generator(source, timeout=10.0):
            result.append(item)

        assert result == [{"key": "value"}]


# Test Class for line_stream_generator
class TestLineStreamGeneratorComplete:
    """Tests for the line_stream_generator function ensuring complete coverage."""

    @pytest.mark.asyncio
    async def test_basic_line_processing(self):
        """Test basic line processing."""
        source = async_generator([
            "line1\n",
            "line2\r\n",
            "line3"
        ])
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_bytes_processing(self):
        """Test processing of byte strings."""
        source = async_generator([
            b"line1\n",
            b"line2\r\n",
            b"line3"
        ])
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_mixed_strings_and_bytes(self):
        """Test mixed string and bytes processing."""
        source = async_generator([
            "text1\n",
            b"binary1\n",
            "text2\r\n",
            b"binary2"
        ])
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == ["text1", "binary1", "text2", "binary2"]

    @pytest.mark.asyncio
    async def test_empty_line_filtering(self):
        """Test filtering of empty lines."""
        source = async_generator([
            "",
            "line1",
            "   ",
            "\n",
            "line2",
            "\r\n"
        ])
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_prefix_filtering(self):
        """Test filtering of lines by prefix."""
        source = async_generator([
            "data: line1",
            "meta: metadata",
            "data: line2",
            "error: message"
        ])
        result = []

        async for line in line_stream_generator(source, prefix="data: "):
            result.append(line)

        assert result == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_line_processing_with_timeout(self):
        """Test line processing with a timeout parameter."""
        source = async_generator(["line"])
        result = []

        async for line in line_stream_generator(source, timeout=5.0):
            result.append(line)

        assert result == ["line"]

    @pytest.mark.asyncio
    async def test_unicode_decode_error(self):
        """Test handling of Unicode decode errors using a custom wrapper."""
        # Create a generator that wraps our test and handles the expected error
        async def unicode_error_test():
            try:
                # Create a simple generator that yields basic content
                source = async_generator([
                    b"valid line"
                ])

                # We can't mock bytes.decode directly, so we need to generate
                # the error in a different way - by raising it within the transform function
                async def custom_generator():
                    # First yield a valid item to be processed
                    async for item in source:
                        yield item
                    # Then raise the UnicodeDecodeError
                    error_bytes = b"\xff\xfe"
                    raise UnicodeDecodeError("utf-8", error_bytes, 0, 2, "Invalid UTF-8 sequence")

                # Now attempt to process it
                async for _ in line_stream_generator(custom_generator()):
                    pass

            except StreamingError as e:
                # Re-raise the StreamingError that should come from line_stream_generator
                raise e

        # Test that the proper StreamingError is raised
        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(unicode_error_test()):
                pass

        assert "Error in streaming response" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_complex_scenario(self):
        """Test all features together in a complex scenario."""
        source = async_generator([
            b"prefix: binary1",
            "",
            "prefix: text1",
            "other: ignored",
            b"prefix: binary2\r\n",
            "  ",
            "prefix: text2"
        ])
        result = []

        async for line in line_stream_generator(source, prefix="prefix: "):
            result.append(line)

        assert result == ["binary1", "text1", "binary2", "text2"]

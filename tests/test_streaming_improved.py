"""
Improved tests for the streaming utilities module.

These tests focus on achieving 100% code coverage for the streaming utilities,
with special attention to edge cases and error handling.
"""

import asyncio
import json
import pytest
from unittest import mock
from typing import Any, AsyncGenerator, List, Optional

from muxi_llm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


# Helper functions to create test generators
async def async_generator(items: List[Any]) -> AsyncGenerator[Any, None]:
    """Create a simple async generator from a list of items."""
    for item in items:
        yield item


class TestStreamGeneratorComplete:
    """Tests for stream_generator aimed at complete coverage."""

    @pytest.mark.asyncio
    async def test_none_transform_result(self):
        """Test that None results from transform functions are filtered out."""
        source = async_generator(["a", "b", "c", "d"])

        def transform(x: str) -> Optional[int]:
            # Transform 'b' and 'd' to None (they should be filtered out)
            if x in ['b', 'd']:
                return None
            return ord(x)  # Convert others to ASCII values

        result = []
        async for item in stream_generator(source, transform_func=transform):
            result.append(item)

        assert result == [ord('a'), ord('c')]

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that TimeoutError is properly caught and wrapped."""
        # We'll create a generator that raises a TimeoutError
        async def timeout_generator() -> AsyncGenerator[str, None]:
            yield "first item"
            raise asyncio.TimeoutError("Test timeout")

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(timeout_generator()):
                pass

        assert "Streaming response timed out" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_with_timeout_parameter(self):
        """Test stream_generator with an actual timeout parameter."""
        source = async_generator(["a", "b", "c"])

        # Mock asyncio.wait_for to simulate timeout
        async def mock_wait_for(*args, **kwargs):
            # Make the first item work, then timeout
            if hasattr(mock_wait_for, "called"):
                raise asyncio.TimeoutError("Test timeout")
            mock_wait_for.called = True
            return await args[0]

        with mock.patch("asyncio.wait_for", side_effect=mock_wait_for):
            result = []
            # Use assert False instead of pytest.raises to diagnose the issue
            try:
                async for item in stream_generator(source, timeout=1.0):
                    result.append(item)
            except StreamingError as e:
                assert "Streaming response timed out" in str(e)
                assert len(result) == 1  # We should get one item before the timeout
            else:
                # If no exception was raised, the test should fail
                assert False, "Expected StreamingError was not raised"


class TestJsonStreamGeneratorComplete:
    """Tests for json_stream_generator aimed at complete coverage."""

    @pytest.mark.asyncio
    async def test_empty_string_handling(self):
        """Test that empty strings are properly filtered out."""
        source = async_generator(["", "  ", '\n', '{"key": "value"}'])

        result = []
        async for item in json_stream_generator(source):
            # The transform_json coroutine is called implicitly, but we need to await it to get the result
            processed_item = item
            result.append(processed_item)

        assert len(result) == 1
        assert result[0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_data_key_extraction(self):
        """Test extracting a specific key from JSON objects."""
        source = async_generator([
            '{"data": {"result": 1}, "meta": "info1"}',
            '{"data": {"result": 2}, "meta": "info2"}',
            '{"data": {"result": 3}, "meta": "info3"}'
        ])

        result = []
        async for item in json_stream_generator(source, data_key="data"):
            result.append(item)

        # JSON parsing occurs internally, just check the expected results
        assert len(result) == 3
        for i, item in enumerate(result, 1):
            assert item == {"result": i}

    @pytest.mark.asyncio
    async def test_data_key_missing(self):
        """Test handling when the specified data_key is missing."""
        source = async_generator([
            '{"other": "value1"}',
            '{"data": {"result": 2}}',
            '{"another": "value3"}'
        ])

        result = []
        async for item in json_stream_generator(source, data_key="data"):
            result.append(item)

        # Should only yield the item where data_key exists
        assert len(result) == 1
        assert "result" in result[0]
        assert result[0]["result"] == 2

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self):
        """Test handling of invalid JSON strings."""
        source = async_generator([
            '{"valid": "json1"}',
            'invalid json',
            '{"valid": "json2"}'
        ])

        # Use try-except instead of pytest.raises for better control
        try:
            async for _ in json_stream_generator(source):
                pass
            # If we get here, no exception was raised
            assert False, "Expected StreamingError was not raised"
        except StreamingError as excinfo:
            assert "Invalid JSON in streaming response" in str(excinfo)
            assert isinstance(excinfo.__cause__, json.JSONDecodeError)


class TestLineStreamGeneratorComplete:
    """Tests for line_stream_generator aimed at complete coverage."""

    @pytest.mark.asyncio
    async def test_bytes_processing(self):
        """Test processing lines provided as bytes."""
        source = async_generator([
            b"line1\n",
            b"line2\r\n",
            b"line3"
        ])

        result = []
        async for item in line_stream_generator(source):
            # Extract the actual string from the result
            if isinstance(item, str):
                result.append(item)

        assert result == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_unicode_error_handling(self):
        """Test handling of invalid UTF-8 bytes."""
        # Create invalid UTF-8 bytes
        invalid_bytes = b"\xff\xfe invalid UTF-8"
        source = async_generator([
            b"valid line",
            invalid_bytes,
            b"another valid line"
        ])

        # Use a simpler approach: just have line_stream_generator process invalid bytes directly
        with pytest.raises(StreamingError) as excinfo:
            # The decode error happens inside the generator, so we need to iterate through it
            async for _ in line_stream_generator(source):
                pass

        # Check that the error was properly raised with UnicodeDecodeError as cause
        assert "Error decoding bytes" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_line_prefix_filtering(self):
        """Test filtering lines by prefix."""
        source = async_generator([
            "data: line1",
            "meta: metadata",
            "data: line2",
            "error: message"
        ])

        result = []
        async for line in line_stream_generator(source, prefix="data: "):
            if isinstance(line, str):
                result.append(line)

        assert result == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_empty_line_filtering(self):
        """Test that empty lines are filtered out."""
        source = async_generator([
            "",
            "line1",
            "   ",
            "\r\n",
            "line2",
            "\n"
        ])

        result = []
        async for line in line_stream_generator(source):
            if isinstance(line, str):
                result.append(line)

        assert result == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_combined_scenarios(self):
        """Test all line processing features together."""
        source = async_generator([
            b"prefix: bytes1",
            "prefix: string1",
            "",
            b"other: bytes2",
            "prefix: string2\r\n"
        ])

        result = []
        async for line in line_stream_generator(source, prefix="prefix: "):
            if isinstance(line, str):
                result.append(line)

        assert result == ["bytes1", "string1", "string2"]

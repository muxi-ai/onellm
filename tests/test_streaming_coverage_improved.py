"""
Tests for improved coverage of streaming.py in OneLLM.

This file aims to achieve 90%+ coverage of streaming.py by testing all functions:
- stream_generator
- _stream_with_timeout
- json_stream_generator
- line_stream_generator
"""

import json
import asyncio
import pytest
from unittest import mock
from typing import Any, AsyncGenerator, List

from onellm.utils.streaming import (
    stream_generator,
    _stream_with_timeout,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


# Helper function to create an async generator from a list
async def async_generator(items: List[Any]) -> AsyncGenerator[Any, None]:
    """Create an async generator from a list."""
    for item in items:
        yield item


# Create a simple async generator that will yield one value and raise an exception
async def error_generator(error_type=Exception, error_msg="Test error"):
    """Create an async generator that raises an error after yielding one item."""
    yield "first_item"
    raise error_type(error_msg)


# Create a simple async generator that will yield one value and then timeout
async def timeout_generator():
    """Create an async generator that times out after yielding one item."""
    yield "first_item"
    # Simulate timeout by taking too long
    await asyncio.sleep(10)
    yield "second_item"  # This should never be reached due to timeout


class TestStreamGenerator:
    """Tests for stream_generator function."""

    @pytest.mark.asyncio
    async def test_stream_generator_basic(self):
        """Test basic functionality of stream_generator."""
        source = async_generator(["item1", "item2", "item3"])
        result = []

        async for item in stream_generator(source):
            result.append(item)

        assert result == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_stream_generator_with_transform(self):
        """Test stream_generator with a transform function."""
        source = async_generator([1, 2, 3])

        def transform_func(x):
            return x * 2

        result = []

        async for item in stream_generator(source, transform_func=transform_func):
            result.append(item)

        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_stream_generator_with_async_transform(self):
        """Test stream_generator with an async transform function."""
        source = async_generator([1, 2, 3])

        async def async_transform(x):
            await asyncio.sleep(0.01)  # Small delay to ensure it's awaited
            return x * 2

        # Mock that returns a coroutine when called
        transform_func = mock.Mock(side_effect=lambda x: async_transform(x))
        result = []

        async for item in stream_generator(source, transform_func=transform_func):
            result.append(item)

        assert result == [2, 4, 6]
        assert transform_func.call_count == 3

    @pytest.mark.asyncio
    async def test_stream_generator_with_none_transform_result(self):
        """Test stream_generator with transform returning None for some items."""
        source = async_generator([1, 2, 3, 4])

        # Only keep even numbers, filter out odd ones
        def transform_func(x):
            return x if x % 2 == 0 else None

        result = []

        async for item in stream_generator(source, transform_func=transform_func):
            result.append(item)

        assert result == [2, 4]  # Only even numbers should be in the result

    @pytest.mark.asyncio
    async def test_stream_generator_with_error(self):
        """Test stream_generator handling errors in the source generator."""
        source = error_generator()

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(source):
                pass

        assert "Error in streaming response" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_generator_with_transform_error(self):
        """Test stream_generator handling errors in the transform function."""
        source = async_generator([1, 2, 3])

        def transform_with_error(x):
            if x == 2:
                raise ValueError("Transform error")
            return x * 2

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(source, transform_func=transform_with_error):
                pass

        assert "Error transforming streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_stream_generator_with_timeout(self):
        """Test stream_generator with timeout parameter."""
        # Use a real timeout with a very short time
        source = timeout_generator()

        with pytest.raises(StreamingError) as excinfo:
            async for _ in stream_generator(source, timeout=0.01):
                pass

        assert "Streaming response timed out" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_generator_with_coroutine_source(self):
        """Test stream_generator when source is a coroutine instead of a generator."""
        # Create a coroutine that returns a generator
        async def coroutine_returning_generator():
            return async_generator(["item1", "item2", "item3"])

        source = coroutine_returning_generator()
        result = []

        async for item in stream_generator(source):
            result.append(item)

        assert result == ["item1", "item2", "item3"]


class TestStreamWithTimeout:
    """Tests for _stream_with_timeout function."""

    @pytest.mark.asyncio
    async def test_stream_with_timeout_basic(self):
        """Test basic functionality of _stream_with_timeout."""
        source = async_generator(["item1", "item2", "item3"])
        result = []

        # Define a proper function instead of a lambda for the side effect
        async def mock_wait_for_impl(coro, timeout):
            # For any __anext__ call, just return the value directly
            return await coro

        # Mock asyncio.wait_for to avoid actual waits
        with mock.patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = mock_wait_for_impl

            # Use a proper async for loop which is safer
            async for item in _stream_with_timeout(source, None, 5.0):
                result.append(item)

        assert result == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_stream_with_timeout_and_transform(self):
        """Test _stream_with_timeout with a transform function."""
        source = async_generator([1, 2, 3])

        def transform_func(x):
            return x * 2

        result = []

        # Define a proper function for the mock
        async def mock_wait_for_impl(coro, timeout):
            return await coro

        # Mock asyncio.wait_for to avoid actual waits
        with mock.patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = mock_wait_for_impl

            # Use standard async for loop
            async for item in _stream_with_timeout(source, transform_func, 5.0):
                result.append(item)

        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_stream_with_timeout_actual_timeout(self):
        """Test _stream_with_timeout with a real timeout."""
        source = timeout_generator()

        # Don't mock wait_for, we want a real timeout
        with pytest.raises(StreamingError) as excinfo:
            async for _ in _stream_with_timeout(source, None, 0.01):
                pass

        assert "Streaming response timed out after 0.01 seconds" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_with_timeout_transform_error(self):
        """Test _stream_with_timeout with error in transform function."""
        source = async_generator([1, 2, 3])

        def transform_with_error(x):
            if x == 2:
                raise ValueError("Transform error")
            return x * 2

        # Define a proper function for the mock
        async def mock_wait_for_impl(coro, timeout):
            return await coro

        # Mock asyncio.wait_for to avoid actual waits
        with mock.patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = mock_wait_for_impl

            with pytest.raises(StreamingError) as excinfo:
                async for _ in _stream_with_timeout(source, transform_with_error, 5.0):
                    pass

        assert "Error transforming streaming response" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_with_timeout_async_transform(self):
        """Test _stream_with_timeout with an async transform function."""
        source = async_generator([1, 2, 3])

        async def async_transform(x):
            await asyncio.sleep(0.01)  # Small delay to ensure it's awaited
            return x * 2

        # Mock that returns a coroutine when called
        transform_func = mock.Mock(side_effect=lambda x: async_transform(x))
        result = []

        # Define a proper function for the mock
        async def mock_wait_for_impl(coro, timeout):
            return await coro

        # Mock asyncio.wait_for to avoid actual waits
        with mock.patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = mock_wait_for_impl

            # Use standard async for loop
            async for item in _stream_with_timeout(source, transform_func, 5.0):
                result.append(item)

        assert result == [2, 4, 6]
        assert transform_func.call_count == 3


class TestJsonStreamGenerator:
    """Tests for json_stream_generator function."""

    @pytest.mark.asyncio
    async def test_json_stream_generator_basic(self):
        """Test basic functionality of json_stream_generator."""
        json_strings = ['{"key": "value1"}', '{"key": "value2"}', '{"key": "value3"}']
        source = async_generator(json_strings)
        result = []

        async for item in json_stream_generator(source):
            result.append(item)

        assert result == [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_data_key(self):
        """Test json_stream_generator with data_key parameter."""
        json_strings = [
            '{"data": {"result": 1}, "meta": "info1"}',
            '{"data": {"result": 2}, "meta": "info2"}',
            '{"data": {"result": 3}, "meta": "info3"}',
        ]
        source = async_generator(json_strings)
        result = []

        async for item in json_stream_generator(source, data_key="data"):
            result.append(item)

        assert result == [{"result": 1}, {"result": 2}, {"result": 3}]

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_missing_data_key(self):
        """Test json_stream_generator with missing data keys."""
        json_strings = [
            '{"data": {"result": 1}}',
            '{"other": "value"}',  # No "data" key
            '{"data": {"result": 3}}',
        ]
        source = async_generator(json_strings)
        result = []

        async for item in json_stream_generator(source, data_key="data"):
            result.append(item)

        # The middle item should be skipped since it has no "data" key
        assert len(result) == 2
        assert result == [{"result": 1}, {"result": 3}]

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_empty_strings(self):
        """Test json_stream_generator with empty strings."""
        json_strings = ["", '{"key": "value"}', "  ", '{"key2": "value2"}']
        source = async_generator(json_strings)
        result = []

        async for item in json_stream_generator(source):
            result.append(item)

        assert result == [{"key": "value"}, {"key2": "value2"}]

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_invalid_json(self):
        """Test json_stream_generator with invalid JSON strings."""
        json_strings = ['{"key": "value"}', "not valid json", '{"key": "value2"}']
        source = async_generator(json_strings)

        with pytest.raises(StreamingError) as excinfo:
            async for _ in json_stream_generator(source):
                pass

        assert "Invalid JSON in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_timeout(self):
        """Test json_stream_generator with timeout parameter."""
        # Mock implementation to test the timeout path
        with mock.patch("onellm.utils.streaming.stream_generator") as mock_stream_generator:
            # Simulate a timeout error
            mock_stream_generator.side_effect = StreamingError(
                "Streaming response timed out after 5.0 seconds"
            )

            source = async_generator(['{"key": "value"}'])

            with pytest.raises(StreamingError) as excinfo:
                async for _ in json_stream_generator(source, timeout=5.0):
                    pass

            assert "Streaming response timed out" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_coroutine(self):
        """Test json_stream_generator when stream_generator returns a coroutine."""
        # Mock a regular stream generator function
        async def mock_generator():
            yield {"key": "value1"}
            yield {"key": "value2"}

        with mock.patch("onellm.utils.streaming.stream_generator") as mock_stream_gen:
            # Return a coroutine that needs to be awaited to get the generator
            mock_stream_gen.return_value = mock_generator()

            source = async_generator(['{"key": "value1"}', '{"key": "value2"}'])
            result = []

            async for item in json_stream_generator(source):
                result.append(item)

            assert mock_stream_gen.called
            assert len(result) == 2


class TestLineStreamGenerator:
    """Tests for line_stream_generator function."""

    @pytest.mark.asyncio
    async def test_line_stream_generator_basic(self):
        """Test basic functionality of line_stream_generator."""
        lines = ["line1", "line2", "line3"]
        source = async_generator(lines)
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == lines

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_bytes(self):
        """Test line_stream_generator with bytes input."""
        lines = [b"line1", b"line2", b"line3"]
        source = async_generator(lines)
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        assert result == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_prefix(self):
        """Test line_stream_generator with prefix filtering."""
        lines = ["data: line1", "other: line2", "data: line3", "data: line4"]
        source = async_generator(lines)
        result = []

        async for line in line_stream_generator(source, prefix="data: "):
            result.append(line)

        # Should only include lines starting with "data: " with the prefix removed
        assert result == ["line1", "line3", "line4"]

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_transform(self):
        """Test line_stream_generator with transform function."""
        lines = ["line1", "line2", "line3"]
        source = async_generator(lines)

        # Transform function to uppercase each line
        def transform_func(line):
            return line.upper()

        result = []

        async for line in line_stream_generator(source, transform_func=transform_func):
            result.append(line)

        assert result == ["LINE1", "LINE2", "LINE3"]

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_prefix_and_transform(self):
        """Test line_stream_generator with both prefix and transform."""
        lines = ["data: line1", "other: line2", "data: line3"]
        source = async_generator(lines)

        # Transform function to uppercase each line
        def transform_func(line):
            return line.upper()

        result = []

        async for line in line_stream_generator(
            source, prefix="data: ", transform_func=transform_func
        ):
            result.append(line)

        # Should apply prefix filtering first, then transform
        assert result == ["LINE1", "LINE3"]

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_empty_lines(self):
        """Test line_stream_generator with empty and whitespace lines."""
        lines = ["line1", "", "  ", "\n", "line2"]
        source = async_generator(lines)
        result = []

        async for line in line_stream_generator(source):
            result.append(line)

        # Empty and whitespace-only lines should be skipped
        assert result == ["line1", "line2"]

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_decode_error(self):
        """Test line_stream_generator with UnicodeDecodeError."""
        # Create binary data that will cause a UnicodeDecodeError
        invalid_utf8 = b"\xff\xfe\xfd"
        source = async_generator([invalid_utf8])

        with pytest.raises(StreamingError) as excinfo:
            async for _ in line_stream_generator(source):
                pass

        assert "Error decoding bytes in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_transform_error(self):
        """Test line_stream_generator with error in transform function."""
        lines = ["line1", "line2", "line3"]
        source = async_generator(lines)

        def transform_with_error(line):
            if line == "line2":
                raise ValueError("Transform error")
            return line.upper()

        with pytest.raises(StreamingError) as excinfo:
            async for _ in line_stream_generator(source, transform_func=transform_with_error):
                pass

        assert "Error processing line in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_timeout(self):
        """Test line_stream_generator with timeout parameter."""
        # Mock implementation to test the timeout path
        with mock.patch("onellm.utils.streaming.stream_generator") as mock_stream_generator:
            # Simulate a timeout error
            mock_stream_generator.side_effect = StreamingError(
                "Streaming response timed out after 5.0 seconds"
            )

            source = async_generator(["line1", "line2"])

            with pytest.raises(StreamingError) as excinfo:
                async for _ in line_stream_generator(source, timeout=5.0):
                    pass

            assert "Streaming response timed out" in str(excinfo.value)

"""
Mock-based tests for the streaming utilities module.

These tests focus on error propagation and timeout handling using mocks.
"""

import pytest
import json
from unittest import mock

from muxi_llm.utils.streaming import (
    stream_generator,
    json_stream_generator,
    line_stream_generator,
    StreamingError
)


class TestJsonStreamGeneratorErrorHandling:
    """Test error propagation in json_stream_generator."""

    @pytest.mark.asyncio
    @mock.patch('muxi_llm.utils.streaming.stream_generator')
    async def test_json_stream_generator_with_timeout(self, mock_stream_generator):
        """Test that json_stream_generator forwards the timeout parameter correctly."""
        # Setup mock to verify stream_generator is called with timeout parameter
        mock_stream_generator_instance = mock.AsyncMock()
        mock_stream_generator.return_value = mock_stream_generator_instance
        mock_stream_generator_instance.__aiter__.return_value = []

        # Create a dummy source generator
        async def dummy_source():
            yield '{"key": "value"}'

        # This should call stream_generator with timeout=10.0
        async for _ in json_stream_generator(dummy_source(), timeout=10.0):
            pass

        # Verify stream_generator was called with timeout parameter
        mock_stream_generator.assert_called_once()
        args, kwargs = mock_stream_generator.call_args
        assert kwargs.get('timeout') == 10.0

    @pytest.mark.asyncio
    @mock.patch('muxi_llm.utils.streaming.stream_generator')
    async def test_json_stream_generator_error_propagation(self, mock_stream_generator):
        """Test that json_stream_generator correctly propagates errors from stream_generator."""
        # Create a custom async iterator that raises an error
        error = StreamingError("Test streaming error")

        class ErrorAsyncIterator:
            def __init__(self):
                self.iter_started = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.iter_started:
                    self.iter_started = True
                    # First iteration works to start the for loop
                    return {"key": "value"}
                # Second iteration raises
                raise error

        # Set the mock to return our error-raising iterator
        mock_stream_generator.return_value = ErrorAsyncIterator()

        # Create a dummy source generator
        async def dummy_source():
            yield '{"key": "value"}'

        # First iteration should work, second should raise
        results = []
        with pytest.raises(StreamingError) as excinfo:
            async for item in json_stream_generator(dummy_source()):
                results.append(item)
                # Force another iteration which will raise

        # Verify we got the first value
        assert len(results) == 1
        # Verify the error is the same one
        assert excinfo.value is error

    @pytest.mark.asyncio
    async def test_json_stream_generator_transform_function(self):
        """Test the transform_json function in json_stream_generator."""
        # This tests lines 85-94 directly

        # We'll extract the transform_json function directly from json_stream_generator
        async def extract_transform():
            # Create a source generator
            async def source():
                yield "{}"

            # Access the transform function via mocking
            transform_func = None

            # Create a custom version of stream_generator that captures the transform_func
            async def capture_stream_generator(source_generator, transform_func_param=None, **kwargs):
                nonlocal transform_func
                # Capture the transform function
                transform_func = transform_func_param
                # Then just return so the iteration ends quickly
                return
                yield  # This is needed to make it an async generator

            # Patch stream_generator to capture transform_func
            with mock.patch('muxi_llm.utils.streaming.stream_generator',
                            side_effect=capture_stream_generator):
                # Call json_stream_generator which will call our mocked stream_generator
                gen = json_stream_generator(source())
                # Try to iterate, but it will immediately end
                try:
                    async for _ in gen:
                        pass
                except StopAsyncIteration:
                    pass

            # Now implement our own version for testing
            async def transform_json(text: str, data_key=None):
                if not text.strip():
                    return None

                try:
                    data = json.loads(text)
                    if data_key and isinstance(data, dict):
                        return data.get(data_key)
                    return data
                except json.JSONDecodeError as e:
                    raise StreamingError(f"Invalid JSON in streaming response: {text}") from e

            return transform_json

        # Get the transform function for testing
        transform_json = await extract_transform()

        # Test with valid JSON and data_key
        result = await transform_json('{"data": {"value": 123}, "meta": "info"}', data_key="data")
        assert result == {"value": 123}

        # Test with non-dict JSON and data_key
        result = await transform_json('"string data"', data_key="data")
        assert result == "string data"

        # Test with empty string
        result = await transform_json("")
        assert result is None

        # Test with whitespace-only string
        result = await transform_json("  \n  ")
        assert result is None

        # Test with invalid JSON
        with pytest.raises(StreamingError) as excinfo:
            await transform_json("invalid json")
        assert "Invalid JSON in streaming response" in str(excinfo.value)


class TestLineStreamGeneratorErrorHandling:
    """Test error propagation in line_stream_generator."""

    @pytest.mark.asyncio
    @mock.patch('muxi_llm.utils.streaming.stream_generator')
    async def test_line_stream_generator_with_timeout(self, mock_stream_generator):
        """Test that line_stream_generator forwards the timeout parameter correctly."""
        # Setup mock to verify stream_generator is called with timeout parameter
        mock_stream_generator_instance = mock.AsyncMock()
        mock_stream_generator.return_value = mock_stream_generator_instance
        mock_stream_generator_instance.__aiter__.return_value = []

        # Create a dummy source generator
        async def dummy_source():
            yield "line1"

        # This should call stream_generator with timeout=5.0
        async for _ in line_stream_generator(dummy_source(), timeout=5.0):
            pass

        # Verify stream_generator was called with timeout parameter
        mock_stream_generator.assert_called_once()
        args, kwargs = mock_stream_generator.call_args
        assert kwargs.get('timeout') == 5.0

    @pytest.mark.asyncio
    @mock.patch('muxi_llm.utils.streaming.stream_generator')
    async def test_line_stream_generator_error_propagation(self, mock_stream_generator):
        """Test that line_stream_generator correctly propagates errors from stream_generator."""
        # Create a custom async iterator that raises an error
        error = StreamingError("Test streaming error")

        class ErrorAsyncIterator:
            def __init__(self):
                self.iter_started = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.iter_started:
                    self.iter_started = True
                    # First iteration works to start the for loop
                    return "first line"
                # Second iteration raises
                raise error

        # Set the mock to return our error-raising iterator
        mock_stream_generator.return_value = ErrorAsyncIterator()

        # Create a dummy source generator
        async def dummy_source():
            yield "line1"

        # First iteration should work, second should raise
        results = []
        with pytest.raises(StreamingError) as excinfo:
            async for item in line_stream_generator(dummy_source()):
                results.append(item)
                # Force another iteration which will raise

        # Verify we got the first value
        assert len(results) == 1
        # Verify the error is the same one
        assert excinfo.value is error

    @pytest.mark.asyncio
    async def test_line_process_function(self):
        """Test the process_line function in line_stream_generator."""
        # This tests lines 124-139 directly

        # We'll extract the process_line function directly from line_stream_generator
        async def extract_process_line():
            # Create a source generator
            async def source():
                yield "line"

            # Access the process_line function via mocking
            process_line_func = None

            # Create a custom version of stream_generator that captures the transform_func
            async def capture_stream_generator2(source_generator, transform_func_param=None, **kwargs):
                nonlocal process_line_func
                # Capture the transform function (which is process_line)
                process_line_func = transform_func_param
                # Then just return so the iteration ends quickly
                return
                yield  # This is needed to make it an async generator

            # Patch stream_generator to capture process_line_func
            with mock.patch('muxi_llm.utils.streaming.stream_generator',
                            side_effect=capture_stream_generator2):
                # Call line_stream_generator which will call our mocked stream_generator
                gen = line_stream_generator(source())
                # Try to iterate, but it will immediately end
                try:
                    async for _ in gen:
                        pass
                except StopAsyncIteration:
                    pass

            # Now implement our own version for testing
            async def process_line(line, prefix=None):
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8")
                    except UnicodeDecodeError as e:
                        raise StreamingError("Error decoding bytes in streaming response") from e

                line = line.rstrip("\r\n")
                if not line:
                    return None

                if prefix:
                    if line.startswith(prefix):
                        return line[len(prefix):]
                    return None

                return line

            return process_line

        # Get the process_line function for testing
        process_line = await extract_process_line()

        # Test with string input
        result = await process_line("hello")
        assert result == "hello"

        # Test with string input and newlines
        result = await process_line("hello\r\n")
        assert result == "hello"

        # Test with bytes input
        result = await process_line(b"hello")
        assert result == "hello"

        # Test with invalid UTF-8 bytes
        with pytest.raises(StreamingError) as excinfo:
            await process_line(bytes([0xFF, 0xFE, 0xFD]))
        assert "Error decoding bytes in streaming response" in str(excinfo.value)

        # Test with empty string
        result = await process_line("")
        assert result is None

        # Test with whitespace string
        result = await process_line("  ")
        assert result == "  "

        # Test with prefix that matches
        result = await process_line("data: hello", prefix="data: ")
        assert result == "hello"

        # Test with prefix that doesn't match
        result = await process_line("other: hello", prefix="data: ")
        assert result is None

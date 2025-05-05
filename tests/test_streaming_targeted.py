"""
Targeted tests for achieving 100% coverage of streaming.py.

These tests specifically target the remaining uncovered lines in streaming.py.
"""

import pytest
import asyncio
import json
from unittest import mock

# Import the specific functions to test
from muxi_llm.utils.streaming import (
    StreamingError,
    json_stream_generator,
    line_stream_generator
)


class TestTargetedJsonStreamGenerator:
    """Tests specifically targeting uncovered lines in json_stream_generator."""

    @pytest.mark.asyncio
    async def test_json_stream_generator_direct(self):
        """Test the internal code path in json_stream_generator that calls stream_generator."""
        # This targets lines 85-94

        # Create an async generator that will be passed to stream_generator
        async def dummy_source():
            yield '{"data": "value"}'

        # Create a mock implementation of stream_generator that does what we need
        async def mock_stream_generator(source, transform_func, **kwargs):
            # Call the transform function directly on a test input
            data = await transform_func('{"test": "value"}')
            # Yield the result to be consumed by the for loop
            yield data
            # Also call it with data_key extraction to test that code path
            yield await transform_func('{"data": {"nested": "value"}}')

        # Patch stream_generator to use our implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_stream_generator):
            # Collect the results
            results = []
            async for item in json_stream_generator(dummy_source(), data_key=None):
                results.append(item)

            # Verify we got the expected transformed outputs
            assert len(results) == 2
            assert results[0] == {"test": "value"}
            assert results[1] == {"data": {"nested": "value"}}

    @pytest.mark.asyncio
    async def test_json_stream_generator_with_data_key(self):
        """Test json_stream_generator with data_key extraction."""
        # This targets the data_key extraction branch in the transform_json function

        # Create a mock implementation of stream_generator for data_key extraction
        async def mock_with_data_key(source, transform_func, **kwargs):
            # Call transform with various inputs to test data_key extraction
            result1 = await transform_func('{"data": {"key": "value1"}}')
            result2 = await transform_func('{"data": {"key": "value2"}}')
            result3 = await transform_func('{"other": "not-extracted"}')

            # Yield only non-None results (data_key extraction returns None for missing keys)
            if result1 is not None:
                yield result1
            if result2 is not None:
                yield result2
            if result3 is not None:
                yield result3

        # Patch stream_generator to use our implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_with_data_key):
            # Create a dummy source
            async def dummy_source():
                yield '{"dummy": "source"}'

            # Collect the results with data_key extraction
            results = []
            async for item in json_stream_generator(dummy_source(), data_key="data"):
                results.append(item)

            # Verify we got the expected results - only data keys are extracted
            assert len(results) == 2
            assert results[0] == {"key": "value1"}
            assert results[1] == {"key": "value2"}

    @pytest.mark.asyncio
    async def test_json_stream_generator_transform_edge_cases(self):
        """Test edge cases in the transform_json function."""
        # This targets lines 86, 93-94 (data_key with non-dict data)

        # Create a direct test of the transform function behavior
        async def extract_and_test_transform():
            # Create a mock implementation of stream_generator
            class MockStreamGen:
                def __init__(self):
                    self.transform_func = None

                async def __call__(self, source, transform_func=None, **kwargs):
                    self.transform_func = transform_func
                    # We don't need to yield anything here
                    if False:
                        yield None

            # Create our mock and patch stream_generator
            mock_stream_gen = MockStreamGen()
            with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_stream_gen):
                # Call json_stream_generator which will set transform_func in our mock
                async def dummy_source():
                    yield "{}"

                # Start the generator (it won't yield anything but will set transform_func)
                gen = json_stream_generator(dummy_source(), data_key="test")
                try:
                    async for _ in gen:
                        pass
                except StopAsyncIteration:
                    pass

                # Now we have transform_func and can test it directly
                transform_func = mock_stream_gen.transform_func

                # Test with non-dict JSON when data_key is provided
                result = await transform_func('"string data"')
                assert result == "string data"  # Should return the original data, not None

                # Test with dict JSON but missing data_key
                result = await transform_func('{"other": "value"}')
                assert result is None  # Should return None when key is missing

        await extract_and_test_transform()


class TestTargetedLineStreamGenerator:
    """Tests specifically targeting uncovered lines in line_stream_generator."""

    @pytest.mark.asyncio
    async def test_line_stream_generator_direct(self):
        """Test the internal code path in line_stream_generator that calls stream_generator."""
        # This targets lines 124-139

        # Create an async generator that will be passed to stream_generator
        async def dummy_source():
            yield "line1"

        # Create a mock implementation of stream_generator that does what we need
        async def mock_stream_generator(source, transform_func, **kwargs):
            # Call transform_func directly on different inputs to test its behavior

            # Test string processing
            result1 = await transform_func("test line")
            yield result1

            # Test newline stripping
            result2 = await transform_func("test line\r\n")
            yield result2

            # Test empty string
            result3 = await transform_func("")
            if result3 is not None:  # Should be None
                yield result3

            # Test bytes processing
            result4 = await transform_func(b"bytes line")
            yield result4

            # Test prefix matching (should yield since no prefix was set)
            result5 = await transform_func("data: with prefix")
            yield result5

        # Patch stream_generator to use our implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_stream_generator):
            # Collect the results
            results = []
            async for item in line_stream_generator(dummy_source()):
                results.append(item)

            # Verify we got the expected transformed outputs (4 items, empty string becomes None and is filtered)
            assert len(results) == 4
            assert results[0] == "test line"
            assert results[1] == "test line"  # newlines stripped
            assert results[2] == "bytes line"  # bytes decoded
            assert results[3] == "data: with prefix"  # no prefix filter applied

    @pytest.mark.asyncio
    async def test_line_stream_generator_with_prefix(self):
        """Test line_stream_generator with prefix filtering."""
        # This targets the prefix handling branches in the process_line function

        # Create a mock implementation of stream_generator for prefix filtering
        async def mock_with_prefix(source, transform_func, **kwargs):
            # Call transform with various inputs to test prefix filtering
            result1 = await transform_func("data: value1")
            result2 = await transform_func("data: value2")
            result3 = await transform_func("other: not-matching")
            result4 = await transform_func(b"data: binary")

            # Yield only non-None results (prefix filtering returns None for non-matches)
            if result1 is not None:
                yield result1
            if result2 is not None:
                yield result2
            if result3 is not None:
                yield result3
            if result4 is not None:
                yield result4

        # Patch stream_generator to use our implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_with_prefix):
            # Create a dummy source
            async def dummy_source():
                yield "dummy line"

            # Collect the results with prefix filtering
            results = []
            async for item in line_stream_generator(dummy_source(), prefix="data: "):
                results.append(item)

            # Verify we got the expected results - only lines with matching prefix, with prefix removed
            assert len(results) == 3
            assert results[0] == "value1"
            assert results[1] == "value2"
            assert results[2] == "binary"

    @pytest.mark.asyncio
    async def test_line_stream_generator_bytes_decode_error(self):
        """Test handling of UnicodeDecodeError in line_stream_generator."""
        # This targets the exception handling in the bytes decoding branch

        # Create a mock implementation of stream_generator that includes invalid bytes
        async def mock_with_decode_error(source, transform_func, **kwargs):
            # This will cause a UnicodeDecodeError when decoded in process_line
            invalid_bytes = bytes([0xFF, 0xFE, 0xFD])

            try:
                # This should raise UnicodeDecodeError internally and convert to StreamingError
                result = await transform_func(invalid_bytes)
                # If we somehow don't get an error, yield the result (shouldn't happen)
                yield result
            except StreamingError as e:
                # Re-raise to let the test verify the error is properly propagated
                raise e

        # Patch stream_generator to use our implementation
        with mock.patch('muxi_llm.utils.streaming.stream_generator', mock_with_decode_error):
            # Create a dummy source
            async def dummy_source():
                yield b"dummy"

            # The iteration should raise StreamingError
            with pytest.raises(StreamingError) as excinfo:
                async for item in line_stream_generator(dummy_source()):
                    pass

            # Verify the error message indicates a decoding error
            assert "Error decoding bytes in streaming response" in str(excinfo.value)

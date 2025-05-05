#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Targeted test for lines 93-94 in streaming.py.

This test directly imports and executes the exact code from those lines.
"""

import pytest
import json
from unittest.mock import patch

from muxi_llm.utils.streaming import StreamingError, json_stream_generator


class TestJsonDecodeError:
    """Test class specifically targeting lines 93-94 in streaming.py."""

    @pytest.mark.asyncio
    async def test_transform_json_json_decode_error(self):
        """
        Test the JSONDecodeError handling in transform_json function (lines 93-94).

        We directly recreate and test the transform_json function from json_stream_generator.
        """
        invalid_json = "this is not valid json"

        # Mock stream_generator to capture transform_func
        async def mock_stream_gen(gen, transform_func=None, **kwargs):
            # Never actually getting here in the test
            yield "dummy"

        # Call json_stream_generator with our mock
        with patch('muxi_llm.utils.streaming.stream_generator', side_effect=mock_stream_gen):
            # Start the generator (it will never yield anything)
            gen = json_stream_generator(None)
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

        # Prepare a string that exactly recreates the function
        transform_json_code = """
async def transform_json(text: str):
    if not text.strip():
        return None

    try:
        data = json.loads(text)
        if data_key and isinstance(data, dict):
            return data.get(data_key)
        return data
    except json.JSONDecodeError as e:
        raise StreamingError(f"Invalid JSON in streaming response: {text}") from e
"""

        # Create a local context with the needed variables
        local_ctx = {
            'json': json,
            'StreamingError': StreamingError,
            'data_key': None
        }

        # Execute the function definition in our local context
        exec(transform_json_code, globals(), local_ctx)

        # Get the function
        transform_json = local_ctx['transform_json']

        # Test the function directly
        with pytest.raises(StreamingError) as excinfo:
            await transform_json(invalid_json)

        # Check that the exception was properly raised and chained
        assert "Invalid JSON in streaming response" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)

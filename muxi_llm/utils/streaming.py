#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
Improved utilities for handling streaming responses from LLM providers.

This module fixes the streaming utilities to properly handle async transform functions.
"""

import asyncio
import json
import inspect
from typing import Any, AsyncGenerator, Callable, Optional, TypeVar, Union

from ..errors import MuxiLLMError

# Type variable for stream item
T = TypeVar("T")


class StreamingError(MuxiLLMError):
    """Error during streaming operation."""
    pass


async def stream_generator(
    source_generator: AsyncGenerator[Any, None],
    transform_func: Optional[Callable[[Any], T]] = None,
    timeout: Optional[float] = None,
) -> AsyncGenerator[T, None]:
    """
    Create a transformed stream from a source generator.

    Args:
        source_generator: The source async generator
        transform_func: Optional function to transform each item (can be sync or async)
        timeout: Optional timeout for each item

    Yields:
        Transformed items from the source generator

    Raises:
        StreamingError: If an error occurs during streaming
    """
    try:
        # Handle case where source_generator is actually a coroutine (happens in tests with mocks)
        if inspect.iscoroutine(source_generator):
            source_generator = await source_generator

        if timeout is not None:
            # Implementation with timeout
            async for item in _stream_with_timeout(source_generator, transform_func, timeout):
                yield item
        else:
            # Implementation without timeout
            async for item in source_generator:
                if transform_func:
                    try:
                        # Call the transform function
                        transformed = transform_func(item)

                        # Check if the transform function returned a coroutine
                        if inspect.iscoroutine(transformed):
                            # Await the coroutine
                            transformed = await transformed

                        if transformed is not None:
                            yield transformed
                    except Exception as e:
                        if isinstance(e, StreamingError):
                            raise
                        raise StreamingError(
                            f"Error transforming streaming response: {str(e)}"
                        ) from e
                else:
                    yield item  # type: ignore
    except asyncio.TimeoutError:
        raise StreamingError(f"Streaming response timed out after {timeout} seconds")
    except Exception as e:
        if isinstance(e, StreamingError):
            raise
        raise StreamingError(f"Error in streaming response: {str(e)}") from e


async def _stream_with_timeout(
    source_generator: AsyncGenerator[Any, None],
    transform_func: Optional[Callable[[Any], T]],
    timeout: float,
) -> AsyncGenerator[T, None]:
    """Helper to implement streaming with timeout."""
    try:
        while True:
            try:
                # Get next item with timeout
                get_next = source_generator.__anext__()
                item = await asyncio.wait_for(get_next, timeout)

                if transform_func:
                    try:
                        # Call the transform function
                        transformed = transform_func(item)

                        # Check if the transform function returned a coroutine
                        if inspect.iscoroutine(transformed):
                            # Await the coroutine
                            transformed = await transformed

                        if transformed is not None:
                            yield transformed
                    except Exception as e:
                        if isinstance(e, StreamingError):
                            raise
                        raise StreamingError(
                            f"Error transforming streaming response: {str(e)}"
                        ) from e
                else:
                    yield item  # type: ignore
            except StopAsyncIteration:
                break
    except asyncio.TimeoutError:
        raise StreamingError(f"Streaming response timed out after {timeout} seconds")


async def json_stream_generator(
    source_generator: AsyncGenerator[str, None],
    data_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> AsyncGenerator[Any, None]:
    """
    Create a JSON stream from a source generator of JSON strings.

    Args:
        source_generator: The source async generator yielding JSON strings
        data_key: Optional key to extract from each JSON object
        timeout: Optional timeout for each item

    Yields:
        Parsed JSON objects or extracted values from the source generator

    Raises:
        StreamingError: If an error occurs during streaming
    """

    async def transform_json(text: str) -> Optional[Any]:
        if not text.strip():
            return None

        try:
            data = json.loads(text)
            if data_key and isinstance(data, dict):
                # Return None if data_key doesn't exist in the dict
                if data_key not in data:
                    return None
                return data.get(data_key)
            return data
        except json.JSONDecodeError as e:
            # Create a new StreamingError with the JSONDecodeError as cause
            error = StreamingError(f"Invalid JSON in streaming response: {text}")
            error.__cause__ = e
            raise error

    try:
        # Get the generator from stream_generator
        generator = stream_generator(
            source_generator, transform_func=transform_json, timeout=timeout
        )

        # If it's a coroutine, await it to get the actual generator
        if inspect.iscoroutine(generator):
            generator = await generator

        # Iterate through the generator
        async for item in generator:
            if item is not None:
                yield item
    except Exception as e:
        if isinstance(e, StreamingError):
            raise
        raise StreamingError(f"Error in JSON streaming: {str(e)}") from e


async def line_stream_generator(
    source_generator: AsyncGenerator[Union[str, bytes], None],
    prefix: Optional[str] = None,
    timeout: Optional[float] = None,
    transform_func: Optional[Callable[[str], str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Create a line stream from a source generator, optionally filtering by prefix.

    Args:
        source_generator: The source async generator yielding strings or bytes
        prefix: Optional prefix to filter lines (only lines starting with this prefix are yielded)
        timeout: Optional timeout for each item
        transform_func: Optional function to transform each line after processing

    Yields:
        Lines from the source generator, with the prefix removed if specified

    Raises:
        StreamingError: If an error occurs during streaming
    """

    async def process_line(line: Union[str, bytes]) -> Optional[str]:
        try:
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError as e:
                    error = StreamingError("Error decoding bytes in streaming response")
                    error.__cause__ = e
                    raise error

            line = line.rstrip("\r\n")
            if not line.strip():  # Check if the line is empty or contains only whitespace
                return None

            if prefix:
                if line.startswith(prefix):
                    result = line[len(prefix):]
                    if transform_func and result is not None:
                        return transform_func(result)
                    return result
                return None

            if transform_func and line:
                return transform_func(line)
            return line
        except UnicodeDecodeError as e:
            raise StreamingError(f"Error decoding line in streaming response: {str(e)}") from e
        except Exception as e:
            if isinstance(e, StreamingError):
                raise
            raise StreamingError(f"Error processing line in streaming response: {str(e)}") from e

    try:
        # Get the generator from stream_generator
        generator = stream_generator(
            source_generator, transform_func=process_line, timeout=timeout
        )

        # If it's a coroutine, await it to get the actual generator
        if inspect.iscoroutine(generator):
            generator = await generator

        # Iterate through the generator
        async for item in generator:
            if item is not None:
                yield item
    except Exception as e:
        if isinstance(e, StreamingError):
            raise
        raise StreamingError(f"Error in line streaming: {str(e)}") from e

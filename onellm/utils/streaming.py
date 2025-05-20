#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    This function takes an async generator and applies an optional transformation
    function to each item. It handles both synchronous and asynchronous transform
    functions and can apply an optional timeout to each item retrieval.

    Args:
        source_generator: The source async generator that produces items
        transform_func: Optional function to transform each item (can be sync or async)
        timeout: Optional timeout in seconds for retrieving each item

    Yields:
        Transformed items from the source generator

    Raises:
        StreamingError: If an error occurs during streaming or transformation
    """
    try:
        # Handle case where source_generator is actually a coroutine (happens in tests with mocks)
        # This ensures we're working with an actual generator
        if inspect.iscoroutine(source_generator):
            source_generator = await source_generator

        if timeout is not None:
            # Use the timeout implementation if a timeout is specified
            async for item in _stream_with_timeout(source_generator, transform_func, timeout):
                yield item
        else:
            # Standard implementation without timeout
            async for item in source_generator:
                if transform_func:
                    try:
                        # Call the transform function on the item
                        transformed = transform_func(item)

                        # Check if the transform function returned a coroutine
                        # This allows both sync and async transform functions
                        if inspect.iscoroutine(transformed):
                            # Await the coroutine to get the actual result
                            transformed = await transformed

                        # Only yield non-None values
                        if transformed is not None:
                            yield transformed
                    except Exception as e:
                        # Preserve StreamingError instances
                        if isinstance(e, StreamingError):
                            raise
                        # Wrap other exceptions in StreamingError for consistent error handling
                        raise StreamingError(
                            f"Error transforming streaming response: {str(e)}"
                        ) from e
                else:
                    # If no transform function, yield the item directly
                    yield item  # type: ignore
    except asyncio.TimeoutError:
        # Handle timeout errors with a specific message
        raise StreamingError(f"Streaming response timed out after {timeout} seconds")
    except Exception as e:
        # Preserve StreamingError instances
        if isinstance(e, StreamingError):
            raise
        # Wrap other exceptions in StreamingError for consistent error handling
        raise StreamingError(f"Error in streaming response: {str(e)}") from e


async def _stream_with_timeout(
    source_generator: AsyncGenerator[Any, None],
    transform_func: Optional[Callable[[Any], T]],
    timeout: float,
) -> AsyncGenerator[T, None]:
    """
    Helper function to implement streaming with timeout.

    This function applies a timeout to each item retrieval from the source generator
    and applies the transform function if provided.

    Args:
        source_generator: The source async generator
        transform_func: Optional function to transform each item
        timeout: Timeout in seconds for retrieving each item

    Yields:
        Transformed items from the source generator

    Raises:
        StreamingError: If an error occurs during streaming or transformation
        asyncio.TimeoutError: If retrieving an item times out
    """
    try:
        while True:
            try:
                # Get next item with timeout
                # This wraps the generator's __anext__ call with a timeout
                get_next = source_generator.__anext__()
                item = await asyncio.wait_for(get_next, timeout)

                if transform_func:
                    try:
                        # Apply the transform function to the item
                        transformed = transform_func(item)

                        # Handle async transform functions
                        if inspect.iscoroutine(transformed):
                            # Await the coroutine to get the actual result
                            transformed = await transformed

                        # Only yield non-None values
                        if transformed is not None:
                            yield transformed
                    except Exception as e:
                        # Preserve StreamingError instances
                        if isinstance(e, StreamingError):
                            raise
                        # Wrap other exceptions in StreamingError
                        raise StreamingError(
                            f"Error transforming streaming response: {str(e)}"
                        ) from e
                else:
                    # If no transform function, yield the item directly
                    yield item  # type: ignore
            except StopAsyncIteration:
                # End of generator reached
                break
    except asyncio.TimeoutError:
        # Handle timeout errors with a specific message
        raise StreamingError(f"Streaming response timed out after {timeout} seconds")


async def json_stream_generator(
    source_generator: AsyncGenerator[str, None],
    data_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> AsyncGenerator[Any, None]:
    """
    Create a JSON stream from a source generator of JSON strings.

    This function parses JSON strings from the source generator and optionally
    extracts a specific key from each parsed JSON object.

    Args:
        source_generator: The source async generator yielding JSON strings
        data_key: Optional key to extract from each JSON object
        timeout: Optional timeout in seconds for retrieving each item

    Yields:
        Parsed JSON objects or extracted values from the source generator

    Raises:
        StreamingError: If an error occurs during streaming or JSON parsing
    """

    async def transform_json(text: str) -> Optional[Any]:
        """
        Inner function to parse JSON and extract data.

        Args:
            text: JSON string to parse

        Returns:
            Parsed JSON object or extracted value, None for empty strings

        Raises:
            StreamingError: If JSON parsing fails
        """
        # Skip empty strings
        if not text.strip():
            return None

        try:
            # Parse the JSON string
            data = json.loads(text)
            if data_key and isinstance(data, dict):
                # Return None if data_key doesn't exist in the dict
                if data_key not in data:
                    return None
                # Extract the specified key
                return data.get(data_key)
            # Return the full parsed object if no key specified
            return data
        except json.JSONDecodeError as e:
            # Create a new StreamingError with the JSONDecodeError as cause
            error = StreamingError(f"Invalid JSON in streaming response: {text}")
            error.__cause__ = e
            raise error

    try:
        # Get the generator from stream_generator with the JSON transform function
        generator = stream_generator(
            source_generator, transform_func=transform_json, timeout=timeout
        )

        # If it's a coroutine, await it to get the actual generator
        if inspect.iscoroutine(generator):
            generator = await generator

        # Iterate through the generator and yield non-None items
        async for item in generator:
            if item is not None:
                yield item
    except Exception as e:
        # Preserve StreamingError instances
        if isinstance(e, StreamingError):
            raise
        # Wrap other exceptions in StreamingError
        raise StreamingError(f"Error in JSON streaming: {str(e)}") from e


async def line_stream_generator(
    source_generator: AsyncGenerator[Union[str, bytes], None],
    prefix: Optional[str] = None,
    timeout: Optional[float] = None,
    transform_func: Optional[Callable[[str], str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Create a line stream from a source generator, optionally filtering by prefix.

    This function processes lines from the source generator, optionally filtering
    by a prefix and applying a transformation function.

    Args:
        source_generator: The source async generator yielding strings or bytes
        prefix: Optional prefix to filter lines (only lines starting with this prefix are yielded)
        timeout: Optional timeout in seconds for retrieving each item
        transform_func: Optional function to transform each line after processing

    Yields:
        Lines from the source generator, with the prefix removed if specified

    Raises:
        StreamingError: If an error occurs during streaming or line processing
    """

    async def process_line(line: Union[str, bytes]) -> Optional[str]:
        """
        Inner function to process each line from the source generator.

        Args:
            line: Line to process (string or bytes)

        Returns:
            Processed line or None if line should be skipped

        Raises:
            StreamingError: If line processing fails
        """
        try:
            # Convert bytes to string if needed
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError as e:
                    error = StreamingError("Error decoding bytes in streaming response")
                    error.__cause__ = e
                    raise error

            # Remove trailing newlines
            line = line.rstrip("\r\n")
            # Skip empty lines
            if not line.strip():  # Check if the line is empty or contains only whitespace
                return None

            if prefix:
                # If prefix is specified, only process lines starting with that prefix
                if line.startswith(prefix):
                    # Remove the prefix from the line
                    result = line[len(prefix):]
                    # Apply transform function if provided
                    if transform_func and result is not None:
                        return transform_func(result)
                    return result
                # Skip lines that don't start with the prefix
                return None

            # Apply transform function if provided
            if transform_func and line:
                return transform_func(line)
            # Return the line as is
            return line
        except UnicodeDecodeError as e:
            # Handle decoding errors
            raise StreamingError(f"Error decoding line in streaming response: {str(e)}") from e
        except Exception as e:
            # Preserve StreamingError instances
            if isinstance(e, StreamingError):
                raise
            # Wrap other exceptions in StreamingError
            raise StreamingError(f"Error processing line in streaming response: {str(e)}") from e

    try:
        # Get the generator from stream_generator with the line processing function
        generator = stream_generator(
            source_generator, transform_func=process_line, timeout=timeout
        )

        # If it's a coroutine, await it to get the actual generator
        if inspect.iscoroutine(generator):
            generator = await generator

        # Iterate through the generator and yield non-None items
        async for item in generator:
            if item is not None:
                yield item
    except Exception as e:
        # Preserve StreamingError instances
        if isinstance(e, StreamingError):
            raise
        # Wrap other exceptions in StreamingError
        raise StreamingError(f"Error in line streaming: {str(e)}") from e

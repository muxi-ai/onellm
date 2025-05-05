"""
Utilities for handling streaming responses from LLM providers.

This module provides utilities for working with streaming responses,
including converting between different streaming formats and handling errors.
"""

import asyncio
import json
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
    timeout: Optional[float] = None
) -> AsyncGenerator[T, None]:
    """
    Create a transformed stream from a source generator.

    Args:
        source_generator: The source async generator
        transform_func: Optional function to transform each item
        timeout: Optional timeout for each item

    Yields:
        Transformed items from the source generator

    Raises:
        StreamingError: If an error occurs during streaming
    """
    try:
        async for item in source_generator:
            if transform_func:
                try:
                    transformed = transform_func(item)
                    if transformed is not None:
                        yield transformed
                except Exception as e:
                    raise StreamingError(
                        f"Error transforming streaming response: {str(e)}"
                    ) from e
            else:
                yield item  # type: ignore
    except asyncio.TimeoutError:
        raise StreamingError(
            f"Streaming response timed out after {timeout} seconds"
        )
    except Exception as e:
        if isinstance(e, StreamingError):
            raise
        raise StreamingError(f"Error in streaming response: {str(e)}") from e


async def json_stream_generator(
    source_generator: AsyncGenerator[str, None],
    data_key: Optional[str] = None,
    timeout: Optional[float] = None
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
                return data.get(data_key)
            return data
        except json.JSONDecodeError as e:
            raise StreamingError(f"Invalid JSON in streaming response: {text}") from e

    async for item in stream_generator(
        source_generator,
        transform_func=transform_json,
        timeout=timeout
    ):
        yield item


async def line_stream_generator(
    source_generator: AsyncGenerator[Union[str, bytes], None],
    prefix: Optional[str] = None,
    timeout: Optional[float] = None
) -> AsyncGenerator[str, None]:
    """
    Create a line stream from a source generator, optionally filtering by prefix.

    Args:
        source_generator: The source async generator yielding strings or bytes
        prefix: Optional prefix to filter lines (only lines starting with this prefix are yielded)
        timeout: Optional timeout for each item

    Yields:
        Lines from the source generator, with the prefix removed if specified

    Raises:
        StreamingError: If an error occurs during streaming
    """
    async def process_line(line: Union[str, bytes]) -> Optional[str]:
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

    async for item in stream_generator(
        source_generator,
        transform_func=process_line,
        timeout=timeout
    ):
        if item is not None:
            yield item

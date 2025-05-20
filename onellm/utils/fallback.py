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
Fallback utilities for OneLLM.

This module provides utilities for model fallback functionality.
When a primary model or provider fails, these utilities help gracefully
fall back to alternative models or providers to maintain service reliability.
"""

from typing import Callable, List, Optional, Type, TypeVar
import inspect

from ..errors import (
    ServiceUnavailableError,
    TimeoutError,
    BadGatewayError,
    RateLimitError,
)

# Define a generic type for the return value
# This allows the fallback mechanism to work with any return type
T = TypeVar("T")


class FallbackConfig:
    """
    Configuration for fallback behavior.

    This class defines how fallbacks should be handled when errors occur,
    including which errors should trigger fallbacks, how many fallbacks to attempt,
    and optional logging and callback functionality.
    """

    def __init__(
        self,
        retriable_errors: Optional[List[Type[Exception]]] = None,
        max_fallbacks: Optional[int] = None,
        log_fallbacks: bool = True,
        fallback_callback: Optional[Callable] = None,
    ):
        """
        Initialize fallback configuration.

        Args:
            retriable_errors: Error types that should trigger fallbacks. If None,
                              defaults to common network and rate limit errors.
            max_fallbacks: Maximum number of fallbacks to try before giving up.
                          If None, will try all available fallbacks.
            log_fallbacks: Whether to log fallback attempts for monitoring and debugging.
            fallback_callback: Optional callback function when fallbacks are used.
                              Can be used for metrics collection or notifications.
        """
        # Default to common API errors if no specific errors are provided
        self.retriable_errors = retriable_errors or [
            ServiceUnavailableError,
            TimeoutError,
            BadGatewayError,
            RateLimitError,
        ]
        self.max_fallbacks = max_fallbacks
        self.log_fallbacks = log_fallbacks
        self.fallback_callback = fallback_callback


async def maybe_await(result):
    """
    Helper to await a result if it's awaitable, otherwise return it directly.

    This utility function allows the fallback mechanism to work with both
    synchronous and asynchronous functions by handling the awaiting logic.

    Args:
        result: The result to potentially await, could be a coroutine or regular value

    Returns:
        The awaited result if it was awaitable, or the original result otherwise
    """
    # Check if the result is a coroutine or other awaitable object
    if inspect.isawaitable(result):
        # If it is awaitable, await it and return the result
        return await result
    # Otherwise, return the result directly
    return result

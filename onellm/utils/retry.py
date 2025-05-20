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
Retry mechanism for handling transient errors.

This module provides utilities for retrying operations that may fail
due to transient errors, with configurable backoff strategies.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type, TypeVar

from ..errors import (
    RateLimitError,
    ServiceUnavailableError,
    BadGatewayError,
    TimeoutError,
)

# Type variable for the return type of the retried function
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for the retry mechanism."""

    max_retries: int = 3  # Maximum number of retry attempts
    initial_backoff: float = 0.5  # Initial backoff time in seconds
    max_backoff: float = 60.0  # Maximum backoff time in seconds
    backoff_multiplier: float = 2.0  # Factor by which backoff increases with each retry
    jitter: bool = True  # Whether to add randomness to backoff times
    retryable_errors: Optional[List[Type[Exception]]] = None  # Exceptions that should trigger retry

    def __post_init__(self):
        """
        Initialize default retryable errors if none are provided.

        This method runs after the dataclass is initialized and sets up
        the default list of exceptions that should trigger a retry.
        """
        if self.retryable_errors is None:
            # Default set of errors that are considered transient and worth retrying
            self.retryable_errors = [
                RateLimitError,  # Server rate limit exceeded
                ServiceUnavailableError,  # Service temporarily unavailable
                BadGatewayError,  # Bad gateway response
                TimeoutError,  # Request timed out
                ConnectionError,  # Connection issues
                asyncio.TimeoutError,  # Async operation timed out
            ]


def _should_retry(error: Exception, config: RetryConfig) -> bool:
    """
    Determine if a retry should be attempted based on the error.

    This function checks if the given error is of a type that should trigger
    a retry according to the configuration.

    Args:
        error: The exception that was raised
        config: Retry configuration

    Returns:
        True if the error is retryable, False otherwise
    """
    if config.retryable_errors:
        # Check if the error is an instance of any of the retryable error types
        return any(isinstance(error, err_type) for err_type in config.retryable_errors)
    return False


def _calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate the backoff time for a retry attempt.

    This implements an exponential backoff strategy with optional jitter.
    The backoff time increases exponentially with each retry attempt but
    is capped at a maximum value.

    Args:
        attempt: The current attempt number (1-based)
        config: Retry configuration

    Returns:
        Backoff time in seconds
    """
    # Calculate exponential backoff: initial_backoff * (multiplier ^ (attempt-1))
    # but cap it at max_backoff
    backoff = min(
        config.max_backoff,
        config.initial_backoff * (config.backoff_multiplier ** (attempt - 1)),
    )

    if config.jitter:
        # Add jitter to prevent thundering herd problem
        # Multiply by a random factor between 0.5 and 1.5
        backoff = backoff * (0.5 + random.random())

    return backoff


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any
) -> Any:
    """
    Retry an async function with exponential backoff.

    This function will call the provided async function and retry it
    if it fails with a retryable error, using an exponential backoff
    strategy between retries.

    Args:
        func: Async function to call
        *args: Positional arguments to pass to func
        config: Retry configuration
        **kwargs: Keyword arguments to pass to func

    Returns:
        The return value of the function

    Raises:
        Exception: The last exception raised by the function if all retries fail
    """
    # Use default config if none provided
    config = config or RetryConfig()
    last_error = None

    # +2 because attempt starts at 1 and we want max_retries + 1 total attempts
    # (original attempt plus max_retries retry attempts)
    for attempt in range(1, config.max_retries + 2):
        try:
            # Attempt to call the function
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Determine if we should retry or re-raise the error
            if attempt > config.max_retries or not _should_retry(e, config):
                # Re-raise the error if we've exhausted retries or if it's not retryable
                raise

            # Calculate backoff time for this attempt
            backoff = _calculate_backoff(attempt, config)

            # Wait before the next attempt
            await asyncio.sleep(backoff)

    # This should never be reached due to the re-raise above, but keeping for safety
    assert last_error is not None
    raise last_error

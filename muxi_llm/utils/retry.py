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

    max_retries: int = 3
    initial_backoff: float = 0.5  # seconds
    max_backoff: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_errors: Optional[List[Type[Exception]]] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                RateLimitError,
                ServiceUnavailableError,
                BadGatewayError,
                TimeoutError,
                ConnectionError,
                asyncio.TimeoutError,
            ]


def _should_retry(error: Exception, config: RetryConfig) -> bool:
    """
    Determine if a retry should be attempted based on the error.

    Args:
        error: The exception that was raised
        config: Retry configuration

    Returns:
        True if the error is retryable, False otherwise
    """
    if config.retryable_errors:
        return any(isinstance(error, err_type) for err_type in config.retryable_errors)
    return False


def _calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate the backoff time for a retry attempt.

    Args:
        attempt: The current attempt number (1-based)
        config: Retry configuration

    Returns:
        Backoff time in seconds
    """
    backoff = min(
        config.max_backoff,
        config.initial_backoff * (config.backoff_multiplier ** (attempt - 1)),
    )

    if config.jitter:
        # Add jitter to prevent thundering herd problem
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
    config = config or RetryConfig()
    last_error = None

    # +2 because attempt starts at 1 and we want max_retries + 1 total attempts
    for attempt in range(1, config.max_retries + 2):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt > config.max_retries or not _should_retry(e, config):
                # Re-raise the error if we've exhausted retries or if it's not retryable
                raise

            # Calculate backoff time
            backoff = _calculate_backoff(attempt, config)

            # Wait before the next attempt
            await asyncio.sleep(backoff)

    # This should never be reached due to the re-raise above, but keeping for safety
    assert last_error is not None
    raise last_error

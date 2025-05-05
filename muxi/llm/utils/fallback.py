"""
Fallback utilities for muxi-llm.

This module provides utilities for model fallback functionality.
"""

from typing import Callable, List, Optional, Type, TypeVar
import inspect

from ..errors import (
    ServiceUnavailableError, TimeoutError, BadGatewayError, RateLimitError
)

# Define a generic type for the return value
T = TypeVar('T')


class FallbackConfig:
    """Configuration for fallback behavior."""

    def __init__(
        self,
        retriable_errors: Optional[List[Type[Exception]]] = None,
        max_fallbacks: Optional[int] = None,
        log_fallbacks: bool = True,
        fallback_callback: Optional[Callable] = None
    ):
        """
        Initialize fallback configuration.

        Args:
            retriable_errors: Error types that should trigger fallbacks
            max_fallbacks: Maximum number of fallbacks to try
            log_fallbacks: Whether to log fallback attempts
            fallback_callback: Optional callback function when fallbacks are used
        """
        self.retriable_errors = retriable_errors or [
            ServiceUnavailableError,
            TimeoutError,
            BadGatewayError,
            RateLimitError
        ]
        self.max_fallbacks = max_fallbacks
        self.log_fallbacks = log_fallbacks
        self.fallback_callback = fallback_callback


async def maybe_await(result):
    """Helper to await a result if it's awaitable, otherwise return it."""
    if inspect.isawaitable(result):
        return await result
    return result

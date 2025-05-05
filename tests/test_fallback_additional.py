"""
Additional tests for the fallback utilities.

These tests focus on the maybe_await helper function and
custom fallback configurations.
"""

import pytest
from unittest import mock

from muxi_llm.utils.fallback import FallbackConfig, maybe_await
from muxi_llm.errors import (
    ServiceUnavailableError, TimeoutError,
    BadGatewayError, RateLimitError
)


class TestMaybeAwait:
    """Tests for the maybe_await helper function."""

    @pytest.mark.asyncio
    async def test_await_coroutine(self):
        """Test that maybe_await awaits coroutines correctly."""
        async def async_func():
            return "async result"

        # Test with a coroutine
        result = await maybe_await(async_func())
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_normal_value(self):
        """Test that maybe_await returns normal values as-is."""
        # Test with regular values
        assert await maybe_await("regular value") == "regular value"
        assert await maybe_await(123) == 123
        assert await maybe_await(None) is None
        assert await maybe_await([1, 2, 3]) == [1, 2, 3]


class TestFallbackConfig:
    """Tests for the FallbackConfig class."""

    def test_default_init(self):
        """Test default initialization of FallbackConfig."""
        config = FallbackConfig()

        # Check default values
        assert ServiceUnavailableError in config.retriable_errors
        assert TimeoutError in config.retriable_errors
        assert BadGatewayError in config.retriable_errors
        assert RateLimitError in config.retriable_errors
        assert config.max_fallbacks is None  # No limit by default
        assert config.log_fallbacks is True  # Logging enabled by default
        assert config.fallback_callback is None  # No callback by default

    def test_custom_error_types(self):
        """Test initialization with custom error types."""
        # Only retry on RateLimitError
        config = FallbackConfig(retriable_errors=[RateLimitError])

        assert RateLimitError in config.retriable_errors
        assert ServiceUnavailableError not in config.retriable_errors
        assert TimeoutError not in config.retriable_errors
        assert BadGatewayError not in config.retriable_errors
        assert len(config.retriable_errors) == 1

    def test_max_fallbacks(self):
        """Test max_fallbacks setting."""
        config = FallbackConfig(max_fallbacks=2)
        assert config.max_fallbacks == 2

    def test_logging_settings(self):
        """Test log_fallbacks setting."""
        # Disable logging
        config = FallbackConfig(log_fallbacks=False)
        assert config.log_fallbacks is False

    def test_fallback_callback(self):
        """Test fallback_callback setting."""
        # Set a callback function
        callback = mock.Mock()
        config = FallbackConfig(fallback_callback=callback)

        assert config.fallback_callback is callback

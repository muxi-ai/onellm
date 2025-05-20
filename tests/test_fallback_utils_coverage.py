#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for onellm/utils/fallback.py.

This test specifically targets the uncovered lines 73-75 in the maybe_await function
where a non-awaitable result is handled.
"""

import pytest

from onellm.utils.fallback import maybe_await, FallbackConfig


class TestFallbackUtils:
    """Tests for the utility functions in fallback.py."""

    @pytest.mark.asyncio
    async def test_maybe_await_with_non_awaitable(self):
        """Test maybe_await with a non-awaitable value (lines 73-75)."""
        # Test with a string (non-awaitable)
        result = await maybe_await("not awaitable")
        assert result == "not awaitable"

        # Test with an integer (non-awaitable)
        result = await maybe_await(42)
        assert result == 42

        # Test with a list (non-awaitable)
        test_list = [1, 2, 3]
        result = await maybe_await(test_list)
        assert result == test_list

        # Test with a dictionary (non-awaitable)
        test_dict = {"key": "value"}
        result = await maybe_await(test_dict)
        assert result == test_dict

        # Test with None (non-awaitable)
        result = await maybe_await(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_maybe_await_with_awaitable(self):
        """Test maybe_await with an awaitable value (for completeness)."""
        # Create a coroutine
        async def async_func():
            return "awaited result"

        # Test with a coroutine (awaitable)
        coro = async_func()
        result = await maybe_await(coro)
        assert result == "awaited result"

    def test_fallback_config_defaults(self):
        """Test FallbackConfig default values."""
        config = FallbackConfig()
        assert len(config.retriable_errors) == 4
        assert config.max_fallbacks is None
        assert config.log_fallbacks is True
        assert config.fallback_callback is None

    def test_fallback_config_custom_values(self):
        """Test FallbackConfig with custom values."""
        # Define a custom callback
        def custom_callback(provider, error):
            return f"Error in {provider}: {error}"

        # Create config with custom values
        config = FallbackConfig(
            retriable_errors=[ValueError, TypeError],
            max_fallbacks=3,
            log_fallbacks=False,
            fallback_callback=custom_callback
        )

        # Verify values
        assert config.retriable_errors == [ValueError, TypeError]
        assert config.max_fallbacks == 3
        assert config.log_fallbacks is False
        assert config.fallback_callback is custom_callback
        assert config.fallback_callback("test", "error") == "Error in test: error"

"""
Additional tests for the retry utility.

This module tests edge cases and error handling in the retry utility,
focusing on previously untested code paths.
"""

import asyncio
import pytest
from unittest import mock

from muxi_llm.utils.retry import retry_async, RetryConfig, _should_retry, _calculate_backoff
from muxi_llm.errors import RateLimitError, BadGatewayError, TimeoutError


class TestShouldRetryFunction:
    """Tests for the _should_retry function."""

    def test_custom_retryable_error_types(self):
        """Test _should_retry with custom error types."""
        # Create a custom error hierarchy
        class CustomError(Exception):
            pass

        class SpecificError(CustomError):
            pass

        # Standard config has different error types
        standard_config = RetryConfig()

        # Custom config with our error types
        custom_config = RetryConfig(retryable_errors=[CustomError])

        # Test with standard config
        assert _should_retry(CustomError(), standard_config) is False
        assert _should_retry(SpecificError(), standard_config) is False

        # Test with custom config
        assert _should_retry(CustomError(), custom_config) is True
        assert _should_retry(SpecificError(), custom_config) is True  # Subclass should match

    def test_empty_retryable_errors(self):
        """Test _should_retry with empty retryable_errors."""
        # Configure RetryConfig with empty error list
        config = RetryConfig(retryable_errors=[])

        # Nothing should be retryable
        assert _should_retry(Exception(), config) is False
        assert _should_retry(RateLimitError("Test"), config) is False
        assert _should_retry(BadGatewayError("Test"), config) is False

    def test_non_exception_error(self):
        """Test _should_retry with non-Exception objects."""
        # Configure standard RetryConfig
        config = RetryConfig()

        # Test with non-Exception objects (should return False)
        assert _should_retry("not an exception", config) is False
        assert _should_retry(123, config) is False
        assert _should_retry(None, config) is False


class TestCalculateBackoffFunction:
    """Tests for the _calculate_backoff function."""

    def test_initial_backoff(self):
        """Test initial backoff calculation."""
        config = RetryConfig(
            initial_backoff=2.0,
            backoff_multiplier=2.0,
            max_backoff=60.0,
            jitter=False
        )

        # First attempt should use initial_backoff
        assert _calculate_backoff(1, config) == 2.0

    def test_exponential_growth(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_backoff=1.0,
            backoff_multiplier=3.0,  # Use 3 instead of default 2
            max_backoff=100.0,
            jitter=False
        )

        # Check exponential growth
        assert _calculate_backoff(1, config) == 1.0  # initial
        assert _calculate_backoff(2, config) == 3.0  # 1.0 * 3^1
        assert _calculate_backoff(3, config) == 9.0  # 1.0 * 3^2
        assert _calculate_backoff(4, config) == 27.0  # 1.0 * 3^3

    def test_max_backoff_cap(self):
        """Test backoff capping at max_backoff."""
        config = RetryConfig(
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=10.0,  # Cap at 10 seconds
            jitter=False
        )

        # Sequence: 1, 2, 4, 8, 16, 32...
        # But capped at 10
        assert _calculate_backoff(1, config) == 1.0
        assert _calculate_backoff(2, config) == 2.0
        assert _calculate_backoff(3, config) == 4.0
        assert _calculate_backoff(4, config) == 8.0
        assert _calculate_backoff(5, config) == 10.0  # Capped at max_backoff
        assert _calculate_backoff(6, config) == 10.0  # Still capped
        assert _calculate_backoff(10, config) == 10.0  # Still capped

    def test_jitter_effect(self):
        """Test the effect of jitter on backoff calculation."""
        config = RetryConfig(
            initial_backoff=10.0,
            jitter=True
        )

        # With jitter enabled, collect multiple samples
        samples = [_calculate_backoff(1, config) for _ in range(10)]

        # All samples should be different due to jitter
        assert len(set(samples)) > 1

        # All samples should be between 0.5 * initial_backoff and 1.5 * initial_backoff
        assert all(5.0 <= sample <= 15.0 for sample in samples)

        # Test without jitter (all samples should be the same)
        config.jitter = False
        samples_no_jitter = [_calculate_backoff(1, config) for _ in range(10)]
        assert len(set(samples_no_jitter)) == 1
        assert samples_no_jitter[0] == 10.0


class TestRetryAsyncFunction:
    """Tests for the retry_async function."""

    @pytest.mark.asyncio
    async def test_immediate_success(self):
        """Test successful execution without any retries."""
        mock_func = mock.AsyncMock(return_value="success")

        result = await retry_async(mock_func, "arg1", kwarg1="value1")

        # Check result and call count
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        # Function raises ValueError which is not in default retryable_errors
        mock_func = mock.AsyncMock(side_effect=ValueError("Not retryable"))

        with pytest.raises(ValueError) as excinfo:
            await retry_async(mock_func)

        assert "Not retryable" in str(excinfo.value)
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_retries_reached(self):
        """Test behavior when max_retries is reached."""
        # Function always raises RateLimitError (which is retryable)
        mock_func = mock.AsyncMock(side_effect=RateLimitError("Rate limited"))

        # Configure RetryConfig with max_retries=2
        config = RetryConfig(max_retries=2)

        # Mock sleep to avoid actual waiting
        with mock.patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(RateLimitError) as excinfo:
                await retry_async(mock_func, config=config)

            assert "Rate limited" in str(excinfo.value)

            # Should be called initial + max_retries times
            assert mock_func.call_count == 3

            # Sleep should be called max_retries times
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_retryable_error_then_success(self):
        """Test retrying after retryable errors until success."""
        # Function fails twice then succeeds
        mock_func = mock.AsyncMock(side_effect=[
            RateLimitError("Rate limited 1"),
            BadGatewayError("Bad gateway"),
            "success"
        ])

        # Mock sleep to avoid actual waiting
        with mock.patch("asyncio.sleep") as mock_sleep:
            result = await retry_async(mock_func)

            # Check final result
            assert result == "success"

            # Function should be called 3 times total
            assert mock_func.call_count == 3

            # Sleep should be called twice (after first and second failures)
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_error_types(self):
        """Test with a mix of retryable and non-retryable errors."""
        # Create a sequence with retryable then non-retryable error
        mock_func = mock.AsyncMock(side_effect=[
            TimeoutError("Timeout error"),  # Retryable
            ValueError("Not retryable")     # Not retryable
        ])

        # Mock sleep to avoid actual waiting
        with mock.patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(ValueError) as excinfo:
                await retry_async(mock_func)

            assert "Not retryable" in str(excinfo.value)

            # Function should be called twice
            assert mock_func.call_count == 2

            # Sleep should be called once (after first failure)
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retry_logic_with_config(self):
        """Test retry with custom configuration."""
        # Function always fails
        mock_func = mock.AsyncMock(side_effect=ValueError("Always fails"))

        # Configure custom RetryConfig that retries ValueError
        config = RetryConfig(
            max_retries=1,
            initial_backoff=5.0,
            retryable_errors=[ValueError]
        )

        # Mock sleep to avoid actual waiting and check backoff time
        with mock.patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                await retry_async(mock_func, config=config)

            # Function should be called twice (original + 1 retry)
            assert mock_func.call_count == 2

            # Sleep should be called once with the configured backoff
            mock_sleep.assert_called_once()
            # Check if sleep was called with a value close to initial_backoff
            # (might not be exact due to jitter)
            sleep_time = mock_sleep.call_args[0][0]
            assert 2.5 <= sleep_time <= 7.5  # 0.5-1.5x range with jitter

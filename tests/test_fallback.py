"""
Tests for the fallback mechanism in muxi-llm.
"""

import pytest
import mock

from muxi_llm.utils.fallback import FallbackConfig
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.errors import (
    APIError, AuthenticationError, RateLimitError, FallbackExhaustionError
)

# Mock response for testing
mock_chat_completion_response = {
    "id": "test-id",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-test",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a test response"
            },
            "finish_reason": "stop",
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}


class MockProvider:
    """Mock provider for testing fallbacks."""

    def __init__(self, should_fail=False, error_type=None):
        self.should_fail = should_fail
        self.error_type = error_type
        self.call_count = 0

    async def create_chat_completion(self, messages, model, **kwargs):
        self.call_count += 1
        if self.should_fail:
            if self.error_type == "rate_limit":
                raise RateLimitError("Rate limit exceeded")
            elif self.error_type == "auth":
                raise AuthenticationError("Invalid API key")
            else:
                raise APIError("Generic API error")
        return mock_chat_completion_response


class TestFallbackMechanism:
    """Tests for the fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_provider_proxy(self):
        """Test the FallbackProviderProxy class."""
        # Create mock providers
        primary_provider = MockProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail on the primary provider and succeed on the fallback
            result = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )

            # Verify the primary provider was called
            assert primary_provider.call_count == 1

            # Verify the fallback provider was called
            assert fallback_provider.call_count == 1

            # Verify we got a result
            assert result == mock_chat_completion_response

    @pytest.mark.asyncio
    async def test_fallback_exhaustion(self):
        """Test fallback exhaustion when all providers fail."""
        # Create mock providers that all fail
        primary_provider = MockProvider(should_fail=True, error_type="rate_limit")
        fallback_provider = MockProvider(should_fail=True, error_type="rate_limit")

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail on all providers
            with pytest.raises(FallbackExhaustionError) as excinfo:
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify the error message contains useful information
            assert "All models failed" in str(excinfo.value)
            assert "openai/gpt-4" in str(excinfo.value)
            assert "anthropic/claude-3" in str(excinfo.value)

            # Verify both providers were called
            assert primary_provider.call_count == 1
            assert fallback_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retriable_error(self):
        """Test that non-retriable errors are raised immediately."""
        # Create mock providers
        primary_provider = MockProvider(should_fail=True, error_type="auth")
        fallback_provider = MockProvider(should_fail=False)

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                primary_provider if provider_name == "openai" else fallback_provider
            )

            # Create a fallback provider proxy with RateLimitError as the only retriable error
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3"],
                FallbackConfig(retriable_errors=[RateLimitError])
            )

            # Call a method that should fail with a non-retriable error
            with pytest.raises(AuthenticationError):
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify only the primary provider was called
            assert primary_provider.call_count == 1
            assert fallback_provider.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_config_max_fallbacks(self):
        """Test max_fallbacks configuration."""
        # Create mock providers
        providers = [
            MockProvider(should_fail=True, error_type="rate_limit"),  # Primary
            MockProvider(should_fail=True, error_type="rate_limit"),  # Fallback 1
            MockProvider(should_fail=False)                           # Fallback 2
        ]

        # Patch get_provider to return our mock providers
        with mock.patch("muxi_llm.providers.fallback.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = lambda provider_name: (
                providers[0] if provider_name == "openai" else (
                    providers[1] if provider_name == "anthropic" else providers[2]
                )
            )

            # Create a fallback provider proxy with max_fallbacks=1
            # This should try the primary and only the first fallback
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"],
                FallbackConfig(retriable_errors=[RateLimitError], max_fallbacks=1)
            )

            # This should fail because we only try 2 models (primary + 1 fallback)
            with pytest.raises(FallbackExhaustionError):
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-4"
                )

            # Verify only the primary and first fallback were called
            assert providers[0].call_count == 1
            assert providers[1].call_count == 1
            assert providers[2].call_count == 0

            # Reset call counts
            for p in providers:
                p.call_count = 0

            # Now create a proxy with max_fallbacks=2
            proxy = FallbackProviderProxy(
                ["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"],
                FallbackConfig(retriable_errors=[RateLimitError], max_fallbacks=2)
            )

            # This should succeed because we try all 3 models
            result = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )

            # Verify all providers were called until we found a working one
            assert providers[0].call_count == 1
            assert providers[1].call_count == 1
            assert providers[2].call_count == 1
            assert result == mock_chat_completion_response

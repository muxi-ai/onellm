#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the get_provider_with_fallbacks function in providers/base.py.

These tests focus on covering the fallback handling logic.
"""

from unittest import mock

from muxi_llm.providers.base import (
    get_provider_with_fallbacks
)
from muxi_llm.utils.fallback import FallbackConfig


class TestProviderWithFallbacks:
    """Tests for the get_provider_with_fallbacks function."""

    def test_with_fallback_models(self):
        """Test the get_provider_with_fallbacks function with fallback models."""
        # Mock the FallbackProviderProxy class that's imported within the function
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_proxy, \
             mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:

            # Configure mock_parse to return different values for different inputs
            def parse_side_effect(model_str):
                if model_str == "openai/gpt-4":
                    return ("openai", "gpt-4")
                elif model_str == "anthropic/claude-v2":
                    return ("anthropic", "claude-v2")
                else:
                    return ("unknown", model_str.split("/")[-1])

            mock_parse.side_effect = parse_side_effect

            # Create a mock proxy instance
            mock_proxy_instance = mock.MagicMock()
            mock_proxy.return_value = mock_proxy_instance

            # Call the function with fallback models
            fallback_models = ["anthropic/claude-v2"]
            provider, model = get_provider_with_fallbacks(
                "openai/gpt-4",
                fallback_models=fallback_models
            )

            # Verify that FallbackProviderProxy was created with the correct models
            mock_proxy.assert_called_once_with(
                ["openai/gpt-4", "anthropic/claude-v2"],
                None  # fallback_config is None
            )

            # Verify the results
            assert provider == mock_proxy_instance
            assert model == "gpt-4"

    def test_with_fallbacks_and_config(self):
        """Test the get_provider_with_fallbacks function with fallbacks and config."""
        # Mock the FallbackProviderProxy class that's imported within the function
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_proxy, \
             mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:

            # Configure mock_parse to return a standard value
            mock_parse.return_value = ("openai", "gpt-4")

            # Create a mock proxy instance
            mock_proxy_instance = mock.MagicMock()
            mock_proxy.return_value = mock_proxy_instance

            # Create a fallback config
            fallback_config = FallbackConfig(
                max_fallbacks=3,
                log_fallbacks=True
            )

            # Call the function with fallback models and config
            fallback_models = ["anthropic/claude-v2", "mistral/mistral-7b"]
            provider, model = get_provider_with_fallbacks(
                "openai/gpt-4",
                fallback_models=fallback_models,
                fallback_config=fallback_config
            )

            # Verify that FallbackProviderProxy was created with the correct parameters
            mock_proxy.assert_called_once_with(
                ["openai/gpt-4", "anthropic/claude-v2", "mistral/mistral-7b"],
                fallback_config
            )

            # Verify the results
            assert provider == mock_proxy_instance
            assert model == "gpt-4"

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Maximum fallbacks coverage test for fallback.py in muxi-llm.

This test specifically targets line 178 in FallbackProviderProxy (max_fallbacks config).
"""

import pytest
from unittest import mock

from muxi_llm.errors import APIError, FallbackExhaustionError
from muxi_llm.providers.fallback import FallbackProviderProxy, parse_model_name
from muxi_llm.providers.base import Provider
from muxi_llm.utils.fallback import FallbackConfig


class SimpleFailingProvider(Provider):
    """Simple provider that always fails with API errors."""

    def __init__(self, name="test"):
        self.name = name
        self.calls = []

    async def create_chat_completion(self, messages, model, **kwargs):
        self.calls.append(("create_chat_completion", model))
        raise APIError(f"{self.name} error")

    async def create_completion(self, prompt, model, **kwargs):
        self.calls.append(("create_completion", model))
        raise APIError(f"{self.name} error")

    async def create_embedding(self, input, model, **kwargs):
        self.calls.append(("create_embedding", model))
        raise APIError(f"{self.name} error")

    async def upload_file(self, file, purpose, **kwargs):
        self.calls.append(("upload_file", None))
        raise APIError(f"{self.name} error")

    async def download_file(self, file_id, **kwargs):
        self.calls.append(("download_file", None))
        raise APIError(f"{self.name} error")

    async def create_speech(self, input, model, **kwargs):
        self.calls.append(("create_speech", model))
        raise APIError(f"{self.name} error")

    async def create_image(self, prompt, model, **kwargs):
        self.calls.append(("create_image", model))
        raise APIError(f"{self.name} error")


class TestFallbackMaxFallbacks:
    """Tests specifically targeting the max_fallbacks configuration."""

    def setup_method(self):
        """Set up test environment."""
        # Patch the functions used by FallbackProviderProxy
        self.mock_get_provider = mock.patch("muxi_llm.providers.fallback.get_provider").start()
        # Use the real parse_model_name function
        self.real_parse_model_name = parse_model_name

    def teardown_method(self):
        """Clean up test environment."""
        mock.patch.stopall()

    @pytest.mark.asyncio
    async def test_max_fallbacks_exhaustion(self):
        """Test that max_fallbacks limits the number of providers tried.

        This targets line 178 - the max_fallbacks logic in _try_with_fallbacks.
        """
        # Create three failing providers
        provider1 = SimpleFailingProvider("provider1")
        provider2 = SimpleFailingProvider("provider2")
        provider3 = SimpleFailingProvider("provider3")

        # Configure get_provider to return our providers
        def get_provider_side_effect(provider_name):
            if provider_name == "provider1":
                return provider1
            elif provider_name == "provider2":
                return provider2
            elif provider_name == "provider3":
                return provider3
            raise ValueError(f"Unknown provider: {provider_name}")

        self.mock_get_provider.side_effect = get_provider_side_effect

        # Create fallback proxy with max_fallbacks=1 (so it should only try 2 providers: primary + 1 fallback)
        proxy = FallbackProviderProxy(
            ["provider1/model1", "provider2/model2", "provider3/model3"],
            FallbackConfig(
                retriable_errors=[APIError],
                max_fallbacks=1  # Key setting - only try one fallback
            )
        )

        # Attempt should raise FallbackExhaustionError after trying only 2 providers
        with pytest.raises(FallbackExhaustionError) as excinfo:
            await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

        # Verify the exception details and models tried
        assert excinfo.value.primary_model == "provider1/model1"
        assert "provider1/model1" in excinfo.value.models_tried
        assert "provider2/model2" in excinfo.value.models_tried
        # provider3 should NOT be tried because of max_fallbacks=1
        assert "provider3/model3" not in excinfo.value.models_tried

        # Verify provider call counts - only first two should be called
        assert len(provider1.calls) == 1
        assert len(provider2.calls) == 1
        assert len(provider3.calls) == 0

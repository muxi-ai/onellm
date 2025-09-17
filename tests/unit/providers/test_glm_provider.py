#!/usr/bin/env python3
"""Unit tests for the GLM provider implementation."""

import pytest
from unittest.mock import patch

from onellm.errors import AuthenticationError
from onellm.providers.glm import GLMProvider


def _mock_config(api_key: str | None = "test-key") -> dict[str, object]:
    return {
        "api_key": api_key,
        "api_base": "https://api.z.ai/api/paas/v4",
        "timeout": 60,
        "max_retries": 3,
    }


def test_glm_initialization_success():
    """Provider should initialize when an API key is available."""
    with patch(
        "onellm.providers.openai_compatible.get_provider_config", return_value=_mock_config()
    ):
        provider = GLMProvider()

    assert provider.provider_name == "glm"
    assert provider.api_base == "https://api.z.ai/api/paas/v4"
    assert provider.api_key == "test-key"
    assert provider.json_mode_support is True
    assert provider.streaming_support is True


def test_glm_initialization_missing_key():
    """Provider should raise when no API key configured."""
    with patch(
        "onellm.providers.openai_compatible.get_provider_config",
        return_value=_mock_config(api_key=None),
    ):
        with pytest.raises(AuthenticationError):
            GLMProvider()

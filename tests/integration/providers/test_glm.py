#!/usr/bin/env python3
"""Integration test for GLM provider connectivity."""

import os

import pytest
import pytest_asyncio  # noqa: F401

pytestmark = pytest.mark.skipif(
    not os.environ.get("GLM_API_KEY") and not os.environ.get("ZAI_API_KEY"),
    reason="GLM_API_KEY or ZAI_API_KEY not set",
)


@pytest.mark.asyncio
async def test_glm_chat_completion():
    from onellm.providers.glm import GLMProvider

    provider = GLMProvider()
    resp = await provider.create_chat_completion(
        model="glm-4-plus",
        messages=[{"role": "user", "content": "Quick connectivity test."}],
        max_tokens=16,
    )
    assert resp.choices[0].message["content"]
    assert resp.usage

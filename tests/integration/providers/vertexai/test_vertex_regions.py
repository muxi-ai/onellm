#!/usr/bin/env python3
"""Test Vertex AI provider with different regions."""

import os

import pytest

from .conftest import skip_no_creds

pytestmark = skip_no_creds


@pytest.mark.parametrize("region", ["us-central1", "europe-west4"])
def test_vertex_region(region):
    """Test that Vertex AI responds in the given region."""
    os.environ["VERTEX_AI_LOCATION"] = region
    try:
        from onellm import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="vertexai/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        assert response is not None
        assert response.choices[0].message.get("content")
    finally:
        os.environ.pop("VERTEX_AI_LOCATION", None)

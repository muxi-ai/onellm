#!/usr/bin/env python3
"""Test Vertex AI provider implementation."""

from .conftest import skip_no_creds

pytestmark = skip_no_creds


def test_vertex_basic():
    """Test Vertex AI provider with a common Gemini model."""
    from onellm import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-flash",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    assert response is not None
    assert response.choices[0].message.get("content")

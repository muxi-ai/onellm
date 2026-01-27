#!/usr/bin/env python3
"""Test Vertex AI with global location."""

import os

from .conftest import skip_no_creds

pytestmark = skip_no_creds


def test_vertex_global_location():
    """Test that Vertex AI works with 'global' location."""
    os.environ["VERTEX_AI_LOCATION"] = "global"
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

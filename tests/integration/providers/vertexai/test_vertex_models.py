#!/usr/bin/env python3
"""Test various Vertex AI model naming conventions."""

import pytest

from .conftest import skip_no_creds

pytestmark = skip_no_creds

_MODELS = [
    "vertexai/gemini-1.5-flash",
    "vertexai/gemini-1.5-pro",
    "vertexai/gemini-pro",
]


@pytest.mark.parametrize("model", _MODELS)
def test_vertex_model(model):
    """Test that a Vertex AI model responds to a basic prompt."""
    from onellm import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=5,
    )
    assert response is not None
    assert response.choices[0].message.get("content")

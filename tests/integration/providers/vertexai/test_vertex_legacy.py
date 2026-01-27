#!/usr/bin/env python3
"""Integration test for Vertex AI with legacy models and different endpoint formats."""

import json

import pytest

from .conftest import CREDS_PATH, skip_no_creds

pytestmark = skip_no_creds


@pytest.fixture(scope="module")
def vertex_env():
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    with open(CREDS_PATH) as f:
        info = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return {
        "project_id": info["project_id"],
        "location": "us-central1",
        "headers": {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
    }


_GENERATE_CONTENT_PAYLOAD = {
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    "generationConfig": {"maxOutputTokens": 10},
}


def test_gemini_flash(vertex_env):
    import requests

    url = (
        f"https://{vertex_env['location']}-aiplatform.googleapis.com/v1/"
        f"projects/{vertex_env['project_id']}/locations/{vertex_env['location']}/"
        f"publishers/google/models/gemini-1.5-flash:generateContent"
    )
    resp = requests.post(url, json=_GENERATE_CONTENT_PAYLOAD, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert "candidates" in resp.json()


def test_gemini_pro(vertex_env):
    import requests

    url = (
        f"https://{vertex_env['location']}-aiplatform.googleapis.com/v1/"
        f"projects/{vertex_env['project_id']}/locations/{vertex_env['location']}/"
        f"publishers/google/models/gemini-pro:generateContent"
    )
    resp = requests.post(url, json=_GENERATE_CONTENT_PAYLOAD, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert "candidates" in resp.json()


def test_text_bison_predict(vertex_env):
    import requests

    url = (
        f"https://{vertex_env['location']}-aiplatform.googleapis.com/v1/"
        f"projects/{vertex_env['project_id']}/locations/{vertex_env['location']}/"
        f"publishers/google/models/text-bison:predict"
    )
    data = {
        "instances": [{"prompt": "Hello, how are you?"}],
        "parameters": {"maxOutputTokens": 10},
    }
    resp = requests.post(url, json=data, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert "predictions" in resp.json()


def test_text_embedding_gecko(vertex_env):
    import requests

    url = (
        f"https://{vertex_env['location']}-aiplatform.googleapis.com/v1/"
        f"projects/{vertex_env['project_id']}/locations/{vertex_env['location']}/"
        f"publishers/google/models/textembedding-gecko:predict"
    )
    data = {"instances": [{"content": "Hello world"}]}
    resp = requests.post(url, json=data, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert "predictions" in resp.json()

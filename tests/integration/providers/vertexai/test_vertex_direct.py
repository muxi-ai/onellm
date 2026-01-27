#!/usr/bin/env python3
"""Integration test for Vertex AI direct API access."""

import json
import os

import pytest

_CREDS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "tests/artifacts/vertexai.json")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_CREDS_PATH),
    reason=f"Vertex AI credentials not found at {_CREDS_PATH}",
)


@pytest.fixture(scope="module")
def vertex_env():
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    with open(_CREDS_PATH) as f:
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


def test_vertex_api_enabled(vertex_env):
    import requests

    url = (
        f"https://serviceusage.googleapis.com/v1/projects/"
        f"{vertex_env['project_id']}/services/aiplatform.googleapis.com"
    )
    resp = requests.get(url, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert resp.json().get("state") == "ENABLED"


def test_gemini_generate_content(vertex_env):
    import requests

    url = (
        f"https://{vertex_env['location']}-aiplatform.googleapis.com/v1/"
        f"projects/{vertex_env['project_id']}/locations/{vertex_env['location']}/"
        f"publishers/google/models/gemini-1.5-flash:generateContent"
    )
    data = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {"maxOutputTokens": 10},
    }
    resp = requests.post(url, json=data, headers=vertex_env["headers"])
    assert resp.status_code == 200
    assert "candidates" in resp.json()

"""Shared fixtures and guards for Vertex AI integration tests."""

import json
import os

import pytest

CREDS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "tests/artifacts/vertexai.json")


def creds_valid() -> bool:
    """Return True only when the credentials file exists with non-empty required fields."""
    if not os.path.isfile(CREDS_PATH):
        return False
    try:
        with open(CREDS_PATH) as f:
            info = json.load(f)
        return bool(info.get("project_id") and info.get("private_key"))
    except (json.JSONDecodeError, OSError):
        return False


skip_no_creds = pytest.mark.skipif(
    not creds_valid(),
    reason=f"Valid Vertex AI credentials not found at {CREDS_PATH}",
)


@pytest.fixture(autouse=True)
def _set_vertex_creds():
    """Set GOOGLE_APPLICATION_CREDENTIALS for the duration of each test."""
    old = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDS_PATH
    yield
    if old is None:
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old

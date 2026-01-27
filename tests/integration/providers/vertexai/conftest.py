"""Shared fixtures and guards for Vertex AI integration tests."""

import json
import os

import pytest

_DEFAULT_CREDS = "tests/artifacts/vertexai.json"


def _get_creds_path() -> str:
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", _DEFAULT_CREDS)


def creds_valid() -> bool:
    """Return True only when the credentials file exists with non-empty required fields."""
    path = _get_creds_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            info = json.load(f)
        return bool(info.get("project_id") and info.get("private_key"))
    except (json.JSONDecodeError, OSError):
        return False


# Re-read at import time for the initial skip decision
CREDS_PATH = _get_creds_path()

skip_no_creds = pytest.mark.skipif(
    not creds_valid(),
    reason=f"Valid Vertex AI credentials not found at {CREDS_PATH}",
)


@pytest.fixture(autouse=True)
def _set_vertex_creds():
    """Set GOOGLE_APPLICATION_CREDENTIALS for the duration of each test."""
    path = _get_creds_path()
    old = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    yield
    if old is None:
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old

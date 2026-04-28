"""Test helpers for the httpx-based provider HTTP layer.

This module replaces the previous per-provider ``MockResponse`` /
``mock_aiohttp_session`` fixtures with a unified pair of helpers that
mimic ``httpx.Response`` and ``httpx.AsyncClient`` closely enough for
unit tests that exercise the provider request/response plumbing.

Why a shared helper instead of per-provider fixtures: the providers
share an HTTP path (``get_session_safe -> client.request -> response``),
so a single mock implementation can be patched into any provider's
``onellm.providers.<name>.get_session_safe`` symbol and yield a clean
fake for that test.

Two patterns are supported:

1. **Patching get_session_safe directly** (used by openai/anthropic-style
   tests). Use ``patch_get_session_safe(provider_module, response)`` to
   install a fake client whose ``request``/``get``/``stream`` calls
   return ``response``.

2. **Direct fake response construction** for tests that build their own
   client wiring. Construct ``MockHttpxResponse(...)`` and pass it to
   ``MockHttpxClient(...)``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch


class MockHttpxResponse:
    """Mimic ``httpx.Response`` for unit tests.

    Captures the difference from the old aiohttp mock:
      - ``status_code`` (not ``status``)
      - ``json()`` and ``text`` are synchronous in httpx
      - ``content`` is a synchronous bytes property
      - ``aread()``/``aclose()`` are async but no-ops here
      - ``aiter_lines()`` yields decoded text lines (line terminators
        already stripped, matching real httpx behaviour)
    """

    def __init__(
        self,
        data: dict[str, Any] | bytes | str | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._data = data

        if isinstance(data, bytes):
            self._content = data
            try:
                self._text = data.decode("utf-8")
            except UnicodeDecodeError:
                self._text = ""
        elif isinstance(data, str):
            self._text = data
            self._content = data.encode("utf-8")
        elif isinstance(data, dict):
            payload = json.dumps(data)
            self._text = payload
            self._content = payload.encode("utf-8")
        else:
            self._text = ""
            self._content = b""

    # --- Buffered-mode accessors (mirror httpx exactly) ----------------

    @property
    def content(self) -> bytes:
        return self._content

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> Any:
        if isinstance(self._data, dict):
            return self._data
        return json.loads(self._text or "{}")

    # --- Streaming-mode accessors --------------------------------------

    async def aread(self) -> bytes:
        """No-op in tests; the body is already in memory. Real httpx
        uses this to drain the stream into the buffered path."""
        return self._content

    async def aclose(self) -> None:
        """No-op; mirrors real ``httpx.Response.aclose``."""
        return None

    async def aiter_lines(self):
        for line in self._text.splitlines():
            yield line

    async def aiter_bytes(self):
        yield self._content


class MockHttpxClient:
    """Mimic ``httpx.AsyncClient`` enough to satisfy provider tests.

    ``request`` / ``get`` / ``post`` return the configured response
    directly (no async-context-manager wrapping; that's the aiohttp
    shape, not httpx).

    ``stream`` is an async context manager that yields the response,
    matching real httpx's streaming entry point.
    """

    def __init__(self, response: MockHttpxResponse | None = None) -> None:
        self.response = response or MockHttpxResponse({})
        self.is_closed = False
        # Lightweight call-recording for test assertions.
        self.calls: list[dict[str, Any]] = []

    async def request(self, **kwargs: Any) -> MockHttpxResponse:
        self.calls.append({"verb": "request", **kwargs})
        return self.response

    async def get(self, url: str, **kwargs: Any) -> MockHttpxResponse:
        self.calls.append({"verb": "get", "url": url, **kwargs})
        return self.response

    async def post(self, url: str, **kwargs: Any) -> MockHttpxResponse:
        self.calls.append({"verb": "post", "url": url, **kwargs})
        return self.response

    @asynccontextmanager
    async def stream(self, **kwargs: Any):
        self.calls.append({"verb": "stream", **kwargs})
        yield self.response

    async def aclose(self) -> None:
        self.is_closed = True


def patch_get_session_safe(
    provider_module: str,
    response: MockHttpxResponse | MockHttpxClient | None = None,
):
    """Return a ``patch`` context that swaps ``get_session_safe`` for a
    fake returning ``(MockHttpxClient, False)``.

    Args:
        provider_module: Dotted path to the provider module whose
            ``get_session_safe`` symbol should be swapped, e.g.
            ``"onellm.providers.openai"``. The patch is applied to that
            module's symbol (not the source) so each provider can be
            mocked independently.
        response: Optional canned response. Pass either a
            ``MockHttpxResponse`` (wrapped in a fresh client) or a
            pre-built ``MockHttpxClient`` (used as-is).
    """
    if isinstance(response, MockHttpxClient):
        client = response
    else:
        client = MockHttpxClient(response)

    async def _fake_get_session_safe(_pool_key: str):
        return client, False

    return patch(f"{provider_module}.get_session_safe", _fake_get_session_safe), client

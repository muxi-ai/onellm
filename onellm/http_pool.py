#!/usr/bin/env python3
#
# HTTP Connection Pool Manager for OneLLM
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP connection-pool manager for OneLLM.

Provides persistent, per-provider :class:`httpx.AsyncClient` instances that
multiplex many in-flight requests over a small number of HTTP/2 connections.
Compared to the previous ``aiohttp.ClientSession``-based pool, the wins are:

- **HTTP/2 multiplexing**: a burst of N parallel calls to the same provider
  no longer needs N TCP+TLS handshakes; httpx negotiates h2 via TLS ALPN
  and streams every request as a separate logical h2 stream over one
  connection. Falls back to HTTP/1.1 transparently when a server does not
  advertise h2.
- **Single transport per provider**: keepalive amortizes TLS the same way
  the old pool did, but the connection pool sizing is no longer the
  bottleneck (1-2 sockets typically suffice instead of 20).
- **Built-in fine-grained timeouts**: connect/read/write/pool can each
  have their own ceiling; we expose a single ``request_timeout`` knob in
  :class:`PoolConfig` for compatibility but use it as the read timeout
  (the dominant axis for LLM streaming).

Usage::

    import onellm
    onellm.init_pooling()           # enable pooling, HTTP/2 on by default
    # ... use OneLLM normally ...
    await onellm.close_pooling()    # cleanup on shutdown
"""

from __future__ import annotations

import asyncio
import warnings

import httpx


class PoolConfig:
    """Configuration for the HTTP connection pool.

    Field names are preserved from the pre-httpx implementation so existing
    callers of :func:`onellm.init_pooling` continue to work without any
    code changes.
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_per_host: int = 20,
        keepalive_timeout: int = 30,
        dns_cache_ttl: int = 300,
        request_timeout: int = 300,
        http2: bool = True,
    ):
        """
        Args:
            max_connections: Hard cap on total connections held open across
                all providers. Maps to :class:`httpx.Limits.max_connections`.
            max_per_host: Maximum keepalive connections per host. Under
                HTTP/2 a value of 1-2 is usually plenty because requests
                multiplex onto a single connection. Maps to
                :class:`httpx.Limits.max_keepalive_connections`.
            keepalive_timeout: Seconds an idle keepalive connection stays
                in the pool before being recycled. Maps to
                :class:`httpx.Limits.keepalive_expiry`.
            dns_cache_ttl: Retained for back-compat. **No-op** under httpx;
                the system resolver caches DNS responses and keepalive
                connections amortize lookups in the steady state. Setting
                a non-default value emits a one-shot
                :class:`DeprecationWarning`.
            request_timeout: Total request deadline in seconds. Used as
                the httpx ``timeout`` value (applied to connect, read,
                write, and pool acquire).
            http2: Whether to negotiate HTTP/2 via TLS ALPN. Defaults to
                ``True``; servers that do not advertise h2 fall back to
                HTTP/1.1 automatically. Set to ``False`` to force h1.1
                across the board (escape hatch for misbehaving upstreams).
        """
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self.keepalive_timeout = keepalive_timeout
        self.dns_cache_ttl = dns_cache_ttl
        self.request_timeout = request_timeout
        self.http2 = http2

        if dns_cache_ttl != 300:
            warnings.warn(
                "PoolConfig.dns_cache_ttl is a no-op under httpx and will be "
                "removed in a future release. Modern OS resolvers cache DNS "
                "responses, and HTTP/2 multiplexing keeps a single connection "
                "alive across requests.",
                DeprecationWarning,
                stacklevel=2,
            )


class HTTPConnectionPool:
    """Global HTTP connection pool manager with per-provider clients.

    Each provider key (``"openai"``, ``"anthropic"``, ...) gets its own
    :class:`httpx.AsyncClient`. A single client multiplexes all in-flight
    calls to that provider over an HTTP/2 connection.

    .. note::
       The class attribute name :attr:`_sessions` is preserved (rather than
       renamed to ``_clients``) so any external monkey-patching that touched
       it continues to work. The values are now :class:`httpx.AsyncClient`
       instances, not :class:`aiohttp.ClientSession`.
    """

    _sessions: dict[str, httpx.AsyncClient] = {}
    _config: PoolConfig | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Lazily construct the asyncio lock so import order doesn't bind it
        to a particular event loop."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def configure(cls, config: PoolConfig) -> None:
        """Configure (or reconfigure) the pool. Subsequent ``get_session``
        calls will use the new config; existing clients are not torn down
        eagerly - call :meth:`close_all` first if you need a clean slate."""
        cls._config = config

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether pooling is currently configured."""
        return cls._config is not None

    @classmethod
    async def get_session(cls, pool_key: str = "default") -> httpx.AsyncClient:
        """Return the :class:`httpx.AsyncClient` for ``pool_key``,
        constructing one on first access.

        The client is shared across all callers of the same ``pool_key``
        and is intentionally never used as an async context manager - the
        whole point of the pool is to keep it alive across many calls.
        Use :meth:`close_all` on shutdown.
        """
        if cls._config is None:
            raise RuntimeError(
                "Connection pooling not initialized. Call onellm.init_pooling() first."
            )

        async with cls._get_lock():
            existing = cls._sessions.get(pool_key)
            if existing is None or existing.is_closed:
                cls._sessions[pool_key] = httpx.AsyncClient(
                    http2=cls._config.http2,
                    limits=httpx.Limits(
                        max_connections=cls._config.max_connections,
                        max_keepalive_connections=cls._config.max_per_host,
                        keepalive_expiry=cls._config.keepalive_timeout,
                    ),
                    timeout=httpx.Timeout(cls._config.request_timeout),
                )
            return cls._sessions[pool_key]

    @classmethod
    async def close_all(cls) -> None:
        """Close every pooled client. Call once on application shutdown."""
        async with cls._get_lock():
            for client in cls._sessions.values():
                if not client.is_closed:
                    await client.aclose()
            cls._sessions.clear()
            cls._config = None


async def get_http_session(pool_key: str = "default") -> httpx.AsyncClient:
    """Return the pooled :class:`httpx.AsyncClient` for ``pool_key``.

    Thin convenience wrapper around :meth:`HTTPConnectionPool.get_session`;
    raises :class:`RuntimeError` if pooling has not been initialized.
    """
    return await HTTPConnectionPool.get_session(pool_key)


async def get_session_safe(pool_key: str) -> tuple[httpx.AsyncClient, bool]:
    """Return an :class:`httpx.AsyncClient` with graceful fallback.

    When pooling is enabled and healthy, returns the shared pooled client
    and ``is_pooled=True`` (caller MUST NOT close it). When pooling is
    disabled or initialization failed, constructs a fresh ad-hoc client
    and returns ``is_pooled=False`` (caller is responsible for closing it,
    typically in a ``finally`` block via ``await client.aclose()``).

    The ad-hoc client also enables HTTP/2; this keeps behaviour consistent
    whether or not the application opted into pooling.
    """
    try:
        if HTTPConnectionPool.is_enabled():
            return await get_http_session(pool_key), True
    except Exception:
        pass
    return httpx.AsyncClient(http2=True), False

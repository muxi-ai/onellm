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

"""
HTTP Connection Pool Manager for OneLLM.

Provides persistent connection pooling using shared aiohttp.ClientSession instances
per provider. This reduces TCP/TLS handshake overhead for sequential LLM calls.

Usage:
    import onellm
    onellm.init_pooling()  # Enable pooling
    # ... use OneLLM normally ...
    await onellm.close_pooling()  # Cleanup on shutdown
"""

import aiohttp


class PoolConfig:
    """Configuration for HTTP connection pooling."""

    def __init__(
        self,
        max_connections: int = 100,
        max_per_host: int = 20,
        keepalive_timeout: int = 30,
        dns_cache_ttl: int = 300,
        request_timeout: int = 300,
    ):
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self.keepalive_timeout = keepalive_timeout
        self.dns_cache_ttl = dns_cache_ttl
        self.request_timeout = request_timeout


class HTTPConnectionPool:
    """Global HTTP connection pool manager with per-provider sessions."""

    _sessions: dict[str, aiohttp.ClientSession] = {}
    _config: PoolConfig | None = None
    _lock: "aiohttp.locks.Lock | None" = None

    @classmethod
    def _get_lock(cls) -> "aiohttp.locks.Lock":
        """Get or create the asyncio lock (lazy initialization)."""
        if cls._lock is None:
            import asyncio
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def configure(cls, config: PoolConfig) -> None:
        """Configure the pool settings."""
        cls._config = config

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if pooling is enabled."""
        return cls._config is not None

    @classmethod
    async def get_session(cls, pool_key: str = "default") -> aiohttp.ClientSession:
        """
        Get or create a session for the given pool key.

        Args:
            pool_key: Identifier for the session pool (usually provider name)

        Returns:
            aiohttp.ClientSession for the given pool key
        """
        if cls._config is None:
            raise RuntimeError(
                "Connection pooling not initialized. Call onellm.init_pooling() first."
            )

        async with cls._get_lock():
            if pool_key not in cls._sessions or cls._sessions[pool_key].closed:
                cls._sessions[pool_key] = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(
                        limit=cls._config.max_connections,
                        limit_per_host=cls._config.max_per_host,
                        ttl_dns_cache=cls._config.dns_cache_ttl,
                        keepalive_timeout=cls._config.keepalive_timeout,
                    ),
                    timeout=aiohttp.ClientTimeout(total=cls._config.request_timeout),
                )
            return cls._sessions[pool_key]

    @classmethod
    async def close_all(cls) -> None:
        """Close all sessions (call on shutdown)."""
        async with cls._get_lock():
            for session in cls._sessions.values():
                if not session.closed:
                    await session.close()
            cls._sessions.clear()
            cls._config = None


async def get_http_session(pool_key: str = "default") -> aiohttp.ClientSession:
    """
    Get a pooled HTTP session for the given provider.

    Args:
        pool_key: Provider identifier (e.g., "openai", "anthropic")

    Returns:
        aiohttp.ClientSession from the pool
    """
    return await HTTPConnectionPool.get_session(pool_key)


async def get_session_safe(pool_key: str) -> tuple[aiohttp.ClientSession, bool]:
    """
    Get an HTTP session with graceful fallback.

    If pooling is enabled and working, returns a pooled session.
    If pooling fails or is disabled, creates a new session (caller must close it).

    Args:
        pool_key: Provider identifier (e.g., "openai", "anthropic")

    Returns:
        Tuple of (session, is_pooled). If is_pooled is False, caller must close session.
    """
    try:
        if HTTPConnectionPool.is_enabled():
            return await get_http_session(pool_key), True
    except Exception:
        pass
    return aiohttp.ClientSession(), False

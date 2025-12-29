"""Unit tests for the HTTP connection pooling module."""

import pytest

from onellm.http_pool import (
    HTTPConnectionPool,
    PoolConfig,
    get_http_session,
    get_session_safe,
)


class TestPoolConfig:
    """Tests for PoolConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PoolConfig()
        assert config.max_connections == 100
        assert config.max_per_host == 20
        assert config.keepalive_timeout == 30
        assert config.dns_cache_ttl == 300
        assert config.request_timeout == 300

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PoolConfig(
            max_connections=50,
            max_per_host=10,
            keepalive_timeout=60,
            dns_cache_ttl=600,
            request_timeout=120,
        )
        assert config.max_connections == 50
        assert config.max_per_host == 10
        assert config.keepalive_timeout == 60
        assert config.dns_cache_ttl == 600
        assert config.request_timeout == 120


class TestHTTPConnectionPool:
    """Tests for HTTPConnectionPool class."""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """Clean up pool state before and after each test."""
        await HTTPConnectionPool.close_all()
        yield
        await HTTPConnectionPool.close_all()

    def test_is_enabled_without_config(self):
        """Test that pool is disabled without configuration."""
        assert HTTPConnectionPool.is_enabled() is False

    def test_is_enabled_with_config(self):
        """Test that pool is enabled after configuration."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)
        assert HTTPConnectionPool.is_enabled() is True

    @pytest.mark.asyncio
    async def test_get_session_without_config_raises(self):
        """Test that getting session without config raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await HTTPConnectionPool.get_session("test")

    @pytest.mark.asyncio
    async def test_get_session_returns_session(self):
        """Test that get_session returns a valid session."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)

        session = await HTTPConnectionPool.get_session("openai")
        assert session is not None
        assert not session.closed

    @pytest.mark.asyncio
    async def test_get_session_reuses_session(self):
        """Test that get_session returns the same session for same key."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)

        session1 = await HTTPConnectionPool.get_session("openai")
        session2 = await HTTPConnectionPool.get_session("openai")
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_different_keys_get_different_sessions(self):
        """Test that different keys get different sessions."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)

        session_openai = await HTTPConnectionPool.get_session("openai")
        session_anthropic = await HTTPConnectionPool.get_session("anthropic")
        assert session_openai is not session_anthropic

    @pytest.mark.asyncio
    async def test_close_all_closes_sessions(self):
        """Test that close_all closes all sessions and clears config."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)

        session = await HTTPConnectionPool.get_session("openai")
        assert not session.closed

        await HTTPConnectionPool.close_all()
        assert session.closed
        assert HTTPConnectionPool.is_enabled() is False


class TestGetSessionSafe:
    """Tests for get_session_safe helper function."""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """Clean up pool state before and after each test."""
        await HTTPConnectionPool.close_all()
        yield
        await HTTPConnectionPool.close_all()

    @pytest.mark.asyncio
    async def test_returns_fallback_when_disabled(self):
        """Test that fallback session is returned when pooling is disabled."""
        session, pooled = await get_session_safe("openai")
        assert session is not None
        assert pooled is False
        # Caller is responsible for closing fallback sessions
        await session.close()

    @pytest.mark.asyncio
    async def test_returns_pooled_when_enabled(self):
        """Test that pooled session is returned when pooling is enabled."""
        config = PoolConfig()
        HTTPConnectionPool.configure(config)

        session, pooled = await get_session_safe("openai")
        assert session is not None
        assert pooled is True
        # Should NOT close pooled sessions - they are managed by the pool

    @pytest.mark.asyncio
    async def test_fallback_sessions_are_independent(self):
        """Test that fallback sessions are not reused."""
        session1, pooled1 = await get_session_safe("openai")
        session2, pooled2 = await get_session_safe("openai")

        assert pooled1 is False
        assert pooled2 is False
        assert session1 is not session2

        # Clean up fallback sessions
        await session1.close()
        await session2.close()


class TestInitPoolingIntegration:
    """Integration tests for init_pooling API."""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """Clean up pool state before and after each test."""
        import onellm
        try:
            await onellm.close_pooling()
        except Exception:
            pass
        yield
        try:
            await onellm.close_pooling()
        except Exception:
            pass

    def test_init_pooling_enables_pool(self):
        """Test that init_pooling enables the connection pool."""
        import onellm

        assert HTTPConnectionPool.is_enabled() is False
        onellm.init_pooling()
        assert HTTPConnectionPool.is_enabled() is True

    def test_init_pooling_with_custom_config(self):
        """Test that init_pooling accepts custom configuration."""
        import onellm

        onellm.init_pooling(max_connections=50, max_per_host=10)
        assert HTTPConnectionPool.is_enabled() is True
        assert HTTPConnectionPool._config.max_connections == 50
        assert HTTPConnectionPool._config.max_per_host == 10

    @pytest.mark.asyncio
    async def test_close_pooling_disables_pool(self):
        """Test that close_pooling disables the connection pool."""
        import onellm

        onellm.init_pooling()
        assert HTTPConnectionPool.is_enabled() is True

        await onellm.close_pooling()
        assert HTTPConnectionPool.is_enabled() is False

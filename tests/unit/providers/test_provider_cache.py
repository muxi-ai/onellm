"""
Tests for provider instance memoization and config-copy semantics.

get_provider() memoizes default (no-kwargs) instances and invalidates the
cache when configuration changes; get_provider_config() returns a copy so
per-instance kwargs never leak into the global configuration.
"""

import pytest

from onellm.config import (
    config,
    get_provider_config,
    set_api_key,
    update_provider_config,
)
from onellm.providers.base import (
    clear_provider_cache,
    get_provider,
    register_provider,
)


@pytest.fixture(autouse=True)
def _isolated_provider_state():
    """Give each test a keyed config and a clean instance cache."""
    original_openai = dict(config["providers"]["openai"])
    set_api_key("test-cache-key", "openai")
    clear_provider_cache()

    yield

    config["providers"]["openai"] = original_openai
    clear_provider_cache()


class TestProviderInstanceCache:
    def test_default_instance_is_memoized(self):
        assert get_provider("openai") is get_provider("openai")

    def test_kwargs_bypass_cache(self):
        cached = get_provider("openai")
        with_kwargs = get_provider("openai", api_key="sk-other")
        assert with_kwargs is not cached
        # And the kwargs instance is not cached in place of the default one
        assert get_provider("openai") is cached

    def test_set_api_key_invalidates_cache(self):
        before = get_provider("openai")
        set_api_key("test-cache-key-2", "openai")
        after = get_provider("openai")
        assert after is not before
        assert after.api_key == "test-cache-key-2"

    def test_update_provider_config_invalidates_cache(self):
        before = get_provider("openai")
        update_provider_config("openai", timeout=99)
        after = get_provider("openai")
        assert after is not before
        assert after.timeout == 99

    def test_register_provider_invalidates_cache(self):
        default_instance = get_provider("openai")

        class FakeProvider:
            def __init__(self, **kwargs):
                pass

        original_class = type(default_instance)
        try:
            register_provider("openai", FakeProvider)
            assert isinstance(get_provider("openai"), FakeProvider)
        finally:
            register_provider("openai", original_class)

    def test_clear_provider_cache(self):
        before = get_provider("openai")
        clear_provider_cache()
        assert get_provider("openai") is not before


class TestProviderConfigCopy:
    def test_returned_config_is_a_copy(self):
        get_provider_config("openai")["api_key"] = "mutated"
        assert config["providers"]["openai"]["api_key"] != "mutated"

    def test_constructor_kwargs_do_not_leak_into_global_config(self):
        get_provider("openai", api_key="sk-leak-check", timeout=123)
        assert config["providers"]["openai"].get("api_key") != "sk-leak-check"
        assert config["providers"]["openai"].get("timeout") != 123

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct-target coverage tests for providers/base.py.

This file specifically targets only the uncovered lines: 91, 113, 133, 153, and 171.
Each test directly hits one of these specific lines for maximum coverage efficiency.
"""

from unittest import mock
import pytest

from muxi_llm.providers.base import (
    Provider, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY
)


class TestDirectLinesCoverage:
    """Tests directly targeting specific lines in providers/base.py."""

    def setup_method(self):
        """Save original registry state before each test."""
        self.original_registry = _PROVIDER_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry state after each test."""
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_line_91_provider_get_name(self):
        """Directly target line 91: Provider.get_provider_name."""
        # Create a minimal concrete Provider class
        class MinimalProvider(Provider):
            """Minimal Provider implementation for testing."""

            async def create_chat_completion(self, *args, **kwargs):
                pass

            async def create_completion(self, *args, **kwargs):
                pass

            async def create_embedding(self, *args, **kwargs):
                pass

            async def upload_file(self, *args, **kwargs):
                pass

            async def download_file(self, *args, **kwargs):
                pass

        # Direct call to the class method to ensure line 91 is covered
        result = MinimalProvider.get_provider_name()

        # Verify correct behavior
        assert result == "minimal"

    def test_line_113_abstractmethod(self):
        """Directly target line 113: @abstractmethod in create_chat_completion."""
        # Attempt to create a Provider subclass without implementing the abstract methods
        class IncompleteProvider(Provider):
            """Deliberately incomplete provider implementation."""
            pass

        # This should fail with a TypeError mentioning the abstract methods
        with pytest.raises(TypeError) as excinfo:
            IncompleteProvider()

        # Verify error message mentions create_chat_completion (line 113)
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "create_chat_completion" in str(excinfo.value)

    def test_line_133_get_provider_not_found(self):
        """Directly target line 133: ValueError in get_provider."""
        # Clear the registry and add one known provider
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY["test"] = mock.MagicMock()

        # Call get_provider with an unknown provider name
        with pytest.raises(ValueError) as excinfo:
            get_provider("unknown")

        # Verify the error message includes the unknown provider name
        error_msg = str(excinfo.value)
        assert "Provider 'unknown' is not supported" in error_msg
        # Verify it includes the supported providers list
        assert "Supported providers: test" in error_msg

    def test_line_153_list_providers(self):
        """Directly target line 153: list_providers function."""
        # Set up a controlled registry state
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY["provider1"] = mock.MagicMock()
        _PROVIDER_REGISTRY["provider2"] = mock.MagicMock()

        # Call list_providers which executes line 153
        result = list_providers()

        # Verify result is correct
        assert set(result) == {"provider1", "provider2"}

    def test_line_171_provider_with_fallbacks_no_fallbacks(self):
        """Directly target line 171: get_provider_with_fallbacks with no fallbacks."""
        # Register a test provider
        _PROVIDER_REGISTRY.clear()
        test_provider = mock.MagicMock()
        register_provider("testprovider", lambda **kwargs: test_provider)

        # Mock parse_model_name to avoid dependency
        with mock.patch("muxi_llm.providers.base.parse_model_name") as mock_parse:
            # Configure the mock to return a known value
            mock_parse.return_value = ("testprovider", "model123")

            # Call the function with no fallback_models
            # This will execute line 171
            provider, model = get_provider_with_fallbacks("testprovider/model123")

            # Verify results
            assert provider is test_provider  # Same instance
            assert model == "model123"

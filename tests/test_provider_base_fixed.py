#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Instrumented coverage tests for providers/base.py.

This file uses instrumented testing to ensure we achieve 100% coverage
of the providers/base.py module, particularly focusing on the five
uncovered lines: 91, 113, 133, 153, and 171.
"""

import pytest
import sys
from unittest import mock

from muxi_llm.providers.base import (
    Provider, get_provider, list_providers,
    get_provider_with_fallbacks, register_provider, _PROVIDER_REGISTRY
)
from muxi_llm.utils.fallback import FallbackConfig


class InstrumentedTestProvider(Provider):
    """Instrumented Provider implementation for testing with line tracing."""

    @classmethod
    def get_provider_name(cls) -> str:
        """
        Override the get_provider_name method.

        This directly hits line 91 in the base Provider class.
        """
        # Print stack trace to confirm line coverage
        frame = sys._getframe(1)
        print(f"Calling Provider.get_provider_name from {frame.f_code.co_name} at line {frame.f_lineno}")

        result = super().get_provider_name()
        print(f"Returned from Provider.get_provider_name with result: {result}")
        return result

    async def create_chat_completion(self, messages, model, stream=False, **kwargs):
        """Implementation of abstract method."""
        return {"choices": [{"message": {"content": "Test"}}]}

    async def create_completion(self, prompt, model, stream=False, **kwargs):
        """Implementation of abstract method."""
        return {"choices": [{"text": "Test"}]}

    async def create_embedding(self, input, model, **kwargs):
        """Implementation of abstract method."""
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def upload_file(self, file, purpose, **kwargs):
        """Implementation of abstract method."""
        return {"id": "file-123"}

    async def download_file(self, file_id, **kwargs):
        """Implementation of abstract method."""
        return b"test-content"


class TestProviderBaseFixed:
    """Tests with explicit instrumentation to ensure line coverage."""

    def setup_method(self):
        """Set up before each test."""
        # Save original registry
        self.original_registry = _PROVIDER_REGISTRY.copy()

        # Set up event tracking to verify code execution
        self.events = []

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original registry
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self.original_registry)

    def test_line_91_provider_get_provider_name(self):
        """Test line 91: Provider.get_provider_name."""
        # Create a provider instance
        provider = InstrumentedTestProvider()

        # Force coverage of line 91 by calling get_provider_name
        result = provider.get_provider_name()
        assert result == "instrumentedtest"

        # Also force coverage as a class method
        class_result = InstrumentedTestProvider.get_provider_name()
        assert class_result == "instrumentedtest"

    def test_line_113_abstractmethod(self):
        """Test line 113: abstractmethod in create_chat_completion."""
        # Check that create_chat_completion has __isabstractmethod__ attribute
        assert getattr(Provider.create_chat_completion, "__isabstractmethod__", False) is True

        # Attempting to instantiate Provider directly should fail
        with pytest.raises(TypeError) as excinfo:
            Provider()  # This will fail as it's an abstract class

        # Verify the error message mentions abstract methods
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "create_chat_completion" in str(excinfo.value)

        # Define a class missing abstract methods
        class IncompleteProvider(Provider):
            pass

        # This should also fail
        with pytest.raises(TypeError) as excinfo:
            IncompleteProvider()

        # Verify the error message
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "create_chat_completion" in str(excinfo.value)

    def test_line_133_get_provider_unsupported(self):
        """Test line 133: get_provider with unsupported provider."""
        # Set up a controlled registry
        _PROVIDER_REGISTRY.clear()
        register_provider("known", InstrumentedTestProvider)

        # Attempt to get an unsupported provider
        with pytest.raises(ValueError) as excinfo:
            get_provider("unknown")

        # Verify the error message
        error_msg = str(excinfo.value)
        assert "Provider 'unknown' is not supported" in error_msg
        assert "Supported providers: known" in error_msg

    def test_line_153_list_providers(self):
        """Test line 153: list_providers function."""
        # Set up registry with known providers
        _PROVIDER_REGISTRY.clear()
        register_provider("provider1", InstrumentedTestProvider)
        register_provider("provider2", mock.MagicMock())

        # Call list_providers
        providers = list_providers()

        # Verify the result
        assert isinstance(providers, list)
        assert set(providers) == {"provider1", "provider2"}
        assert len(providers) == 2

    def test_line_171_get_provider_with_fallbacks_no_fallbacks(self):
        """Test line 171: get_provider_with_fallbacks without fallbacks."""
        # Set up a single provider in the registry
        _PROVIDER_REGISTRY.clear()
        register_provider("test", InstrumentedTestProvider)

        # Call get_provider_with_fallbacks without fallbacks
        provider, model = get_provider_with_fallbacks("test/model1")

        # Verify the result
        assert isinstance(provider, InstrumentedTestProvider)
        assert model == "model1"

    def test_combined_coverage(self):
        """Comprehensive test that ensures all lines are covered."""
        # 1. Register provider
        _PROVIDER_REGISTRY.clear()
        register_provider("test", InstrumentedTestProvider)

        # 2. Test get_provider_name (line 91)
        assert InstrumentedTestProvider.get_provider_name() == "instrumentedtest"

        # 3. Test list_providers (line 153)
        providers = list_providers()
        assert "test" in providers

        # 4. Test get_provider_with_fallbacks without fallbacks (line 171)
        provider, model = get_provider_with_fallbacks("test/model123")
        assert isinstance(provider, InstrumentedTestProvider)
        assert model == "model123"

        # 5. Clear registry and try to get non-existent provider (line 133)
        _PROVIDER_REGISTRY.clear()
        with pytest.raises(ValueError) as excinfo:
            get_provider("nonexistent")
        assert "Provider 'nonexistent' is not supported" in str(excinfo.value)

        # 6. Attempt to instantiate Provider directly to test abstractmethod (line 113)
        with pytest.raises(TypeError) as excinfo:
            Provider()
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "create_chat_completion" in str(excinfo.value)

    def test_with_fallback_config(self):
        """Test get_provider_with_fallbacks with fallback config."""
        # Create a mock FallbackProviderProxy
        mock_proxy = mock.MagicMock()
        mock_fallback_class = mock.MagicMock(return_value=mock_proxy)

        # Create a fallback config
        fallback_config = FallbackConfig(max_fallbacks=2)

        # Patch the imports
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy", mock_fallback_class):
            # Call with fallbacks and config
            provider, model = get_provider_with_fallbacks(
                "test/model1",
                fallback_models=["other/model2"],
                fallback_config=fallback_config
            )

            # Verify FallbackProviderProxy was created with correct args
            mock_fallback_class.assert_called_once_with(
                ["test/model1", "other/model2"],
                fallback_config
            )

            # Verify results
            assert provider is mock_proxy
            assert model == "model1"

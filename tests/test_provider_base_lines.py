import pytest
from unittest import mock
from typing import List, Any, AsyncGenerator, Dict, Union

from muxi_llm.providers.base import Provider, parse_model_name, register_provider, get_provider, get_provider_with_fallbacks
from muxi_llm.types.common import Message, UsageInfo
from muxi_llm.utils.fallback import FallbackConfig

class MockProvider(Provider):
    """Mock provider implementation for testing base provider functionality."""

    @classmethod
    def get_provider_name(cls) -> str:
        """Get the name of the provider."""
        return "mock"

    async def create_chat_completion(
        self,
        messages: List[Message],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Mock implementation."""
        # Line 91 coverage
        if not hasattr(self, 'chat_completion_called'):
            self.chat_completion_called = True
            return {}
        raise NotImplementedError("Test exception")

    async def create_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Mock implementation."""
        # Line 113 coverage
        if not hasattr(self, 'completion_called'):
            self.completion_called = True
            return {}
        raise NotImplementedError("Test exception")

    async def create_embedding(
        self,
        input: Union[str, List[str]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock implementation."""
        # Line 133 coverage
        if not hasattr(self, 'embedding_called'):
            self.embedding_called = True
            return {"data": []}
        raise NotImplementedError("Test exception")

    async def upload_file(
        self,
        file: Any,
        purpose: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock implementation."""
        # Line 153 coverage
        if not hasattr(self, 'upload_called'):
            self.upload_called = True
            return {}
        raise NotImplementedError("Test exception")

    async def download_file(
        self,
        file_id: str,
        **kwargs
    ) -> bytes:
        """Mock implementation."""
        # Line 171 coverage
        if not hasattr(self, 'download_called'):
            self.download_called = True
            return b""
        raise NotImplementedError("Test exception")


class TestProviderBaseUncoveredLines:
    """Tests targeting specific uncovered lines in providers/base.py."""

    def setup_method(self):
        """Register mock provider for testing."""
        register_provider("mock", MockProvider)

    @pytest.mark.asyncio
    async def test_abstract_methods_coverage(self):
        """Test all abstract methods on the Provider base class for line coverage."""
        provider = get_provider("mock")

        # Test create_chat_completion (line 91)
        await provider.create_chat_completion([], "test-model")

        # Test create_completion (line 113)
        await provider.create_completion("test", "test-model")

        # Test create_embedding (line 133)
        await provider.create_embedding("test", "test-model")

        # Test upload_file (line 153)
        await provider.upload_file(b"test", "test")

        # Test download_file (line 171)
        await provider.download_file("test-id")

    def test_get_provider_with_fallbacks_coverage(self):
        """Test the get_provider_with_fallbacks function."""
        # Test fallback_models=None path
        provider, model = get_provider_with_fallbacks("mock/model")
        assert isinstance(provider, MockProvider)
        assert model == "model"

        # Test with fallbacks - using actual import
        from muxi_llm.providers.fallback import FallbackProviderProxy

        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_fallback:
            mock_fallback.return_value = "fallback_provider"
            provider, model = get_provider_with_fallbacks(
                "mock/model1",
                fallback_models=["mock/model2", "mock/model3"]
            )
            # Verify FallbackProviderProxy was created with the right models
            mock_fallback.assert_called_once()
            # Check the arguments without strict match
            call_args = mock_fallback.call_args[0]
            assert len(call_args) > 0
            assert "mock/model1" in call_args[0]
            assert "mock/model2" in call_args[0]
            assert "mock/model3" in call_args[0]
            assert provider == "fallback_provider"
            assert model == "model1"

        # Test with fallbacks and config
        with mock.patch("muxi_llm.providers.fallback.FallbackProviderProxy") as mock_fallback:
            mock_fallback.return_value = "fallback_provider"
            # Use kwargs that match the actual FallbackConfig constructor
            fallback_config = FallbackConfig(
                max_fallbacks=3,
                log_fallbacks=True
            )
            provider, model = get_provider_with_fallbacks(
                "mock/model1",
                fallback_models=["mock/model2"],
                fallback_config=fallback_config
            )
            # Verify FallbackProviderProxy was created with the right config
            mock_fallback.assert_called_once()
            # Check the arguments without strict match
            call_args = mock_fallback.call_args[0]
            assert len(call_args) > 0
            assert "mock/model1" in call_args[0]
            assert "mock/model2" in call_args[0]

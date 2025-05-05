"""
Tests for complete coverage of providers/fallback.py in muxi-llm.

These tests specifically target uncovered lines in the FallbackProviderProxy class.
"""

import pytest
from unittest import mock

from muxi_llm.errors import APIError, FallbackExhaustionError
from muxi_llm.utils.fallback import FallbackConfig
from muxi_llm.providers.fallback import FallbackProviderProxy
from muxi_llm.providers.base import Provider


class TestFallbackProviderComplete:
    """Tests targeting uncovered lines in FallbackProviderProxy."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        provider = mock.MagicMock(spec=Provider)

        # Set up mock methods
        async def create_chat_completion(*args, **kwargs):
            return {"choices": [{"message": {"content": "Test response"}}]}

        async def create_embedding(*args, **kwargs):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        async def upload_file(*args, **kwargs):
            return {"id": "file-123"}

        async def download_file(*args, **kwargs):
            return b"test file content"

        async def create_speech(*args, **kwargs):
            return b"audio content"

        async def create_image(*args, **kwargs):
            return {"data": [{"url": "https://example.com/image.png"}]}

        provider.create_chat_completion = create_chat_completion
        provider.create_embedding = create_embedding
        provider.upload_file = upload_file
        provider.download_file = download_file
        provider.create_speech = create_speech
        provider.create_image = create_image

        return provider

    @pytest.mark.asyncio
    async def test_fallback_with_callback(self, mock_provider):
        """Test fallback with callback function."""
        # Setup mock providers
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock_provider

        # Make primary provider fail
        primary_provider.create_chat_completion.side_effect = APIError("Test error")

        # Create a callback to track fallbacks
        callback_called = False

        async def fallback_callback(primary_model, fallback_model, error):
            nonlocal callback_called
            callback_called = True
            assert primary_model == "openai/gpt-4"
            assert fallback_model == "anthropic/claude-3"
            assert isinstance(error, APIError)

        # Setup fallback config with callback
        config = FallbackConfig(
            fallback_callback=fallback_callback,
            log_fallbacks=True,
            retriable_errors=[APIError]
        )

        # Create fallback proxy with mocked get_provider
        models = ["openai/gpt-4", "anthropic/claude-3"]
        proxy = FallbackProviderProxy(models, config)

        # Mock the get_provider to return our mock providers
        def get_provider_mock(provider_name):
            if provider_name == "openai":
                return primary_provider
            elif provider_name == "anthropic":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            # Test the method that should trigger fallback
            result = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

        # Verify results
        assert result["choices"][0]["message"]["content"] == "Test response"
        assert callback_called is True

    @pytest.mark.asyncio
    async def test_fallback_with_max_fallbacks(self, mock_provider):
        """Test limiting the number of fallbacks."""
        # Create providers - primary and two fallbacks
        primary_provider = mock.MagicMock(spec=Provider)
        fallback1_provider = mock.MagicMock(spec=Provider)
        fallback2_provider = mock_provider

        # Make first two providers fail
        primary_provider.create_chat_completion.side_effect = APIError("Test error 1")
        fallback1_provider.create_chat_completion.side_effect = APIError("Test error 2")

        # Setup fallback config with max_fallbacks=1 (primary + 1 fallback)
        config = FallbackConfig(
            max_fallbacks=1,
            retriable_errors=[APIError]
        )

        # Create fallback proxy with 3 models
        models = ["provider1/model1", "provider2/model2", "provider3/model3"]
        proxy = FallbackProviderProxy(models, config)

        # Mock get_provider to return our mocks
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback1_provider
            elif provider_name == "provider3":
                return fallback2_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test with max_fallbacks=1, which means only the first fallback should be tried
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            with pytest.raises(FallbackExhaustionError) as excinfo:
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}]
                )

        # Verify the exception details
        assert excinfo.value.primary_model == models[0]
        assert excinfo.value.fallback_models == models[1:2]  # Only the first fallback
        assert excinfo.value.models_tried == models[:2]  # Primary + first fallback
        assert isinstance(excinfo.value.original_error, APIError)

    @pytest.mark.asyncio
    async def test_non_retriable_error(self, mock_provider):
        """Test that non-retriable errors are propagated immediately."""
        # Create providers
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock_provider

        # Make primary provider fail with a non-retriable error (ValueError)
        primary_provider.create_chat_completion.side_effect = ValueError("Non-retriable error")

        # Setup fallback config with only APIError as retriable
        config = FallbackConfig(
            retriable_errors=[APIError]  # ValueError is not included
        )

        # Create fallback proxy
        models = ["provider1/model1", "provider2/model2"]
        proxy = FallbackProviderProxy(models, config)

        # Mock get_provider
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test that ValueError is raised immediately without trying fallbacks
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            with pytest.raises(ValueError) as excinfo:
                await proxy.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}]
                )

        # Verify the error message
        assert "Non-retriable error" in str(excinfo.value)
        # Fallback provider should not have been called
        fallback_provider.create_chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_with_fallbacks(self, mock_provider):
        """Test streaming with fallbacks."""
        # Create a streaming generator for the fallback provider
        async def mock_stream():
            for i in range(3):
                yield {"choices": [{"delta": {"content": f"part {i}"}}]}

        # Set up providers
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock.MagicMock(spec=Provider)

        # Primary provider fails, fallback provider returns the generator
        primary_provider.create_chat_completion.side_effect = APIError("Test error")
        fallback_provider.create_chat_completion.return_value = mock_stream()

        # Setup fallback config
        config = FallbackConfig(retriable_errors=[APIError])

        # Create proxy
        models = ["provider1/model1", "provider2/model2"]
        proxy = FallbackProviderProxy(models, config)

        # Mock get_provider
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test streaming
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            result_generator = await proxy.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

            # Verify the generator works
            chunks = []
            async for chunk in result_generator:
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0]["choices"][0]["delta"]["content"] == "part 0"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Test handling when all providers fail."""
        # Create providers that all fail
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock.MagicMock(spec=Provider)

        primary_provider.create_completion.side_effect = APIError("Error 1")
        fallback_provider.create_completion.side_effect = APIError("Error 2")

        # Create proxy
        models = ["provider1/model1", "provider2/model2"]
        proxy = FallbackProviderProxy(models)

        # Mock get_provider
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test with all failing providers
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            with pytest.raises(FallbackExhaustionError) as excinfo:
                await proxy.create_completion(prompt="Hello")

        # Verify exception
        assert "All models failed" in str(excinfo.value)
        assert excinfo.value.models_tried == models

    @pytest.mark.asyncio
    async def test_streaming_providers_all_fail(self):
        """Test handling when all streaming providers fail."""
        # Set up providers that all fail for streaming
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock.MagicMock(spec=Provider)

        primary_provider.create_completion.side_effect = APIError("Error 1")
        fallback_provider.create_completion.side_effect = APIError("Error 2")

        # Create proxy
        models = ["provider1/model1", "provider2/model2"]
        proxy = FallbackProviderProxy(models, FallbackConfig(retriable_errors=[APIError]))

        # Mock get_provider
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test streaming with all failing providers
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            with pytest.raises(FallbackExhaustionError) as excinfo:
                await proxy.create_completion(prompt="Hello", stream=True)

        # Verify exception
        assert "All models failed" in str(excinfo.value)
        assert excinfo.value.models_tried == models

    @pytest.mark.asyncio
    async def test_other_methods(self, mock_provider):
        """Test other methods in the FallbackProviderProxy class."""
        # Setup mock providers
        primary_provider = mock.MagicMock(spec=Provider)
        fallback_provider = mock_provider

        # Make primary provider fail for all methods
        primary_provider.create_embedding.side_effect = APIError("Embedding error")
        primary_provider.upload_file.side_effect = APIError("Upload error")
        primary_provider.download_file.side_effect = APIError("Download error")
        primary_provider.create_speech.side_effect = APIError("Speech error")
        primary_provider.create_image.side_effect = APIError("Image error")

        # Create proxy
        models = ["provider1/model1", "provider2/model2"]
        proxy = FallbackProviderProxy(models, FallbackConfig(retriable_errors=[APIError]))

        # Mock get_provider
        def get_provider_mock(provider_name):
            if provider_name == "provider1":
                return primary_provider
            elif provider_name == "provider2":
                return fallback_provider
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        # Test all methods
        with mock.patch("muxi_llm.providers.fallback.get_provider", side_effect=get_provider_mock):
            # Test create_embedding
            embedding_result = await proxy.create_embedding(input="Test text")
            assert "data" in embedding_result

            # Test upload_file
            upload_result = await proxy.upload_file(file="test file", purpose="fine-tune")
            assert upload_result["id"] == "file-123"

            # Test download_file
            download_result = await proxy.download_file(file_id="file-456")
            assert download_result == b"test file content"

            # Test create_speech
            speech_result = await proxy.create_speech(input="Test speech")
            assert speech_result == b"audio content"

            # Test create_image
            image_result = await proxy.create_image(prompt="A beautiful sunset")
            assert "data" in image_result

    @pytest.mark.asyncio
    async def test_completion_streaming(self, mock_provider):
        """Test create_completion with streaming."""
        # Create a streaming generator for the mock provider
        async def mock_stream():
            for i in range(3):
                yield {"choices": [{"text": f"part {i}"}]}

        # Setup providers
        provider = mock.MagicMock(spec=Provider)
        provider.create_completion.return_value = mock_stream()

        # Create proxy
        models = ["test/model"]
        proxy = FallbackProviderProxy(models)

        # Mock get_provider
        with mock.patch("muxi_llm.providers.fallback.get_provider", return_value=provider):
            # Test create_completion with streaming
            result_generator = await proxy.create_completion(
                prompt="Hello",
                stream=True
            )

            # Verify generator
            chunks = []
            async for chunk in result_generator:
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0]["choices"][0]["text"] == "part 0"

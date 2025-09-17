#!/usr/bin/env python3
"""
Tests for the llama.cpp provider.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import time

from onellm.providers.llama_cpp import LlamaCppProvider, _MODEL_CACHE, _LAST_ACCESS
from onellm.errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    InvalidConfigurationError,
)


class TestLlamaCppProvider:
    """Test cases for llama.cpp provider."""

    def test_init_import_error(self):
        """Test initialization when llama-cpp-python is not installed."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(InvalidConfigurationError) as exc:
                LlamaCppProvider()
            assert "llama-cpp-python" in str(exc.value)
            assert "pip install" in str(exc.value)

    @patch("onellm.providers.llama_cpp.import")
    def test_init_default(self, mock_import):
        """Test initialization with default settings."""
        # Mock llama_cpp module
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            assert provider.model_dir == Path.home() / "llama_models"
            assert provider.n_ctx == 2048
            assert provider.n_gpu_layers == 0
            assert provider.n_threads > 0  # Auto-detected
            assert provider.timeout == 300

    @patch("onellm.providers.llama_cpp.import")
    def test_init_custom_config(self, mock_import):
        """Test initialization with custom configuration."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider(
                model_dir="/custom/models", n_ctx=4096, n_gpu_layers=32, n_threads=8
            )

            assert provider.model_dir == Path("/custom/models")
            assert provider.n_ctx == 4096
            assert provider.n_gpu_layers == 32
            assert provider.n_threads == 8

    @patch("onellm.providers.llama_cpp.import")
    def test_init_env_var(self, mock_import):
        """Test initialization with environment variable."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            with patch.dict("os.environ", {"LLAMA_CPP_MODEL_DIR": "/env/models"}):
                provider = LlamaCppProvider()
                assert provider.model_dir == Path("/env/models")

    @patch("onellm.providers.llama_cpp.import")
    def test_parse_model_path_simple(self, mock_import):
        """Test parsing simple model names."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock file existence
            with patch("pathlib.Path.exists", return_value=True):
                # Simple model name
                path = provider._parse_model_path("model.gguf")
                assert path == provider.model_dir / "model.gguf"

                # Model without extension
                path = provider._parse_model_path("model")
                assert path == provider.model_dir / "model.gguf"

    @patch("onellm.providers.llama_cpp.import")
    def test_parse_model_path_full(self, mock_import):
        """Test parsing full model paths."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock file existence
            with patch("pathlib.Path.exists", return_value=True):
                # Unix absolute path
                path = provider._parse_model_path("/path/to/model.gguf")
                assert path == Path("/path/to/model.gguf")

                # Windows absolute path
                path = provider._parse_model_path("C:/models/model.gguf")
                assert str(path) == "C:/models/model.gguf"

    @patch("onellm.providers.llama_cpp.import")
    def test_parse_model_path_not_found(self, mock_import):
        """Test parsing model path when file doesn't exist."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock file doesn't exist
            with patch("pathlib.Path.exists", return_value=False):
                # Relative path
                with pytest.raises(ResourceNotFoundError) as exc:
                    provider._parse_model_path("missing.gguf")
                assert "missing.gguf" in str(exc.value)
                assert str(provider.model_dir) in str(exc.value)

                # Absolute path
                with pytest.raises(ResourceNotFoundError) as exc:
                    provider._parse_model_path("/path/to/missing.gguf")
                assert "/path/to/missing.gguf" in str(exc.value)

    @patch("onellm.providers.llama_cpp.import")
    def test_load_model_new(self, mock_import):
        """Test loading a new model."""
        mock_llama_cpp = MagicMock()
        mock_model = MagicMock()
        mock_model.n_ctx = 2048
        mock_model.n_gpu_layers = 0
        mock_llama_cpp.Llama.return_value = mock_model

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Clear cache
            _MODEL_CACHE.clear()
            _LAST_ACCESS.clear()

            model_path = Path("/path/to/model.gguf")
            loaded_model = provider._load_model(model_path)

            # Check model was loaded
            assert loaded_model == mock_model
            assert str(model_path) in _MODEL_CACHE
            assert str(model_path) in _LAST_ACCESS

            # Check Llama was called correctly
            mock_llama_cpp.Llama.assert_called_once_with(
                model_path=str(model_path),
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=provider.n_threads,
                verbose=False,
            )

    @patch("onellm.providers.llama_cpp.import")
    def test_load_model_cached(self, mock_import):
        """Test loading a cached model."""
        mock_llama_cpp = MagicMock()
        mock_model = MagicMock()
        mock_model.n_ctx = 2048
        mock_model.n_gpu_layers = 0

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Pre-populate cache
            model_path = Path("/path/to/model.gguf")
            _MODEL_CACHE[str(model_path)] = mock_model
            _LAST_ACCESS[str(model_path)] = time.time()

            loaded_model = provider._load_model(model_path)

            # Should return cached model
            assert loaded_model == mock_model
            # Llama constructor should not be called
            mock_llama_cpp.Llama.assert_not_called()

    @patch("onellm.providers.llama_cpp.import")
    def test_load_model_cache_cleanup(self, mock_import):
        """Test model cache cleanup of old entries."""
        mock_llama_cpp = MagicMock()
        mock_model_new = MagicMock()
        mock_llama_cpp.Llama.return_value = mock_model_new

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Add old entry to cache
            old_model_path = "/old/model.gguf"
            _MODEL_CACHE[old_model_path] = MagicMock()
            _LAST_ACCESS[old_model_path] = time.time() - 400  # Older than timeout

            # Load new model
            new_model_path = Path("/new/model.gguf")
            provider._load_model(new_model_path)

            # Old model should be removed
            assert old_model_path not in _MODEL_CACHE
            assert old_model_path not in _LAST_ACCESS

            # New model should be cached
            assert str(new_model_path) in _MODEL_CACHE

    @patch("onellm.providers.llama_cpp.import")
    def test_convert_messages_to_prompt(self, mock_import):
        """Test converting messages to prompt."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]

            prompt = provider._convert_messages_to_prompt(messages)

            expected = (
                "System: You are helpful.\n\n"
                "User: Hello!\n\n"
                "Assistant: Hi there!\n\n"
                "User: How are you?\n\n"
                "Assistant: "
            )

            assert prompt == expected

    @pytest.mark.asyncio
    @patch("onellm.providers.llama_cpp.import")
    async def test_create_chat_completion(self, mock_import):
        """Test creating a chat completion."""
        mock_llama_cpp = MagicMock()
        mock_model = MagicMock()

        # Mock model response
        mock_model.return_value = {"choices": [{"text": "I'm doing well, thank you!"}]}

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock methods
            with patch.object(provider, "_parse_model_path") as mock_parse:
                mock_parse.return_value = Path("/path/to/model.gguf")

                with patch.object(provider, "_load_model") as mock_load:
                    mock_load.return_value = mock_model

                    response = await provider.create_chat_completion(
                        messages=[{"role": "user", "content": "How are you?"}],
                        model="model.gguf",
                        max_tokens=50,
                        temperature=0.7,
                    )

                    # Check response
                    assert response.choices[0].message["content"] == "I'm doing well, thank you!"
                    assert response.model == "model.gguf"
                    assert response.object == "chat.completion"

    @pytest.mark.asyncio
    @patch("onellm.providers.llama_cpp.import")
    async def test_create_chat_completion_streaming(self, mock_import):
        """Test creating a streaming chat completion."""
        mock_llama_cpp = MagicMock()
        mock_model = MagicMock()

        # Mock streaming response
        mock_stream = [
            {"choices": [{"text": "I'm "}]},
            {"choices": [{"text": "doing "}]},
            {"choices": [{"text": "well!"}]},
        ]
        mock_model.return_value = iter(mock_stream)

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            with patch.object(provider, "_parse_model_path") as mock_parse:
                mock_parse.return_value = Path("/path/to/model.gguf")

                with patch.object(provider, "_load_model") as mock_load:
                    mock_load.return_value = mock_model

                    stream = await provider.create_chat_completion(
                        messages=[{"role": "user", "content": "How are you?"}],
                        model="model.gguf",
                        stream=True,
                    )

                    # Collect chunks
                    chunks = []
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)

                    assert chunks == ["I'm ", "doing ", "well!"]

    @pytest.mark.asyncio
    @patch("onellm.providers.llama_cpp.import")
    async def test_create_completion(self, mock_import):
        """Test creating a text completion."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock chat completion
            mock_chat_response = MagicMock()
            mock_chat_response.choices = [MagicMock()]
            mock_chat_response.choices[0].message = {"content": "Test response"}
            mock_chat_response.choices[0].finish_reason = "stop"
            mock_chat_response.id = "test-id"
            mock_chat_response.created = 123456
            mock_chat_response.model = "model.gguf"
            mock_chat_response.usage = {"total_tokens": 10}
            mock_chat_response.system_fingerprint = None

            with patch.object(provider, "create_chat_completion") as mock_chat:
                mock_chat.return_value = mock_chat_response

                response = await provider.create_completion(
                    prompt="Complete this: ", model="model.gguf"
                )

                assert response.choices[0].text == "Test response"
                assert response.object == "text_completion"

    @pytest.mark.asyncio
    @patch("onellm.providers.llama_cpp.import")
    async def test_create_embedding_not_supported(self, mock_import):
        """Test that embeddings are not supported."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            with pytest.raises(InvalidRequestError) as exc:
                await provider.create_embedding("test", "model.gguf")

            assert "embeddings" in str(exc.value).lower()

    @patch("onellm.providers.llama_cpp.import")
    def test_list_available_models(self, mock_import):
        """Test listing available models."""
        mock_llama_cpp = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            provider = LlamaCppProvider()

            # Mock directory structure
            mock_files = [
                Path("/models/llama-3-8b-q4_K_M.gguf"),
                Path("/models/subfolder/mistral-7b-q5_K_M.gguf"),
                Path("/models/codellama-13b-q4_0.gguf"),
            ]

            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "rglob") as mock_rglob:
                    # Mock rglob to return our test files
                    mock_rglob.return_value = mock_files

                    # Need to mock relative_to for each file
                    for mock_file in mock_files:
                        mock_file.relative_to = MagicMock(return_value=Path(mock_file.name))

                    models = provider.list_available_models()

                    # Should return sorted model names
                    expected = [
                        "codellama-13b-q4_0.gguf",
                        "llama-3-8b-q4_K_M.gguf",
                        "mistral-7b-q5_K_M.gguf",
                    ]
                    assert models == expected

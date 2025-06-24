#!/usr/bin/env python3
#
# Unified interface for LLM providers using OpenAI format
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
llama.cpp provider implementation for OneLLM.

This module implements the llama.cpp provider adapter, supporting local
GGUF model files with GPU acceleration and efficient memory management.

Model naming format:
- Full path: llama-cpp//path/to/model.gguf
- Model name: llama-cpp/model.gguf (searches in configured directory)
"""

import os
import time
import multiprocessing
from pathlib import Path
from typing import Any, Dict
from collections.abc import AsyncGenerator
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import get_provider_config
from ..errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    InvalidConfigurationError,
)
from ..models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    CompletionChoice,
    CompletionResponse,
    StreamingChoice,
    EmbeddingResponse,
    FileObject,
)
from ..types import Message
from .base import Provider, register_provider


# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_TIMEOUT = 300  # 5 minutes
_LAST_ACCESS: Dict[str, float] = {}


class LlamaCppProvider(Provider):
    """llama.cpp provider implementation using Python bindings."""

    # Set capability flags
    json_mode_support = True  # Via grammar/schema
    vision_support = False  # Not yet supported
    audio_input_support = False
    video_input_support = False
    streaming_support = True
    token_by_token_support = True
    realtime_support = False
    function_calling_support = False  # Can be done via grammar

    def __init__(self, **kwargs):
        """
        Initialize the llama.cpp provider.

        Args:
            model_dir: Directory containing GGUF models
            n_ctx: Context window size
            n_gpu_layers: Number of layers to run on GPU
            n_threads: Number of CPU threads
            **kwargs: Additional configuration options
        """
        # Check if llama-cpp-python is installed
        try:
            import llama_cpp

            self.llama_cpp = llama_cpp
        except ImportError:
            raise InvalidConfigurationError(
                "llama.cpp provider requires llama-cpp-python. Install it with:\n\n"
                "# For CPU only:\n"
                "pip install llama-cpp-python\n\n"
                "# For GPU acceleration (Mac M1/M2/M3):\n"
                'CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python\n\n'
                "# For NVIDIA GPUs:\n"
                'CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python\n\n'
                "See docs/llama_cpp_tutorial.md for detailed setup instructions.",
                provider="llama_cpp",
            )

        # Get configuration
        self.config = get_provider_config("llama_cpp")

        # Extract parameters
        model_dir = kwargs.pop("model_dir", None)
        n_ctx = kwargs.pop("n_ctx", None)
        n_gpu_layers = kwargs.pop("n_gpu_layers", None)
        n_threads = kwargs.pop("n_threads", None)

        # Update configuration with provided values
        if model_dir:
            self.config["model_dir"] = model_dir
        if n_ctx is not None:
            self.config["n_ctx"] = n_ctx
        if n_gpu_layers is not None:
            self.config["n_gpu_layers"] = n_gpu_layers
        if n_threads is not None:
            self.config["n_threads"] = n_threads

        # Set model directory
        if not self.config.get("model_dir"):
            # Check environment variable
            self.config["model_dir"] = os.environ.get(
                "LLAMA_CPP_MODEL_DIR", os.path.expanduser("~/llama_models")
            )

        # Auto-detect CPU threads if not set
        if not self.config.get("n_threads"):
            self.config["n_threads"] = multiprocessing.cpu_count()

        # Store configuration
        self.model_dir = Path(self.config["model_dir"])
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.n_gpu_layers = self.config.get("n_gpu_layers", 0)
        self.n_threads = self.config.get("n_threads", 4)
        self.timeout = self.config.get("timeout", 300)

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _parse_model_path(self, model: str) -> Path:
        """
        Parse model string to get the full path to the GGUF file.

        Args:
            model: Model string (e.g., "model.gguf" or "/path/to/model.gguf")

        Returns:
            Path to the model file

        Raises:
            InvalidRequestError: If model path is invalid
            ResourceNotFoundError: If model file doesn't exist
        """
        # Handle full path (starts with /)
        if model.startswith("/") or (
            len(model) > 1 and model[1] == ":"
        ):  # Unix or Windows absolute path
            model_path = Path(model)
        else:
            # Look in model directory
            model_path = self.model_dir / model

            # Ensure .gguf extension
            if not model_path.suffix:
                model_path = model_path.with_suffix(".gguf")

        # Check if file exists
        if not model_path.exists():
            if model.startswith("/"):
                raise ResourceNotFoundError(
                    f"Model file not found: {model_path}", provider="llama_cpp"
                )
            else:
                raise ResourceNotFoundError(
                    f"Model '{model}' not found in {self.model_dir}\n"
                    f"Looked for: {model_path}\n"
                    f"Make sure the model file exists or provide a full path.",
                    provider="llama_cpp",
                )

        return model_path

    def _load_model(self, model_path: Path, **kwargs) -> Any:
        """
        Load a model from disk or return cached instance.

        Args:
            model_path: Path to the GGUF model file
            **kwargs: Additional model parameters

        Returns:
            Loaded Llama model instance
        """
        model_key = str(model_path)
        current_time = time.time()

        # Check cache
        if model_key in _MODEL_CACHE:
            # Update last access time
            _LAST_ACCESS[model_key] = current_time

            # Check if model parameters changed
            cached_model = _MODEL_CACHE[model_key]
            if cached_model.n_ctx == kwargs.get(
                "n_ctx", self.n_ctx
            ) and cached_model.n_gpu_layers == kwargs.get("n_gpu_layers", self.n_gpu_layers):
                return cached_model
            else:
                # Parameters changed, need to reload
                del _MODEL_CACHE[model_key]

        # Clean up old models
        for key, last_access in list(_LAST_ACCESS.items()):
            if current_time - last_access > _CACHE_TIMEOUT:
                if key in _MODEL_CACHE:
                    del _MODEL_CACHE[key]
                del _LAST_ACCESS[key]

        # Load new model
        try:
            model = self.llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=kwargs.get("n_ctx", self.n_ctx),
                n_gpu_layers=kwargs.get("n_gpu_layers", self.n_gpu_layers),
                n_threads=kwargs.get("n_threads", self.n_threads),
                verbose=False,
            )

            # Cache the model
            _MODEL_CACHE[model_key] = model
            _LAST_ACCESS[model_key] = current_time

            return model

        except Exception as e:
            raise InvalidRequestError(f"Failed to load model: {str(e)}", provider="llama_cpp")

    def _convert_messages_to_prompt(self, messages: list[Message]) -> str:
        """
        Convert OpenAI-style messages to a prompt string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        # Simple chat format - can be customized per model
        prompt = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Add prompt for assistant response
        prompt += "Assistant: "

        return prompt

    async def create_chat_completion(
        self, messages: list[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion using llama.cpp.

        Args:
            messages: List of messages in the conversation
            model: Model path or name
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or async generator of chunks
        """
        # Parse model path
        model_path = self._parse_model_path(model)

        # Load model with parameters
        llm = await asyncio.get_event_loop().run_in_executor(
            self._executor, lambda: self._load_model(model_path, **kwargs)
        )

        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)

        # Extract generation parameters
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 40)
        stop = kwargs.get("stop", [])

        if isinstance(stop, str):
            stop = [stop]

        # Add line breaks as stop tokens for chat format
        stop.extend(["\nUser:", "\n\nUser:"])

        if stream:
            return self._stream_chat_completion(
                llm,
                prompt,
                model_path.name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )
        else:
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                ),
            )

            # Extract text from response
            text = response["choices"][0]["text"]

            # Create OpenAI-compatible response
            choice = Choice(
                message={"role": "assistant", "content": text},
                finish_reason="stop",
                index=0,
            )

            # Token usage (approximate)
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(prompt.split()) + len(text.split()),
            }

            return ChatCompletionResponse(
                id=f"llama-cpp-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=model_path.name,
                choices=[choice],
                usage=usage,
                system_fingerprint=None,
            )

    async def _stream_chat_completion(
        self, llm: Any, prompt: str, model_name: str, **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream chat completion chunks.

        Args:
            llm: Loaded model instance
            prompt: Formatted prompt
            model_name: Model name for response
            **kwargs: Generation parameters

        Yields:
            ChatCompletionChunk objects
        """
        # Create stream in thread
        stream = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: llm(
                prompt,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 40),
                stop=kwargs.get("stop", []),
                stream=True,
            ),
        )

        # Stream tokens
        chunk_id = f"llama-cpp-{int(time.time())}"

        for token in stream:
            # Extract token text
            token_text = token["choices"][0]["text"]

            # Create delta
            delta = ChoiceDelta(
                content=token_text,
                role=None,
                function_call=None,
                tool_calls=None,
            )

            # Create streaming choice
            choice = StreamingChoice(
                delta=delta,
                finish_reason=None,
                index=0,
            )

            # Create chunk
            chunk = ChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                system_fingerprint=None,
            )

            yield chunk

        # Final chunk with finish reason
        delta = ChoiceDelta(
            content=None,
            role=None,
            function_call=None,
            tool_calls=None,
        )

        choice = StreamingChoice(
            delta=delta,
            finish_reason="stop",
            index=0,
        )

        chunk = ChatCompletionChunk(
            id=chunk_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model_name,
            choices=[choice],
            system_fingerprint=None,
        )

        yield chunk

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion.

        Args:
            prompt: Text prompt to complete
            model: Model path or name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Convert to chat format and use chat completion
        messages = [{"role": "user", "content": prompt}]

        if stream:
            # Create a generator that converts chat chunks to completion format
            async def completion_generator():
                async for chunk in await self.create_chat_completion(
                    messages, model, stream=True, **kwargs
                ):
                    # Convert chat chunk to completion format
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {
                            "id": chunk.id,
                            "object": "text_completion",
                            "created": chunk.created,
                            "model": chunk.model,
                            "choices": [
                                {
                                    "text": chunk.choices[0].delta.content,
                                    "index": 0,
                                    "finish_reason": chunk.choices[0].finish_reason,
                                }
                            ],
                        }

            return completion_generator()
        else:
            # Get chat response
            chat_response = await self.create_chat_completion(
                messages, model, stream=False, **kwargs
            )

            # Convert to completion format
            choice = CompletionChoice(
                text=chat_response.choices[0].message.get("content", ""),
                index=0,
                logprobs=None,
                finish_reason=chat_response.choices[0].finish_reason,
            )

            return CompletionResponse(
                id=chat_response.id,
                object="text_completion",
                created=chat_response.created,
                model=chat_response.model,
                choices=[choice],
                usage=chat_response.usage,
                system_fingerprint=chat_response.system_fingerprint,
            )

    async def create_embedding(
        self, input: str | list[str], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Note: llama.cpp models are primarily for text generation.
        For embeddings, consider using specialized embedding models.

        Args:
            input: Text or list of texts to embed
            model: Model path or name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Raises:
            InvalidRequestError: llama.cpp doesn't support embeddings well
        """
        raise InvalidRequestError(
            "llama.cpp models are optimized for text generation, not embeddings. "
            "Consider using specialized embedding models with other providers.",
            provider="llama_cpp",
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file.

        Note: llama.cpp doesn't support file uploads.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: Not supported
        """
        raise InvalidRequestError(
            "llama.cpp provider does not support file uploads. "
            "Models must be loaded from local GGUF files.",
            provider="llama_cpp",
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file.

        Note: llama.cpp doesn't support file downloads.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file

        Raises:
            InvalidRequestError: Not supported
        """
        raise InvalidRequestError(
            "llama.cpp provider does not support file downloads.", provider="llama_cpp"
        )

    def list_available_models(self) -> list[str]:
        """
        List available GGUF models in the model directory.

        Returns:
            List of available model file names
        """
        if not self.model_dir.exists():
            return []

        models = []
        for file in self.model_dir.rglob("*.gguf"):
            # Get relative path from model_dir
            relative_path = file.relative_to(self.model_dir)
            models.append(str(relative_path))

        return sorted(models)


# Register the provider
register_provider("llama_cpp", LlamaCppProvider)

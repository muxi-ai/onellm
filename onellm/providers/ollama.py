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
Ollama provider implementation for OneLLM.

This module implements the Ollama provider adapter, supporting local and remote
Ollama servers with dynamic endpoint routing based on model names.

Model naming format: ollama/model:tag@host:port
Examples:
- ollama/llama3:8b (uses default localhost:11434)
- ollama/llama3:8b@gpu-server:11434
- ollama/mixtral:8x7b-instruct-q4_K_M@10.0.0.5:11434
"""

import re
from typing import Any, Dict, Optional, Tuple
from collections.abc import AsyncGenerator

import aiohttp

from ..errors import (
    APIError,
    InvalidRequestError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from ..models import ChatCompletionChunk, ChatCompletionResponse
from ..types import Message
from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama provider implementation with dynamic endpoint routing."""

    provider_name = "ollama"
    default_api_base = "http://localhost:11434/v1"

    # Ollama doesn't require API keys
    requires_api_key = False

    # Vision models in Ollama
    vision_models = [
        "llava",
        "bakllava",
        "llava-llama3",
        "llava-phi3",
        "moondream",
        "minicpm-v",
        "llama3.2-vision",
    ]

    def __init__(self, **kwargs):
        """
        Initialize the Ollama provider.

        Args:
            **kwargs: Configuration options
        """
        # Initialize endpoint cache for different servers
        self._endpoint_cache: Dict[str, str] = {}
        self._model_cache: Dict[str, set] = {}  # Cache available models per endpoint

        # Call parent init
        super().__init__(**kwargs)

        # Override to not require API key
        self.api_key = "not-required"  # Ollama ignores this but OpenAI client needs it

    def _parse_ollama_model(self, model: str) -> Tuple[str, str]:
        """
        Parse Ollama model string to extract model name and endpoint.

        Args:
            model: Model string in format "model:tag@host:port" or just "model:tag"

        Returns:
            Tuple of (model_name, endpoint_url)
        """
        if "@" in model:
            # Split model and endpoint
            model_name, endpoint = model.rsplit("@", 1)

            # Ensure endpoint has protocol
            if not endpoint.startswith(("http://", "https://")):
                endpoint = f"http://{endpoint}"

            # Validate endpoint format
            if not re.match(r"https?://[^:]+:\d+", endpoint):
                raise InvalidRequestError(
                    f"Invalid endpoint format: {endpoint}. Expected format: host:port",
                    provider=self.provider_name,
                )

            # Add /v1 suffix for OpenAI compatibility
            if not endpoint.endswith("/v1"):
                endpoint = f"{endpoint}/v1"

            return model_name, endpoint
        else:
            # No endpoint specified, use default
            return model, self.api_base

    async def _check_model_available(self, model: str, endpoint: str) -> bool:
        """
        Check if a model is available on the Ollama server.

        Args:
            model: Model name (e.g., "llama3:8b")
            endpoint: Server endpoint

        Returns:
            True if model is available, False otherwise
        """
        # Check cache first
        if endpoint in self._model_cache and model in self._model_cache[endpoint]:
            return True

        try:
            # Query available models
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = {m["name"] for m in data.get("models", [])}

                        # Cache the results
                        self._model_cache[endpoint] = models

                        # Check if our model is available
                        return model in models
                    else:
                        return False
        except Exception:
            # If we can't check, assume it might be available
            return True

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        """
        Make a request to the Ollama API with dynamic endpoint routing.

        This method overrides the parent to support per-model endpoints.
        """
        # Extract model from data to determine endpoint
        model = data.get("model", "") if data else ""

        # Parse model to get endpoint
        clean_model, endpoint = self._parse_ollama_model(model)

        # Update model in data to remove endpoint suffix
        if data and "@" in model:
            data["model"] = clean_model

        # Store original api_base and temporarily override
        original_api_base = self.api_base
        self.api_base = endpoint

        try:
            # Check if model is available (non-blocking, just for user info)
            if not await self._check_model_available(clean_model, endpoint):
                # Log warning but continue - model might still work
                print(
                    f"Warning: Model '{clean_model}' not found on {endpoint}. "
                    f"You may need to pull it with: ollama pull {clean_model}"
                )

            # Call parent's _make_request with the correct endpoint
            result = await super()._make_request(
                method=method, path=path, data=data, stream=stream, timeout=timeout, files=files
            )

            return result

        except ServiceUnavailableError as e:
            # Provide more helpful error for Ollama
            raise ServiceUnavailableError(
                f"Cannot connect to Ollama server at {endpoint}. "
                f"Make sure Ollama is running (ollama serve).",
                provider=self.provider_name,
                status_code=e.status_code,
            )
        except ResourceNotFoundError as e:
            # Model not found error
            raise ResourceNotFoundError(
                f"Model '{clean_model}' not found on {endpoint}. "
                f"Pull the model with: ollama pull {clean_model}",
                provider=self.provider_name,
                status_code=e.status_code,
            )
        finally:
            # Restore original api_base
            self.api_base = original_api_base

    def _is_vision_model(self, model: str) -> bool:
        """
        Check if a model supports vision/multimodal input.

        Args:
            model: Model name

        Returns:
            True if model supports vision
        """
        # Remove endpoint suffix if present
        clean_model, _ = self._parse_ollama_model(model)

        # Check if any vision model keyword is in the model name
        model_lower = clean_model.lower()
        return any(vision_model in model_lower for vision_model in self.vision_models)

    async def create_chat_completion(
        self, messages: list[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion using Ollama.

        Supports dynamic endpoint routing via model@host:port syntax.

        Args:
            messages: List of messages
            model: Model name with optional endpoint (e.g., "llama3:8b@server:11434")
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ChatCompletionResponse or async generator of chunks
        """
        # Set vision support based on model
        self.vision_support = self._is_vision_model(model)

        # Ollama-specific parameters that might be passed
        ollama_params = ["num_gpu", "num_thread", "num_ctx", "temperature", "top_k", "top_p"]

        # Extract Ollama-specific params from kwargs
        options = {}
        for param in ollama_params:
            if param in kwargs:
                options[param] = kwargs.pop(param)

        # Add options to kwargs if any were provided
        if options:
            kwargs["options"] = options

        # Call parent implementation
        return await super().create_chat_completion(
            messages=messages, model=model, stream=stream, **kwargs
        )

    async def list_models(self, endpoint: Optional[str] = None) -> list[str]:
        """
        List available models on an Ollama server.

        Args:
            endpoint: Optional endpoint URL. If not provided, uses default.

        Returns:
            List of available model names
        """
        endpoint = endpoint or self.api_base

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
                    else:
                        raise APIError(
                            f"Failed to list models: HTTP {response.status}",
                            provider=self.provider_name,
                            status_code=response.status,
                        )
        except aiohttp.ClientError as e:
            raise ServiceUnavailableError(
                f"Cannot connect to Ollama server at {endpoint}: {str(e)}",
                provider=self.provider_name,
            )


# Register the provider
register_provider("ollama", OllamaProvider)

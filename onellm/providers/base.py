#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Base provider interface for OneLLM.

This module defines the abstract base class that all provider
implementations must follow, as well as utility functions for
working with providers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

from ..models import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    CompletionResponse,
    EmbeddingResponse,
    FileObject,
)
from ..types import Message
from ..utils.fallback import FallbackConfig


def parse_model_name(model: str) -> Tuple[str, str]:
    """
    Parse a model name with a provider prefix.

    Args:
        model: Model name with provider prefix (e.g., 'openai/gpt-4')

    Returns:
        Tuple of (provider, model_name)

    Raises:
        ValueError: If no provider prefix is found
    """
    # Check if the model name contains a provider prefix (separated by '/')
    if "/" in model:
        # Split the model name into provider and model components
        provider, model_name = model.split("/", 1)
        return provider, model_name
    else:
        # Raise an error if the model name doesn't follow the expected format
        raise ValueError(
            f"Model name '{model}' does not contain a provider prefix. "
            f"Use format 'provider/model-name' (e.g., 'openai/gpt-4')."
        )


class Provider(ABC):
    """
    Base class for all LLM providers.

    This abstract class defines the interface that all provider implementations
    must implement. It includes capability flags that indicate what features
    each provider supports, and abstract methods for core LLM operations.
    """

    # Provider capability flags
    json_mode_support = False     # Whether the provider supports JSON mode output

    # Multi-modal capabilities
    vision_support = False        # Image input support
    audio_input_support = False   # Audio input support
    video_input_support = False   # Video input support

    # Streaming capabilities
    streaming_support = False     # Basic streaming support
    token_by_token_support = False  # Granular streaming

    # Realtime capabilities
    realtime_support = False      # Realtime API support

    @classmethod
    def get_provider_name(cls) -> str:
        """
        Get the name of the provider.

        Returns:
            The provider name in lowercase, derived from the class name.
        """
        # Extract provider name from class name by removing "Provider" suffix
        return cls.__name__.replace("Provider", "").lower()

    @abstractmethod
    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion.

        Args:
            messages: List of messages in the conversation
            model: Model name without provider prefix
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        pass

    @abstractmethod
    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Args:
            prompt: Text prompt to complete
            model: Model name without provider prefix
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or a generator yielding completion chunks
        """
        pass

    @abstractmethod
    async def create_embedding(
        self, input: Union[str, List[str]], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Args:
            input: Text or list of texts to embed
            model: Model name without provider prefix
            **kwargs: Additional model parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        pass

    @abstractmethod
    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to the provider.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file
        """
        pass

    @abstractmethod
    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from the provider.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file
        """
        pass


# Registry of provider classes - stores mapping of provider names to their implementation classes
_PROVIDER_REGISTRY: Dict[str, Type[Provider]] = {}


def register_provider(provider_name: str, provider_class: Type[Provider]) -> None:
    """
    Register a provider class.

    This function adds a provider implementation to the registry, making it
    available for use throughout the library.

    Args:
        provider_name: Name of the provider (lowercase)
        provider_class: Provider class to register
    """
    # Add the provider class to the registry with the given name as the key
    _PROVIDER_REGISTRY[provider_name] = provider_class


def get_provider(provider_name: str, **kwargs) -> Provider:
    """
    Get a provider instance by name.

    This function instantiates a provider class from the registry based on the
    provided name.

    Args:
        provider_name: Name of the provider (lowercase)
        **kwargs: Additional parameters to pass to the provider constructor

    Returns:
        Provider instance

    Raises:
        ValueError: If the provider is not supported
    """
    # Look up the provider class in the registry
    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if provider_class is None:
        # If provider not found, generate a helpful error message with available options
        supported = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Provider '{provider_name}' is not supported. "
            f"Supported providers: {supported}"
        )

    # Instantiate and return the provider class with the provided kwargs
    return provider_class(**kwargs)


def list_providers() -> List[str]:
    """
    Get a list of registered provider names.

    Returns:
        List of provider names available in the registry
    """
    # Return the keys from the provider registry
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_with_fallbacks(
    primary_model: str,
    fallback_models: Optional[List[str]] = None,
    fallback_config: Optional[FallbackConfig] = None,
) -> Tuple[Provider, str]:
    """
    Get a provider with fallback support.

    This function creates either a standard provider or a fallback provider
    that can try multiple models in sequence if the primary model fails.

    Args:
        primary_model: Primary model to use (in 'provider/model' format)
        fallback_models: Optional list of fallback models (in 'provider/model' format)
        fallback_config: Optional configuration for fallback behavior

    Returns:
        Tuple of (provider, model_name)
    """
    # Parse primary model name to extract provider and model
    provider_name, model_name = parse_model_name(primary_model)

    # If no fallbacks specified, just return the normal provider
    if not fallback_models:
        return get_provider(provider_name), model_name

    # Import here to avoid circular imports
    from .fallback import FallbackProviderProxy

    # Create a fallback provider with all models (primary + fallbacks)
    all_models = [primary_model] + fallback_models
    return FallbackProviderProxy(all_models, fallback_config), model_name

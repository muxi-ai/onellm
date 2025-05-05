"""
Base provider interface for muxi-llm.

This module defines the abstract base class that all provider
implementations must follow, as well as utility functions for
working with providers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Tuple, Type, Union

from ..models import (
    ChatCompletionResponse, ChatCompletionChunk,
    CompletionResponse, EmbeddingResponse, FileObject
)
from ..types import Message


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
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    else:
        raise ValueError(
            f"Model name '{model}' does not contain a provider prefix. "
            f"Use format 'provider/model-name' (e.g., 'openai/gpt-4')."
        )


class Provider(ABC):
    """Base class for all LLM providers."""

    @classmethod
    def get_provider_name(cls) -> str:
        """Get the name of the provider."""
        return cls.__name__.replace("Provider", "").lower()

    @abstractmethod
    async def create_chat_completion(
        self,
        messages: List[Message],
        model: str,
        stream: bool = False,
        **kwargs
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
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs
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
        self,
        input: Union[str, List[str]],
        model: str,
        **kwargs
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
    async def upload_file(
        self,
        file: Any,
        purpose: str,
        **kwargs
    ) -> FileObject:
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
    async def download_file(
        self,
        file_id: str,
        **kwargs
    ) -> bytes:
        """
        Download a file from the provider.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file
        """
        pass


# Registry of provider classes
_PROVIDER_REGISTRY: Dict[str, Type[Provider]] = {}


def register_provider(provider_name: str, provider_class: Type[Provider]) -> None:
    """
    Register a provider class.

    Args:
        provider_name: Name of the provider (lowercase)
        provider_class: Provider class to register
    """
    _PROVIDER_REGISTRY[provider_name] = provider_class


def get_provider(provider_name: str, **kwargs) -> Provider:
    """
    Get a provider instance by name.

    Args:
        provider_name: Name of the provider (lowercase)
        **kwargs: Additional parameters to pass to the provider constructor

    Returns:
        Provider instance

    Raises:
        ValueError: If the provider is not supported
    """
    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if provider_class is None:
        supported = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Provider '{provider_name}' is not supported. "
            f"Supported providers: {supported}"
        )

    return provider_class(**kwargs)

"""
Embedding functionality for muxi-llm.

This module provides an Embedding class that can be used to create embeddings
from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import List, Union

from .providers.base import parse_model_name, get_provider
from .models import EmbeddingResponse


class Embedding:
    """Class for creating embeddings with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-embedding-ada-002')
            input: Text or list of texts to embed
            **kwargs: Additional model parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Example:
            >>> response = Embedding.create(
            ...     model="openai/text-embedding-ada-002",
            ...     input="Hello, world!"
            ... )
            >>> print(len(response.data[0].embedding))
        """
        # Parse model name to get provider and model
        provider_name, model_name = parse_model_name(model)

        # Get provider instance
        provider = get_provider(provider_name)

        # Call the provider's method synchronously
        return asyncio.run(
            provider.create_embedding(
                input=input,
                model=model_name,
                **kwargs
            )
        )

    @classmethod
    async def acreate(
        cls,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input asynchronously.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-embedding-ada-002')
            input: Text or list of texts to embed
            **kwargs: Additional model parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Example:
            >>> response = await Embedding.acreate(
            ...     model="openai/text-embedding-ada-002",
            ...     input="Hello, world!"
            ... )
            >>> print(len(response.data[0].embedding))
        """
        # Parse model name to get provider and model
        provider_name, model_name = parse_model_name(model)

        # Get provider instance
        provider = get_provider(provider_name)

        # Call the provider's method asynchronously
        return await provider.create_embedding(
            input=input,
            model=model_name,
            **kwargs
        )

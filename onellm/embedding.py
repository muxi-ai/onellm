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
Embedding functionality for OneLLM.

This module provides an Embedding class that can be used to create embeddings
from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .models import EmbeddingResponse
from .utils.fallback import FallbackConfig
from .errors import InvalidRequestError


def validate_embedding_input(input_data: Union[str, List[str]]) -> None:
    """
    Validate the input for embedding.

    This function checks if the input data is valid for embedding generation.
    It ensures that the input is not empty, and if it's a list, that it contains
    at least one non-empty string.

    Args:
        input_data: Text or list of texts to validate

    Raises:
        InvalidRequestError: If the input is empty or invalid
    """
    # Check if input is completely empty
    if not input_data:
        raise InvalidRequestError("Input cannot be empty")

    # If input is a list, check that it's not empty and contains at least one non-empty string
    if isinstance(input_data, list):
        if not input_data or all(not text for text in input_data):
            raise InvalidRequestError("Input cannot be empty")


class Embedding:
    """Class for creating embeddings with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        input: Union[str, List[str]],
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        This method provides a synchronous interface for embedding generation.
        It handles model fallbacks if the primary model fails.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-embedding-ada-002')
            input: Text or list of texts to embed
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional model parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Example:
            >>> response = Embedding.create(
            ...     model="openai/text-embedding-ada-002",
            ...     input="Hello, world!",
            ...     fallback_models=["openai/text-embedding-3-small"]
            ... )
            >>> print(len(response.data[0].embedding))
        """
        # Validate input before proceeding
        validate_embedding_input(input)

        # Process fallback configuration
        fb_config = None
        if fallback_config:
            # Convert dictionary to FallbackConfig object
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This returns both the provider instance and the specific model name to use
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method synchronously by running the async method in an event loop
        return asyncio.run(
            provider.create_embedding(input=input, model=model_name, **kwargs)
        )

    @classmethod
    async def acreate(
        cls,
        model: str,
        input: Union[str, List[str]],
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input asynchronously.

        This method provides an asynchronous interface for embedding generation.
        It's useful when working within an async context to avoid blocking the event loop.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-embedding-ada-002')
            input: Text or list of texts to embed
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional model parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Example:
            >>> response = await Embedding.acreate(
            ...     model="openai/text-embedding-ada-002",
            ...     input="Hello, world!",
            ...     fallback_models=["openai/text-embedding-3-small"]
            ... )
            >>> print(len(response.data[0].embedding))
        """
        # Validate input before proceeding
        validate_embedding_input(input)

        # Process fallback configuration
        fb_config = None
        if fallback_config:
            # Convert dictionary to FallbackConfig object
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This returns both the provider instance and the specific model name to use
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method asynchronously
        return await provider.create_embedding(input=input, model=model_name, **kwargs)

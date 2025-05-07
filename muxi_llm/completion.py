#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
Text completion functionality for muxi-llm.

This module provides a Completion class that can be used to create text
completions from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import Any, AsyncGenerator, List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .models import CompletionResponse
from .utils.fallback import FallbackConfig


class Completion:
    """Class for creating text completions with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        retries: int = 0,
        **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-davinci-003')
            prompt: Text prompt to complete
            stream: Whether to stream the response
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            retries: Number of times to retry the primary model before falling back (default: 0)
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or a streaming generator

        Raises:
            ValueError: If prompt is empty

        Example:
            >>> response = Completion.create(
            ...     model="openai/text-davinci-003",
            ...     prompt="Once upon a time",
            ...     max_tokens=50,
            ...     fallback_models=["anthropic/claude-instant-1", "openai/gpt-3.5-turbo-instruct"]
            ... )
            >>> print(response.choices[0].text)
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Add retries by prepending the primary model to fallback_models
        effective_fallback_models = fallback_models
        if retries > 0:
            if effective_fallback_models is None:
                effective_fallback_models = [model] * retries
            else:
                effective_fallback_models = [model] * retries + effective_fallback_models

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=effective_fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method synchronously
        if stream:
            # For streaming, we need to use async properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                provider.create_completion(
                    prompt=prompt, model=model_name, stream=stream, **kwargs
                )
            )
        else:
            # For non-streaming, we can just run and get the result
            return asyncio.run(
                provider.create_completion(
                    prompt=prompt, model=model_name, stream=stream, **kwargs
                )
            )

    @classmethod
    async def acreate(
        cls,
        model: str,
        prompt: str,
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        retries: int = 0,
        **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion asynchronously.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-davinci-003')
            prompt: Text prompt to complete
            stream: Whether to stream the response
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            retries: Number of times to retry the primary model before falling back (default: 0)
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or a streaming generator

        Raises:
            ValueError: If prompt is empty

        Example:
            >>> response = await Completion.acreate(
            ...     model="openai/text-davinci-003",
            ...     prompt="Once upon a time",
            ...     max_tokens=50,
            ...     fallback_models=["anthropic/claude-instant-1", "openai/gpt-3.5-turbo-instruct"]
            ... )
            >>> print(response.choices[0].text)
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Add retries by prepending the primary model to fallback_models
        effective_fallback_models = fallback_models
        if retries > 0:
            if effective_fallback_models is None:
                effective_fallback_models = [model] * retries
            else:
                effective_fallback_models = [model] * retries + effective_fallback_models

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=effective_fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method asynchronously
        return await provider.create_completion(
            prompt=prompt, model=model_name, stream=stream, **kwargs
        )

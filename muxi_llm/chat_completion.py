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
Chat completion functionality for muxi-llm.

This module provides a ChatCompletion class that can be used to create chat
completions from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .models import ChatCompletionResponse, ChatCompletionChunk
from .utils.fallback import FallbackConfig


class ChatCompletion:
    """Class for creating chat completions with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion.

        Args:
            model: Model name with provider prefix (e.g., 'openai/gpt-4')
            messages: List of messages in the conversation
            stream: Whether to stream the response
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects

        Example:
            >>> response = ChatCompletion.create(
            ...     model="openai/gpt-4",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Hello, how are you?"}
            ...     ],
            ...     fallback_models=["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"]
            ... )
            >>> print(response.choices[0].message["content"])
        """
        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method synchronously
        if stream:
            # For streaming, we need to use async properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                provider.create_chat_completion(
                    messages=messages, model=model_name, stream=stream, **kwargs
                )
            )
        else:
            # For non-streaming, we can just run and get the result
            return asyncio.run(
                provider.create_chat_completion(
                    messages=messages, model=model_name, stream=stream, **kwargs
                )
            )

    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion asynchronously.

        Args:
            model: Model name with provider prefix (e.g., 'openai/gpt-4')
            messages: List of messages in the conversation
            stream: Whether to stream the response
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects

        Example:
            >>> response = await ChatCompletion.acreate(
            ...     model="openai/gpt-4",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Hello, how are you?"}
            ...     ],
            ...     fallback_models=["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"]
            ... )
            >>> print(response.choices[0].message["content"])
        """
        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Call the provider's method asynchronously
        return await provider.create_chat_completion(
            messages=messages, model=model_name, stream=stream, **kwargs
        )

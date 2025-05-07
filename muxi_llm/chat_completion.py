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
import logging
import warnings
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .models import ChatCompletionResponse, ChatCompletionChunk
from .utils.fallback import FallbackConfig


class ChatCompletion:
    """Class for creating chat completions with various providers."""

    logger = logging.getLogger("muxi_llm.chat_completion")

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        retries: int = 0,
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
            retries: Number of times to retry the primary model before falling back (default: 0)
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

        # Process kwargs based on provider capabilities
        processed_kwargs = dict(kwargs)

        # Handle JSON mode (response_format parameter)
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]

            # Check if this is a JSON mode request
            is_json_mode = (
                isinstance(response_format, dict) and
                response_format.get("type") == "json_object"
            )

            if is_json_mode and not provider.json_mode_support:
                # Provider doesn't support JSON mode, remove parameter and warn
                processed_kwargs.pop("response_format", None)
                warnings.warn(
                    "The selected provider does not support JSON mode. "
                    "The 'response_format' parameter will be ignored.",
                    UserWarning,
                    stacklevel=2
                )

                # Add a system message to request JSON format if not already present
                has_system_message = False
                for msg in messages:
                    if msg.get("role") == "system":
                        has_system_message = True
                        if "json" not in msg.get("content", "").lower():
                            # Append JSON instruction to existing system message
                            msg["content"] += " Please provide your response in valid JSON format."
                        break

                if not has_system_message:
                    # Add a new system message requesting JSON
                    messages_copy = list(messages)  # Create a copy to avoid modifying the original
                    messages_copy.insert(0, {
                        "role": "system",
                        "content": "Please provide your response in valid JSON format."
                    })
                    messages = messages_copy

        # Call the provider's method synchronously
        if stream:
            # For streaming, we need to use async properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                provider.create_chat_completion(
                    messages=messages, model=model_name, stream=stream, **processed_kwargs
                )
            )
        else:
            # For non-streaming, we can just run and get the result
            return asyncio.run(
                provider.create_chat_completion(
                    messages=messages, model=model_name, stream=stream, **processed_kwargs
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
        retries: int = 0,
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
            retries: Number of times to retry the primary model before falling back (default: 0)
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

        # Process kwargs based on provider capabilities
        processed_kwargs = dict(kwargs)

        # Handle JSON mode (response_format parameter)
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]

            # Check if this is a JSON mode request
            is_json_mode = (
                isinstance(response_format, dict) and
                response_format.get("type") == "json_object"
            )

            if is_json_mode and not provider.json_mode_support:
                # Provider doesn't support JSON mode, remove parameter and warn
                processed_kwargs.pop("response_format", None)
                warnings.warn(
                    "The selected provider does not support JSON mode. "
                    "The 'response_format' parameter will be ignored.",
                    UserWarning,
                    stacklevel=2
                )

                # Add a system message to request JSON format if not already present
                has_system_message = False
                for msg in messages:
                    if msg.get("role") == "system":
                        has_system_message = True
                        if "json" not in msg.get("content", "").lower():
                            # Append JSON instruction to existing system message
                            msg["content"] += " Please provide your response in valid JSON format."
                        break

                if not has_system_message:
                    # Add a new system message requesting JSON
                    messages_copy = list(messages)  # Create a copy to avoid modifying the original
                    messages_copy.insert(0, {
                        "role": "system",
                        "content": "Please provide your response in valid JSON format."
                    })
                    messages = messages_copy

        # Call the provider's method asynchronously
        return await provider.create_chat_completion(
            messages=messages, model=model_name, stream=stream, **processed_kwargs
        )

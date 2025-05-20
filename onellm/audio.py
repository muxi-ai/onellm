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
OpenAI audio capabilities for transcription and translation.

This module provides a high-level API for OpenAI's audio capabilities.
It includes classes for audio transcription and translation to English,
with support for fallback models if the primary model fails.
"""

import asyncio
from typing import Any, Dict, IO, List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .utils.fallback import FallbackConfig


class AudioTranscription:
    """
    API class for audio transcription.

    This class provides methods to transcribe audio files to text using
    various provider models with fallback support.
    """

    @classmethod
    async def create(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        This async method takes an audio file and transcribes it to text using
        the specified model. If the primary model fails, it can fall back to
        alternative models if provided.

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters:
                - language: Optional language code (e.g., "en")
                - prompt: Optional text to guide transcription
                - response_format: Format of the response ("json", "text", "srt",
                  "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Transcription result
        """
        # Process fallback configuration - convert dict to FallbackConfig object if provided
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This returns both the provider instance and the specific model name to use
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Delegate the actual transcription to the provider implementation
        return await provider.create_transcription(file, model_name, **kwargs)

    @classmethod
    def create_sync(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of create().

        This method provides a synchronous interface to the async create() method
        by running it in an event loop. It has the same functionality but can be
        called from synchronous code.

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Transcription result
        """
        # Use asyncio.run to execute the async create method in a new event loop
        return asyncio.run(
            cls.create(
                file=file,
                model=model,
                fallback_models=fallback_models,
                fallback_config=fallback_config,
                **kwargs
            )
        )


class AudioTranslation:
    """
    API class for audio translation to English.

    This class provides methods to translate audio files to English text
    using various provider models with fallback support.
    """

    @classmethod
    async def create(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Translate audio to English text.

        This async method takes an audio file in any language and translates it
        to English text using the specified model. If the primary model fails,
        it can fall back to alternative models if provided.

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters:
                - prompt: Optional text to guide translation
                - response_format: Format of the response ("json", "text", "srt",
                  "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Translation result with text in English
        """
        # Process fallback configuration - convert dict to FallbackConfig object if provided
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This returns both the provider instance and the specific model name to use
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Delegate the actual translation to the provider implementation
        return await provider.create_translation(file, model_name, **kwargs)

    @classmethod
    def create_sync(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of create().

        This method provides a synchronous interface to the async create() method
        by running it in an event loop. It has the same functionality but can be
        called from synchronous code.

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Translation result with text in English
        """
        # Use asyncio.run to execute the async create method in a new event loop
        return asyncio.run(
            cls.create(
                file=file,
                model=model,
                fallback_models=fallback_models,
                fallback_config=fallback_config,
                **kwargs
            )
        )

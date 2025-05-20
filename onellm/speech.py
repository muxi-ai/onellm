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
Text-to-speech capabilities.

This module provides a high-level API for text-to-speech generation across multiple providers.
It supports various providers and models with fallback options, and offers both
synchronous and asynchronous interfaces for speech generation.
"""

import asyncio
from typing import List, Optional

from .providers.base import get_provider_with_fallbacks
from .utils.fallback import FallbackConfig


class Speech:
    """
    API class for text-to-speech operations.

    This class provides methods to convert text to speech using various
    provider models, with support for fallback options if the primary model fails.
    """

    @classmethod
    async def create(
        cls,
        input: str,
        model: str = "openai/tts-1",
        voice: str = "alloy",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech from text asynchronously.

        This method handles the asynchronous generation of speech from text input.
        It supports fallback models if the primary model fails and can save the
        generated audio to a file if an output path is specified.

        Args:
            input: Text to convert to speech
            model: Model ID in format "provider/model" (default: "openai/tts-1")
            voice: Voice to use (default: "alloy")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters:
                - response_format: Format of the audio ("mp3", "opus", "aac", "flac")
                - speed: Speed of the generated audio (0.25 to 4.0)
                - output_file: Optional path to save the audio to a file

        Returns:
            Audio data as bytes
        """
        # Extract output_file if provided, removing it from kwargs to avoid passing
        # it to the provider's create_speech method
        output_file = kwargs.pop("output_file", None)

        # Process fallback configuration by creating a FallbackConfig object
        # if fallback settings were provided
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This handles selecting the appropriate provider and model based on the input
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Generate speech using the selected provider and model
        audio_data = await provider.create_speech(input, model_name, voice, **kwargs)

        # Save to file if requested
        # This allows users to directly save the audio without additional code
        if output_file:
            with open(output_file, "wb") as f:
                f.write(audio_data)

        return audio_data

    @classmethod
    def create_sync(
        cls,
        input: str,
        model: str = "openai/tts-1",
        voice: str = "alloy",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> bytes:
        """
        Synchronous version of create() for text-to-speech generation.

        This method provides a convenient synchronous interface to the asynchronous
        create() method by running it in an event loop. It has the same functionality
        and parameters as the asynchronous version.

        Args:
            input: Text to convert to speech
            model: Model ID in format "provider/model" (default: "openai/tts-1")
            voice: Voice to use (default: "alloy")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Audio data as bytes
        """
        # Use asyncio.run to execute the async create method in a synchronous context
        return asyncio.run(
            cls.create(
                input=input,
                model=model,
                voice=voice,
                fallback_models=fallback_models,
                fallback_config=fallback_config,
                **kwargs
            )
        )

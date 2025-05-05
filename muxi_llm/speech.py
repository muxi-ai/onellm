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
OpenAI text-to-speech capabilities.

This module provides a high-level API for OpenAI's text-to-speech capabilities.
"""

import asyncio
from typing import List, Optional

from .providers.base import get_provider_with_fallbacks
from .utils.fallback import FallbackConfig


class Speech:
    """API class for text-to-speech."""

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
        Generate speech from text.

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
        # Extract output_file if provided
        output_file = kwargs.pop("output_file", None)

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

        # Generate speech
        audio_data = await provider.create_speech(input, model_name, voice, **kwargs)

        # Save to file if requested
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
        Synchronous version of create().

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

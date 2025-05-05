"""
OpenAI text-to-speech capabilities.

This module provides a high-level API for OpenAI's text-to-speech capabilities.
"""

import asyncio
import os
from typing import Any, Dict, Optional, Union

from .providers import get_provider
from .utils.model import parse_model_name


class Speech:
    """API class for text-to-speech."""

    @classmethod
    async def create(
        cls,
        input: str,
        model: str = "openai/tts-1",
        voice: str = "alloy",
        **kwargs
    ) -> bytes:
        """
        Generate speech from text.

        Args:
            input: Text to convert to speech
            model: Model ID in format "provider/model" (default: "openai/tts-1")
            voice: Voice to use (default: "alloy")
            **kwargs: Additional parameters:
                - response_format: Format of the audio ("mp3", "opus", "aac", "flac")
                - speed: Speed of the generated audio (0.25 to 4.0)
                - output_file: Optional path to save the audio to a file

        Returns:
            Audio data as bytes
        """
        # Extract output_file if provided
        output_file = kwargs.pop("output_file", None)

        # Get provider and model name
        provider_name, model_name = parse_model_name(model)
        provider = get_provider(provider_name)

        # Generate speech
        audio_data = await provider.create_speech(input, model_name, voice, **kwargs)

        # Save to file if requested
        if output_file:
            with open(output_file, "wb") as f:
                f.write(audio_data)

        return audio_data

    @classmethod
    def create_sync(cls, *args, **kwargs) -> bytes:
        """
        Synchronous version of create().

        Args:
            Same arguments as create()

        Returns:
            Audio data as bytes
        """
        return asyncio.run(cls.create(*args, **kwargs))

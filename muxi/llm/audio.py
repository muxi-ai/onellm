"""
OpenAI audio capabilities for transcription and translation.

This module provides a high-level API for OpenAI's audio capabilities.
"""

import asyncio
from typing import Any, Dict, IO, Union

from .providers import get_provider
from .utils.model import parse_model_name


class AudioTranscription:
    """API class for audio transcription."""

    @classmethod
    async def create(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            **kwargs: Additional parameters:
                - language: Optional language code (e.g., "en")
                - prompt: Optional text to guide transcription
                - response_format: Format of the response ("json", "text", "srt",
                  "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Transcription result
        """
        provider_name, model_name = parse_model_name(model)
        provider = get_provider(provider_name)
        return await provider.create_transcription(file, model_name, **kwargs)

    @classmethod
    def create_sync(cls, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous version of create()."""
        return asyncio.run(cls.create(*args, **kwargs))


class AudioTranslation:
    """API class for audio translation to English."""

    @classmethod
    async def create(
        cls,
        file: Union[str, bytes, IO[bytes]],
        model: str = "openai/whisper-1",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Translate audio to English text.

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            **kwargs: Additional parameters:
                - prompt: Optional text to guide translation
                - response_format: Format of the response ("json", "text", "srt",
                  "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Translation result with text in English
        """
        provider_name, model_name = parse_model_name(model)
        provider = get_provider(provider_name)
        return await provider.create_translation(file, model_name, **kwargs)

    @classmethod
    def create_sync(cls, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous version of create()."""
        return asyncio.run(cls.create(*args, **kwargs))

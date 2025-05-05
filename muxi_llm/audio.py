"""
OpenAI audio capabilities for transcription and translation.

This module provides a high-level API for OpenAI's audio capabilities.
"""

import asyncio
from typing import Any, Dict, IO, List, Optional, Union

from .providers.base import get_provider_with_fallbacks
from .utils.fallback import FallbackConfig


class AudioTranscription:
    """API class for audio transcription."""

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
        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config
        )

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

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Transcription result
        """
        return asyncio.run(cls.create(
            file=file,
            model=model,
            fallback_models=fallback_models,
            fallback_config=fallback_config,
            **kwargs
        ))


class AudioTranslation:
    """API class for audio translation to English."""

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
        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config
        )

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

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model ID in format "provider/model" (default: "openai/whisper-1")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Translation result with text in English
        """
        return asyncio.run(cls.create(
            file=file,
            model=model,
            fallback_models=fallback_models,
            fallback_config=fallback_config,
            **kwargs
        ))

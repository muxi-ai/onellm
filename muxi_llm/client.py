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
OpenAI-compatible client interface for muxi-llm.

This module implements an interface that matches OpenAI's Python client structure,
making it a drop-in replacement for OpenAI's client with the same API structure.
"""

from typing import Any, Dict, List, Optional, Union

from .chat_completion import ChatCompletion
from .completion import Completion
from .embedding import Embedding
from .image import Image
from .audio import AudioTranscription, AudioTranslation
from .speech import Speech
from .files import File


class ChatCompletionsResource:
    """Chat completions API resource"""

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create a chat completion using ChatCompletion.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return ChatCompletion.create(
            model=model,
            messages=messages,
            stream=stream,
            fallback_models=fallback_models,
            **kwargs
        )

    async def acreate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create a chat completion asynchronously using ChatCompletion.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return await ChatCompletion.acreate(
            model=model,
            messages=messages,
            stream=stream,
            fallback_models=fallback_models,
            **kwargs
        )


class ChatResource:
    """Chat API resource"""

    def __init__(self):
        self.completions = ChatCompletionsResource()


class CompletionsResource:
    """Completions API resource"""

    def create(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create a completion using Completion.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return Completion.create(
            model=model,
            prompt=prompt,
            stream=stream,
            fallback_models=fallback_models,
            **kwargs
        )

    async def acreate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create a completion asynchronously using Completion.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return await Completion.acreate(
            model=model,
            prompt=prompt,
            stream=stream,
            fallback_models=fallback_models,
            **kwargs
        )


class EmbeddingsResource:
    """Embeddings API resource"""

    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create embeddings using Embedding.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return Embedding.create(
            model=model,
            input=input,
            fallback_models=fallback_models,
            **kwargs
        )

    async def acreate(
        self,
        model: str,
        input: Union[str, List[str]],
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        """Create embeddings asynchronously using Embedding.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        if fallback_models:
            for i, fallback_model in enumerate(fallback_models):
                if "/" not in fallback_model:
                    fallback_models[i] = f"openai/{fallback_model}"

        return await Embedding.acreate(
            model=model,
            input=input,
            fallback_models=fallback_models,
            **kwargs
        )


class ImagesResource:
    """Images API resource"""

    def create(
        self,
        model: str,
        prompt: str,
        **kwargs
    ):
        """Create images using Image.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return Image.create(model=model, prompt=prompt, **kwargs)

    async def acreate(
        self,
        model: str,
        prompt: str,
        **kwargs
    ):
        """Create images asynchronously using Image.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return await Image.acreate(model=model, prompt=prompt, **kwargs)


class AudioResource:
    """Audio API resource"""

    def __init__(self):
        self.transcriptions = AudioTranscriptionsResource()
        self.translations = AudioTranslationsResource()


class AudioTranscriptionsResource:
    """Audio transcriptions API resource"""

    def create(
        self,
        model: str,
        file: str,
        **kwargs
    ):
        """Transcribe audio using AudioTranscription.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return AudioTranscription.create(model=model, file=file, **kwargs)

    async def acreate(
        self,
        model: str,
        file: str,
        **kwargs
    ):
        """Transcribe audio asynchronously using AudioTranscription.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return await AudioTranscription.acreate(model=model, file=file, **kwargs)


class AudioTranslationsResource:
    """Audio translations API resource"""

    def create(
        self,
        model: str,
        file: str,
        **kwargs
    ):
        """Translate audio using AudioTranslation.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return AudioTranslation.create(model=model, file=file, **kwargs)

    async def acreate(
        self,
        model: str,
        file: str,
        **kwargs
    ):
        """Translate audio asynchronously using AudioTranslation.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return await AudioTranslation.acreate(model=model, file=file, **kwargs)


class SpeechResource:
    """Speech API resource"""

    def create(
        self,
        model: str,
        input: str,
        voice: str,
        **kwargs
    ):
        """Generate speech using Speech.create()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return Speech.create(model=model, input=input, voice=voice, **kwargs)

    async def acreate(
        self,
        model: str,
        input: str,
        voice: str,
        **kwargs
    ):
        """Generate speech asynchronously using Speech.acreate()"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        return await Speech.acreate(model=model, input=input, voice=voice, **kwargs)


class FilesResource:
    """Files API resource"""

    def create(
        self,
        file: str,
        purpose: str,
        **kwargs
    ):
        """Upload a file using File.create()"""
        return File.create(file=file, purpose=purpose, **kwargs)

    async def acreate(
        self,
        file: str,
        purpose: str,
        **kwargs
    ):
        """Upload a file asynchronously using File.acreate()"""
        return await File.acreate(file=file, purpose=purpose, **kwargs)

    def retrieve(self, file_id: str, **kwargs):
        """Retrieve a file"""
        return File.retrieve(file_id=file_id, **kwargs)

    async def aretrieve(self, file_id: str, **kwargs):
        """Retrieve a file asynchronously"""
        return await File.aretrieve(file_id=file_id, **kwargs)

    def list(self, **kwargs):
        """List files"""
        return File.list(**kwargs)

    async def alist(self, **kwargs):
        """List files asynchronously"""
        return await File.alist(**kwargs)

    def delete(self, file_id: str, **kwargs):
        """Delete a file"""
        return File.delete(file_id=file_id, **kwargs)

    async def adelete(self, file_id: str, **kwargs):
        """Delete a file asynchronously"""
        return await File.adelete(file_id=file_id, **kwargs)

    def content(self, file_id: str, **kwargs):
        """Get file content"""
        return File.content(file_id=file_id, **kwargs)

    async def acontent(self, file_id: str, **kwargs):
        """Get file content asynchronously"""
        return await File.acontent(file_id=file_id, **kwargs)


class Client:
    """
    Base client class for muxi-llm that mimics OpenAI's client interface.
    This provides a drop-in replacement for OpenAI's client.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize client with API key and other options.

        Args:
            api_key: Optional API key to use for provider requests
            **kwargs: Additional configuration options
        """
        self.chat = ChatResource()
        self.completions = CompletionsResource()
        self.embeddings = EmbeddingsResource()
        self.images = ImagesResource()
        self.audio = AudioResource()
        self.speech = SpeechResource()
        self.files = FilesResource()

        # Store API key and other configuration (to be used by providers)
        self.api_key = api_key
        self.config = kwargs


# Alias for OpenAI = Client for backward compatibility
OpenAI = Client


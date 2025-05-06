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
from .files import File
from .image import Image
from .audio import AudioTranscription, AudioTranslation
from .speech import Speech


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
        """Create images asynchronously using Image.create() with await"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        # Use the same create method - it will return a coroutine when called from here
        return await Image.create(model=model, prompt=prompt, **kwargs)


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
        """Create audio transcriptions using AudioTranscription.create()"""
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
        """Create audio transcriptions asynchronously"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        # Use the same create method - it will return a coroutine when called from here
        return await AudioTranscription.create(model=model, file=file, **kwargs)


class AudioTranslationsResource:
    """Audio translations API resource"""

    def create(
        self,
        model: str,
        file: str,
        **kwargs
    ):
        """Create audio translations using AudioTranslation.create()"""
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
        """Create audio translations asynchronously using AudioTranslation.create() with await"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        # Use the same create method - it will return a coroutine when called from here
        return await AudioTranslation.create(model=model, file=file, **kwargs)


class SpeechResource:
    """Speech API resource"""

    def create(
        self,
        model: str,
        input: str,
        voice: str,
        **kwargs
    ):
        """Create speech synthesis using Speech.create()"""
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
        """Create speech synthesis asynchronously using Speech.create() with await"""
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        # Use the same create method - it will return a coroutine when called from here
        return await Speech.create(model=model, input=input, voice=voice, **kwargs)


class FilesResource:
    """Files API resource"""

    def create(
        self,
        file: str,
        purpose: str,
        **kwargs
    ):
        """Create file using File.upload()"""
        return File.upload(file=file, purpose=purpose, provider="openai", **kwargs)

    async def acreate(
        self,
        file: str,
        purpose: str,
        **kwargs
    ):
        """Create file asynchronously using File.aupload()"""
        return await File.aupload(file=file, purpose=purpose, provider="openai", **kwargs)

    def retrieve(self, file_id: str, **kwargs):
        """Retrieve file using File.download()"""
        return File.download(file_id=file_id, provider="openai", **kwargs)

    async def aretrieve(self, file_id: str, **kwargs):
        """Retrieve file asynchronously using File.adownload()"""
        return await File.adownload(file_id=file_id, provider="openai", **kwargs)

    def list(self, **kwargs):
        """List files"""
        return File.list(provider="openai", **kwargs)

    async def alist(self, **kwargs):
        """List files asynchronously"""
        return await File.alist(provider="openai", **kwargs)

    def delete(self, file_id: str, **kwargs):
        """Delete file"""
        return File.delete(file_id=file_id, provider="openai", **kwargs)

    async def adelete(self, file_id: str, **kwargs):
        """Delete file asynchronously"""
        return await File.adelete(file_id=file_id, provider="openai", **kwargs)

    def content(self, file_id: str, **kwargs):
        """Get file content"""
        raise NotImplementedError("File content retrieval not implemented yet")

    async def acontent(self, file_id: str, **kwargs):
        """Get file content asynchronously"""
        raise NotImplementedError("Async file content retrieval not implemented yet")


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


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
OpenAI-compatible client interface for OneLLM.

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
        """
        Create a chat completion using ChatCompletion.create()

        Args:
            model: The model identifier to use for the completion
            messages: A list of message dictionaries representing the conversation history
            stream: Whether to stream the response or return it all at once
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            A chat completion response object or a stream of completion chunks
        """
        # Automatically add provider prefix if not present
        # This allows users to specify just "gpt-4" instead of "openai/gpt-4"
        if "/" not in model:
            model = f"openai/{model}"

        # Add provider prefix to fallback models if needed
        # This ensures consistent model naming across primary and fallback models
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
        """
        Create a chat completion asynchronously using ChatCompletion.acreate()

        Args:
            model: The model identifier to use for the completion
            messages: A list of message dictionaries representing the conversation history
            stream: Whether to stream the response or return it all at once
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            A chat completion response object or an async generator of completion chunks
        """
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
        """
        Initialize the Chat resource with completions subresource
        """
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
        """
        Create a completion using Completion.create()

        Args:
            model: The model identifier to use for the completion
            prompt: The text prompt to generate completions for
            stream: Whether to stream the response or return it all at once
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            A completion response object or a stream of completion chunks
        """
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
        """
        Create a completion asynchronously using Completion.acreate()

        Args:
            model: The model identifier to use for the completion
            prompt: The text prompt to generate completions for
            stream: Whether to stream the response or return it all at once
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            A completion response object or an async generator of completion chunks
        """
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
        """
        Create embeddings using Embedding.create()

        Args:
            model: The model identifier to use for generating embeddings
            input: The text or list of texts to generate embeddings for
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            An embedding response object containing vector representations
        """
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
        """
        Create embeddings asynchronously using Embedding.acreate()

        Args:
            model: The model identifier to use for generating embeddings
            input: The text or list of texts to generate embeddings for
            fallback_models: Optional list of models to try if the primary model fails
            **kwargs: Additional parameters to pass to the model

        Returns:
            An embedding response object containing vector representations
        """
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
        """
        Create images using Image.create()

        Args:
            model: The model identifier to use for image generation
            prompt: The text prompt describing the image to generate
            **kwargs: Additional parameters like size, quality, style, etc.

        Returns:
            An image generation response object with URLs or base64 data
        """
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
        """
        Create images asynchronously using Image.create() with await

        Args:
            model: The model identifier to use for image generation
            prompt: The text prompt describing the image to generate
            **kwargs: Additional parameters like size, quality, style, etc.

        Returns:
            An image generation response object with URLs or base64 data
        """
        # Automatically add provider prefix if not present
        if "/" not in model:
            model = f"openai/{model}"
        # Use the same create method - it will return a coroutine when called from here
        return await Image.create(model=model, prompt=prompt, **kwargs)


class AudioResource:
    """Audio API resource"""

    def __init__(self):
        """
        Initialize the Audio resource with transcriptions and translations subresources
        """
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
        """
        Create audio transcriptions using AudioTranscription.create()

        Args:
            model: The model identifier to use for transcription
            file: Path to the audio file or file-like object to transcribe
            **kwargs: Additional parameters like language, prompt, etc.

        Returns:
            A transcription response object containing the transcribed text
        """
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
        """
        Create audio transcriptions asynchronously

        Args:
            model: The model identifier to use for transcription
            file: Path to the audio file or file-like object to transcribe
            **kwargs: Additional parameters like language, prompt, etc.

        Returns:
            A transcription response object containing the transcribed text
        """
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
        """
        Create audio translations using AudioTranslation.create()

        Args:
            model: The model identifier to use for translation
            file: Path to the audio file or file-like object to translate
            **kwargs: Additional parameters like prompt, response_format, etc.

        Returns:
            A translation response object containing the translated text
        """
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
        """
        Create audio translations asynchronously using AudioTranslation.create() with await

        Args:
            model: The model identifier to use for translation
            file: Path to the audio file or file-like object to translate
            **kwargs: Additional parameters like prompt, response_format, etc.

        Returns:
            A translation response object containing the translated text
        """
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
        """
        Create speech synthesis using Speech.create()

        Args:
            model: The model identifier to use for speech synthesis
            input: The text to convert to speech
            voice: The voice to use for the speech (e.g., "alloy", "echo", "fable")
            **kwargs: Additional parameters like response_format, speed, etc.

        Returns:
            A speech response object containing the audio data
        """
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
        """
        Create speech synthesis asynchronously using Speech.create() with await

        Args:
            model: The model identifier to use for speech synthesis
            input: The text to convert to speech
            voice: The voice to use for the speech (e.g., "alloy", "echo", "fable")
            **kwargs: Additional parameters like response_format, speed, etc.

        Returns:
            A speech response object containing the audio data
        """
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
        """
        Create file using File.upload()

        Args:
            file: Path to the file to upload
            purpose: The intended purpose of the file (e.g., "fine-tune", "assistants")
            **kwargs: Additional parameters for the file upload

        Returns:
            A file object containing metadata about the uploaded file
        """
        return File.upload(file=file, purpose=purpose, provider="openai", **kwargs)

    async def acreate(
        self,
        file: str,
        purpose: str,
        **kwargs
    ):
        """
        Create file asynchronously using File.aupload()

        Args:
            file: Path to the file to upload
            purpose: The intended purpose of the file (e.g., "fine-tune", "assistants")
            **kwargs: Additional parameters for the file upload

        Returns:
            A file object containing metadata about the uploaded file
        """
        return await File.aupload(file=file, purpose=purpose, provider="openai", **kwargs)

    def retrieve(self, file_id: str, **kwargs):
        """
        Retrieve file using File.download()

        Args:
            file_id: The ID of the file to retrieve
            **kwargs: Additional parameters for the file retrieval

        Returns:
            The content of the requested file
        """
        return File.download(file_id=file_id, provider="openai", **kwargs)

    async def aretrieve(self, file_id: str, **kwargs):
        """
        Retrieve file asynchronously using File.adownload()

        Args:
            file_id: The ID of the file to retrieve
            **kwargs: Additional parameters for the file retrieval

        Returns:
            The content of the requested file
        """
        return await File.adownload(file_id=file_id, provider="openai", **kwargs)

    def list(self, **kwargs):
        """
        List files

        Args:
            **kwargs: Additional parameters for filtering the file list

        Returns:
            A list of file objects containing metadata about available files
        """
        return File.list(provider="openai", **kwargs)

    async def alist(self, **kwargs):
        """
        List files asynchronously

        Args:
            **kwargs: Additional parameters for filtering the file list

        Returns:
            A list of file objects containing metadata about available files
        """
        return await File.alist(provider="openai", **kwargs)

    def delete(self, file_id: str, **kwargs):
        """
        Delete file

        Args:
            file_id: The ID of the file to delete
            **kwargs: Additional parameters for the file deletion

        Returns:
            A deletion status object
        """
        return File.delete(file_id=file_id, provider="openai", **kwargs)

    async def adelete(self, file_id: str, **kwargs):
        """
        Delete file asynchronously

        Args:
            file_id: The ID of the file to delete
            **kwargs: Additional parameters for the file deletion

        Returns:
            A deletion status object
        """
        return await File.adelete(file_id=file_id, provider="openai", **kwargs)

    def content(self, file_id: str, **kwargs):
        """
        Get file content

        Args:
            file_id: The ID of the file to retrieve content for
            **kwargs: Additional parameters for the content retrieval

        Returns:
            The content of the requested file
        """
        return File.download(file_id=file_id, provider="openai", **kwargs)

    async def acontent(self, file_id: str, **kwargs):
        """
        Get file content asynchronously

        Args:
            file_id: The ID of the file to retrieve content for
            **kwargs: Additional parameters for the content retrieval

        Returns:
            The content of the requested file
        """
        return await File.adownload(file_id=file_id, provider="openai", **kwargs)


class Client:
    """
    Base client class for OneLLM that mimics OpenAI's client interface.
    This provides a drop-in replacement for OpenAI's client.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize client with API key and other options.

        Args:
            api_key: Optional API key to use for provider requests
            **kwargs: Additional configuration options
        """
        # Initialize all API resources that match OpenAI's client structure
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
# This allows users to use either onellm.OpenAI or onellm.Client
OpenAI = Client

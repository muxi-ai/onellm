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
OpenAI provider implementation for muxi-llm.

This module implements the OpenAI provider adapter, supporting all OpenAI API
endpoints including chat completions, completions, embeddings, and file operations.
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, IO

import aiohttp

from ..config import get_provider_config
from ..models import (
    ChatCompletionResponse, ChatCompletionChunk, ChoiceDelta, StreamingChoice,
    CompletionResponse, CompletionChoice, EmbeddingResponse, EmbeddingData, FileObject,
    Choice
)
from ..errors import (
    APIError, AuthenticationError, RateLimitError,
    InvalidRequestError, ServiceUnavailableError, TimeoutError,
    BadGatewayError, PermissionError, ResourceNotFoundError
)
from ..types import Message
from ..types.common import TranscriptionResult, ImageGenerationResult
from ..utils.retry import retry_async, RetryConfig

from .base import Provider, register_provider


class OpenAIProvider(Provider):
    """OpenAI provider implementation."""

    def __init__(self, **kwargs):
        """
        Initialize the OpenAI provider.

        Args:
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides
        self.config = get_provider_config("openai")
        self.config.update(kwargs)

        # Check for required configuration
        if not self.config.get("api_key"):
            raise AuthenticationError(
                "OpenAI API key is required. Set it via environment variable OPENAI_API_KEY "
                "or pass it explicitly as api_key=<key> when creating the provider.",
                provider="openai"
            )

        # Set up configuration
        self.api_base = self.config.get("api_base", "https://api.openai.com/v1")
        self.api_key = self.config.get("api_key")
        self.organization_id = self.config.get("organization_id")
        self.timeout = float(self.config.get("timeout", 60))
        self.max_retries = int(self.config.get("max_retries", 3))

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries,
            initial_backoff=1.0,
            max_backoff=60.0
        )

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests.

        Returns:
            Dict of headers
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id

        return headers

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Make a request to the OpenAI API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            data: Request data
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            files: Files to upload

        Returns:
            Response data or streaming response

        Raises:
            MuxiLLMError: On API errors
        """
        url = f"{self.api_base}/{path.lstrip('/')}"
        timeout = timeout or self.timeout
        headers = self._get_headers()

        # Handle file uploads
        if files:
            # Need to use multipart/form-data for file uploads
            headers.pop("Content-Type", None)
            form_data = aiohttp.FormData()

            # Add file data
            for key, file_info in files.items():
                form_data.add_field(
                    key,
                    file_info["data"],
                    filename=file_info.get("filename", "file"),
                    content_type=file_info.get("content_type", "application/octet-stream")
                )

            # Add other fields
            if data:
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        form_data.add_field(key, json.dumps(value), content_type="application/json")
                    else:
                        form_data.add_field(key, str(value))

            body = form_data
        else:
            body = json.dumps(data) if data else None

        async def execute_request():
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout,
                    ssl=None  # Use default SSL settings
                ) as response:
                    if stream:
                        return self._handle_streaming_response(response)
                    else:
                        return await self._handle_response(response)

        # Use retry mechanism for non-streaming requests
        if not stream:
            return await retry_async(execute_request, config=self.retry_config)
        else:
            return await execute_request()

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Handle an API response.

        Args:
            response: API response

        Returns:
            Response data

        Raises:
            MuxiLLMError: On API errors
        """
        response_data = await response.json()

        if response.status != 200:
            self._handle_error_response(response.status, response_data)

        return response_data

    async def _handle_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a streaming API response.

        Args:
            response: API response

        Yields:
            Parsed JSON chunks

        Raises:
            MuxiLLMError: On API errors
        """
        if response.status != 200:
            error_data = await response.json()
            self._handle_error_response(response.status, error_data)

        # Process the stream line by line
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix

                # Check for the stream end marker
                if line == '[DONE]':
                    break

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip invalid lines
                    continue

    def _handle_error_response(
        self, status_code: int, response_data: Dict[str, Any]
    ) -> None:
        """
        Handle an error response.

        Args:
            status_code: HTTP status code
            response_data: Error response data

        Raises:
            MuxiLLMError: Appropriate error based on the status code
        """
        error = response_data.get("error", {})
        message = error.get("message", "Unknown error")

        if status_code == 401:
            raise AuthenticationError(message, provider="openai", status_code=status_code)
        elif status_code == 403:
            raise PermissionError(message, provider="openai", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="openai", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="openai", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="openai", status_code=status_code)
        elif status_code == 500:
            raise ServiceUnavailableError(message, provider="openai", status_code=status_code)
        elif status_code == 502:
            raise BadGatewayError(message, provider="openai", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="openai", status_code=status_code)
        else:
            raise APIError(
                f"OpenAI API error: {message} (status code: {status_code})",
                provider="openai",
                status_code=status_code,
                error_data=error
            )

    async def create_chat_completion(
        self,
        messages: List[Message],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion with OpenAI.

        Args:
            messages: List of messages in the conversation
            model: Model name (without provider prefix)
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Validate messages for multi-modal content
        processed_messages = self._process_messages_for_vision(messages, model)

        # Set up the request
        data = {
            "model": model,
            "messages": processed_messages,
            "stream": stream,
            **kwargs
        }

        # Make the request
        if stream:
            async def chunk_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
                async for chunk in await self._make_request(
                    method="POST",
                    path="chat/completions",
                    data=data,
                    stream=True
                ):
                    if chunk:
                        # Skip empty chunks
                        if "choices" not in chunk or not chunk["choices"]:
                            continue

                        # Transform choices
                        choices = []
                        for choice_data in chunk["choices"]:
                            delta_data = choice_data.get("delta", {})
                            delta = ChoiceDelta(
                                content=delta_data.get("content"),
                                role=delta_data.get("role"),
                                function_call=delta_data.get("function_call"),
                                tool_calls=delta_data.get("tool_calls"),
                                finish_reason=choice_data.get("finish_reason")
                            )
                            choice = StreamingChoice(
                                delta=delta,
                                finish_reason=choice_data.get("finish_reason"),
                                index=choice_data.get("index", 0)
                            )
                            choices.append(choice)

                        # Create the chunk response
                        chunk_resp = ChatCompletionChunk(
                            id=chunk.get("id", ""),
                            object=chunk.get("object", "chat.completion.chunk"),
                            created=chunk.get("created", int(time.time())),
                            model=chunk.get("model", model),
                            choices=choices,
                            system_fingerprint=chunk.get("system_fingerprint")
                        )
                        yield chunk_resp

            return chunk_generator()
        else:
            response_data = await self._make_request(
                method="POST",
                path="chat/completions",
                data=data
            )

            # Transform choices
            choices = []
            for choice_data in response_data.get("choices", []):
                choice = Choice(
                    message=choice_data.get("message", {}),
                    finish_reason=choice_data.get("finish_reason"),
                    index=choice_data.get("index", 0)
                )
                choices.append(choice)

            # Create the response
            response = ChatCompletionResponse(
                id=response_data.get("id", ""),
                object=response_data.get("object", "chat.completion"),
                created=response_data.get("created", int(time.time())),
                model=response_data.get("model", model),
                choices=choices,
                usage=response_data.get("usage"),
                system_fingerprint=response_data.get("system_fingerprint")
            )
            return response

    def _process_messages_for_vision(self, messages: List[Message], model: str) -> List[Message]:
        """
        Process messages to ensure they're compatible with vision models if needed.

        This checks for image content items and formats them correctly for the OpenAI API.
        Also validates that vision content is only sent to models that support it.

        Args:
            messages: Original messages
            model: Model name to check for vision support

        Returns:
            Processed messages suitable for the API

        Raises:
            InvalidRequestError: If trying to send images to a non-vision model
        """
        # Check if any message contains images
        has_images = False
        for message in messages:
            content = message.get("content", "")
            # Check for image content in list format
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url" or item.get("type") == "image":
                        has_images = True
                        break
                if has_images:
                    break

        # If no images found, return original messages
        if not has_images:
            return messages

        # Check if model supports vision
        vision_models = {
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13"
        }
        model_base = model.split("-")[0]
        model_supports_vision = any(vm in model for vm in vision_models) or model_base == "gpt4o"

        if not model_supports_vision:
            raise InvalidRequestError(
                f"Model '{model}' does not support vision inputs. "
                f"Use a vision-capable model like 'gpt-4-vision-preview' or 'gpt-4o'."
            )

        # Process each message to ensure image_url formats are correct
        processed_messages = []
        for message in messages:
            processed_message = dict(message)  # Create a copy
            content = message.get("content", "")

            # Only process if content is a list
            if isinstance(content, list):
                processed_content = []
                for item in content:
                    # Process image_url to ensure correct format
                    if item.get("type") == "image_url" and isinstance(item.get("image_url"), dict):
                        image_url = item["image_url"]
                        # Ensure url field exists
                        if "url" not in image_url:
                            raise InvalidRequestError("Image URL must contain a 'url' field")

                        # Ensure detail field is valid if present
                        if ("detail" in image_url and
                                image_url["detail"] not in ["auto", "low", "high"]):
                            image_url["detail"] = "auto"

                        processed_content.append(item)
                    # Handle other content types
                    else:
                        processed_content.append(item)

                processed_message["content"] = processed_content

            processed_messages.append(processed_message)

        return processed_messages

    async def create_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Args:
            prompt: Text prompt to complete
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Prepare request data
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }

        if stream:
            # Use streaming API
            raw_generator = await self._make_request(
                method="POST",
                path="/completions",
                data=request_data,
                stream=True
            )

            # Just return the raw chunks for now - further processing can be added as needed
            return raw_generator
        else:
            # Use non-streaming API
            response_data = await self._make_request(
                method="POST",
                path="/completions",
                data=request_data
            )

            # Convert to CompletionResponse
            choices = []
            for choice_data in response_data.get("choices", []):
                choice = CompletionChoice(
                    text=choice_data.get("text", ""),
                    index=choice_data.get("index", 0),
                    logprobs=choice_data.get("logprobs"),
                    finish_reason=choice_data.get("finish_reason")
                )
                choices.append(choice)

            return CompletionResponse(
                id=response_data.get("id", ""),
                object=response_data.get("object", "text_completion"),
                created=response_data.get("created", int(time.time())),
                model=response_data.get("model", model),
                choices=choices,
                usage=response_data.get("usage"),
                system_fingerprint=response_data.get("system_fingerprint")
            )

    async def create_embedding(
        self,
        input: Union[str, List[str]],
        model: str,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Args:
            input: Text or list of texts to embed
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Prepare request data
        request_data = {
            "model": model,
            "input": input,
            **kwargs
        }

        # Make API request
        response_data = await self._make_request(
            method="POST",
            path="/embeddings",
            data=request_data
        )

        # Convert to EmbeddingResponse
        embedding_data = []
        for data in response_data.get("data", []):
            embedding = EmbeddingData(
                embedding=data.get("embedding", []),
                index=data.get("index", 0),
                object=data.get("object", "embedding")
            )
            embedding_data.append(embedding)

        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=embedding_data,
            model=response_data.get("model", model),
            usage=response_data.get("usage")
        )

    async def upload_file(
        self,
        file: Any,
        purpose: str,
        **kwargs
    ) -> FileObject:
        """
        Upload a file to OpenAI.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file
        """
        # Prepare file data
        if isinstance(file, str):
            # File path
            with open(file, "rb") as f:
                file_data = f.read()
            filename = file.split("/")[-1]
        elif isinstance(file, bytes):
            # Bytes data
            file_data = file
            filename = kwargs.get("filename", "file.dat")
        elif hasattr(file, "read"):
            # File-like object
            file_data = file.read()
            filename = getattr(file, "name", "file.dat")
        else:
            error_msg = "Invalid file type. Expected file path, bytes, or file-like object."
            raise InvalidRequestError(error_msg)

        # Prepare request data
        request_data = {
            "purpose": purpose,
            **{k: v for k, v in kwargs.items() if k != "filename"}
        }

        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": "application/octet-stream"
            }
        }

        # Make API request
        response_data = await self._make_request(
            method="POST",
            path="/files",
            data=request_data,
            files=files
        )

        # Convert to FileObject
        return FileObject(
            id=response_data.get("id", ""),
            object=response_data.get("object", "file"),
            bytes=response_data.get("bytes", 0),
            created_at=response_data.get("created_at", int(time.time())),
            filename=response_data.get("filename", filename),
            purpose=response_data.get("purpose", purpose),
            status=response_data.get("status"),
            status_details=response_data.get("status_details")
        )

    async def download_file(
        self,
        file_id: str,
        **kwargs
    ) -> bytes:
        """
        Download a file from OpenAI.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file
        """
        url = f"{self.api_base}/files/{file_id}/content"
        timeout = kwargs.get("timeout", self.timeout)
        headers = self._get_headers()

        async def execute_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    ssl=None  # Use default SSL settings
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        self._handle_error_response(response.status, error_data)

                    return await response.read()

        # Use retry mechanism
        return await retry_async(execute_request, config=self.retry_config)

    async def create_transcription(
        self,
        file: Union[str, bytes, IO[bytes]],
        model: str = "whisper-1",
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using OpenAI's Whisper model.

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model to use for transcription (default: whisper-1)
            **kwargs: Additional parameters:
                - language: Optional language code (e.g., "en")
                - prompt: Optional text to guide transcription
                - response_format: Format of the response
                  ("json", "text", "srt", "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Transcription result
        """
        # Process the file
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data
        request_data = {
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ["filename"]}
        }

        # Set up files dictionary
        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": self._guess_audio_content_type(filename)
            }
        }

        # Make the API request
        response_data = await self._make_request(
            method="POST",
            path="/audio/transcriptions",
            data=request_data,
            files=files
        )

        # Process the response based on the response format
        response_format = kwargs.get("response_format", "json")
        if isinstance(response_format, str) and response_format != "json":
            # For non-JSON formats, return a simplified result
            return TranscriptionResult(text=response_data)

        # For JSON or default format, parse the structured response
        return TranscriptionResult(
            text=response_data.get("text", ""),
            task=response_data.get("task"),
            language=response_data.get("language"),
            duration=response_data.get("duration"),
            segments=response_data.get("segments"),
            words=response_data.get("words")
        )

    async def create_translation(
        self,
        file: Union[str, bytes, IO[bytes]],
        model: str = "whisper-1",
        **kwargs
    ) -> TranscriptionResult:
        """
        Translate audio to English text using OpenAI's Whisper model.

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model to use for translation (default: whisper-1)
            **kwargs: Additional parameters:
                - prompt: Optional text to guide translation
                - response_format: Format of the response
                  ("json", "text", "srt", "verbose_json", "vtt")
                - temperature: Temperature for sampling

        Returns:
            Translation result
        """
        # Process the file
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data
        request_data = {
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ["filename"]}
        }

        # Set up files dictionary
        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": self._guess_audio_content_type(filename)
            }
        }

        # Make the API request
        response_data = await self._make_request(
            method="POST",
            path="/audio/translations",
            data=request_data,
            files=files
        )

        # Process the response based on the response format
        response_format = kwargs.get("response_format", "json")
        if isinstance(response_format, str) and response_format != "json":
            # For non-JSON formats, return a simplified result
            return TranscriptionResult(text=response_data)

        # For JSON or default format, parse the structured response
        return TranscriptionResult(
            text=response_data.get("text", ""),
            task="translation",
            language="en",  # Translations are always to English
            duration=response_data.get("duration"),
            segments=response_data.get("segments"),
            words=response_data.get("words")
        )

    def _process_audio_file(
        self,
        file: Union[str, bytes, IO[bytes]],
        filename: Optional[str] = None
    ) -> tuple:
        """
        Process an audio file for API requests.

        Args:
            file: Audio file (path, bytes, or file-like object)
            filename: Optional filename override

        Returns:
            Tuple of (file_data, filename)

        Raises:
            InvalidRequestError: If file type is invalid
        """
        if isinstance(file, str):
            # File path
            with open(file, "rb") as f:
                file_data = f.read()
            filename = filename or file.split("/")[-1]
        elif isinstance(file, bytes):
            # Bytes data
            file_data = file
            filename = filename or "audio.mp3"
        elif hasattr(file, "read"):
            # File-like object
            file_data = file.read() if callable(file.read) else file.read
            filename = filename or getattr(file, "name", "audio.mp3")
        else:
            raise InvalidRequestError(
                "Invalid file type. Expected file path, bytes, or file-like object."
            )

        return file_data, filename

    def _guess_audio_content_type(self, filename: str) -> str:
        """
        Guess the content type based on the audio file extension.

        Args:
            filename: Name of the audio file

        Returns:
            MIME type for the audio file
        """
        # Map file extensions to MIME types
        mime_types = {
            ".mp3": "audio/mpeg",
            ".mp4": "audio/mp4",
            ".mpeg": "audio/mpeg",
            ".mpga": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
            ".webm": "audio/webm"
        }

        # Get the file extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        # Return the MIME type or a default
        return mime_types.get(ext, "audio/mpeg")

    async def _make_request_raw(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> bytes:
        """
        Make a request to the OpenAI API and return raw binary data.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            data: Request data
            timeout: Request timeout in seconds

        Returns:
            Raw binary response data

        Raises:
            MuxiLLMError: On API errors
        """
        url = f"{self.api_base}/{path.lstrip('/')}"
        timeout = timeout or self.timeout
        headers = self._get_headers()
        body = json.dumps(data) if data else None

        async def execute_request():
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout,
                    ssl=None  # Use default SSL settings
                ) as response:
                    if response.status != 200:
                        # Handle error response as JSON
                        try:
                            error_data = await response.json()
                            self._handle_error_response(response.status, error_data)
                        except json.JSONDecodeError:
                            # If not valid JSON, raise a generic error with the status code
                            error_text = await response.text()
                            raise APIError(
                                f"OpenAI API error: {error_text} (status code: {response.status})",
                                provider="openai",
                                status_code=response.status
                            )

                    # Return the raw binary data
                    return await response.read()

        # Use retry mechanism
        return await retry_async(execute_request, config=self.retry_config)

    async def create_speech(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        **kwargs
    ) -> bytes:
        """
        Generate audio from text using OpenAI's text-to-speech models.

        Args:
            input: Text to convert to speech
            model: Model to use (default: tts-1)
            voice: Voice to use (default: alloy)
            **kwargs: Additional parameters:
                - response_format: Format of the response ("mp3", "opus", "aac", "flac")
                - speed: Speed of the generated audio (0.25 to 4.0)

        Returns:
            Audio data as bytes
        """
        # Validate parameters
        if not input or not isinstance(input, str):
            raise InvalidRequestError("Input text is required and must be a string")

        # Check model - supported models as of current version
        supported_models = {"tts-1", "tts-1-hd"}
        if model not in supported_models:
            raise InvalidRequestError(
                f"Model '{model}' is not a supported TTS model. "
                f"Use one of: {', '.join(supported_models)}"
            )

        # Check voice
        supported_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        if voice not in supported_voices:
            raise InvalidRequestError(
                f"Voice '{voice}' is not supported. "
                f"Use one of: {', '.join(supported_voices)}"
            )

        # Check response format if provided
        response_format = kwargs.get("response_format", "mp3")
        supported_formats = {"mp3", "opus", "aac", "flac"}
        if response_format not in supported_formats:
            raise InvalidRequestError(
                f"Response format '{response_format}' is not supported. "
                f"Use one of: {', '.join(supported_formats)}"
            )

        # Check speed if provided
        speed = kwargs.get("speed", 1.0)
        if not isinstance(speed, (int, float)) or speed < 0.25 or speed > 4.0:
            raise InvalidRequestError(
                "Speed must be a number between 0.25 and 4.0"
            )

        # Prepare request data
        request_data = {
            "input": input,
            "model": model,
            "voice": voice,
            **{k: v for k, v in kwargs.items() if k in ["response_format", "speed"]}
        }

        # Make the API request with raw binary response
        return await self._make_request_raw(
            method="POST",
            path="/audio/speech",
            data=request_data
        )

    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        **kwargs
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt using OpenAI's DALL-E models.

        Args:
            prompt: Text description of the desired image
            model: Model to use (default: dall-e-3)
            n: Number of images to generate (default: 1)
            size: Size of the generated images (default: 1024x1024)
            **kwargs: Additional parameters:
                - quality: Quality of the image ("standard" or "hd"), for DALL-E 3
                - style: Style of image ("natural" or "vivid"), for DALL-E 3
                - response_format: Format of the response ("url" or "b64_json")
                - user: End-user ID for tracking

        Returns:
            Image generation result
        """
        # Validate parameters
        if not prompt or not isinstance(prompt, str):
            raise InvalidRequestError("Prompt is required and must be a string")

        # Check model
        supported_models = {"dall-e-2", "dall-e-3"}
        if model not in supported_models:
            raise InvalidRequestError(
                f"Model '{model}' is not a supported image generation model. "
                f"Use one of: {', '.join(supported_models)}"
            )

        # Check size based on model
        supported_sizes = {
            "dall-e-2": {"256x256", "512x512", "1024x1024"},
            "dall-e-3": {"1024x1024", "1792x1024", "1024x1792"}
        }
        if size not in supported_sizes[model]:
            raise InvalidRequestError(
                f"Size '{size}' is not supported for {model}. "
                f"Use one of: {', '.join(supported_sizes[model])}"
            )

        # Check n (number of images)
        # DALL-E 3 only supports n=1
        if model == "dall-e-3" and n > 1:
            raise InvalidRequestError(
                "DALL-E 3 only supports generating one image at a time (n=1)"
            )

        # For DALL-E 2, n can be between 1 and 10
        if model == "dall-e-2" and (not isinstance(n, int) or n < 1 or n > 10):
            raise InvalidRequestError(
                "For DALL-E 2, the number of images (n) must be between 1 and 10"
            )

        # Check quality (DALL-E 3 only)
        quality = kwargs.get("quality")
        if model == "dall-e-3" and quality is not None:
            supported_qualities = {"standard", "hd"}
            if quality not in supported_qualities:
                raise InvalidRequestError(
                    f"Quality '{quality}' is not supported. "
                    f"Use one of: {', '.join(supported_qualities)}"
                )

        # Check style (DALL-E 3 only)
        style = kwargs.get("style")
        if model == "dall-e-3" and style is not None:
            supported_styles = {"natural", "vivid"}
            if style not in supported_styles:
                raise InvalidRequestError(
                    f"Style '{style}' is not supported. "
                    f"Use one of: {', '.join(supported_styles)}"
                )

        # Check response format
        response_format = kwargs.get("response_format", "url")
        supported_formats = {"url", "b64_json"}
        if response_format not in supported_formats:
            raise InvalidRequestError(
                f"Response format '{response_format}' is not supported. "
                f"Use one of: {', '.join(supported_formats)}"
            )

        # Prepare request data
        request_data = {
            "prompt": prompt,
            "model": model,
            "n": n,
            "size": size,
            **{k: v for k, v in kwargs.items() if k in [
                "quality", "style", "response_format", "user"
            ]}
        }

        # Make the API request
        response_data = await self._make_request(
            method="POST",
            path="/images/generations",
            data=request_data
        )

        return ImageGenerationResult(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", [])
        )


# Register the OpenAI provider
register_provider("openai", OpenAIProvider)

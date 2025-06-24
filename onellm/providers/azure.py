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
Azure OpenAI provider implementation for OneLLM.

This module implements the Azure OpenAI provider adapter, supporting all Azure OpenAI API
endpoints including chat completions, completions, embeddings, and file operations.
It handles Azure-specific authentication and deployment configurations.
"""

import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, IO

import aiohttp

from ..config import get_provider_config
from ..models import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChoiceDelta,
    StreamingChoice,
    CompletionResponse,
    CompletionChoice,
    EmbeddingResponse,
    EmbeddingData,
    FileObject,
    Choice,
)
from ..errors import (
    APIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ServiceUnavailableError,
    TimeoutError,
    BadGatewayError,
    PermissionError,
    ResourceNotFoundError,
)
from ..types import Message
from ..types.common import TranscriptionResult, ImageGenerationResult
from ..utils.retry import retry_async, RetryConfig

from .base import Provider, register_provider


class AzureProvider(Provider):
    """Azure OpenAI provider implementation."""

    # Set capability flags (same as OpenAI since it's OpenAI-compatible)
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True          # GPT-4V, GPT-4 Turbo, GPT-4o support images
    audio_input_support = False    # No direct audio in chat
    video_input_support = False    # No video support

    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False       # No realtime API yet

    def __init__(self, **kwargs):
        """
        Initialize the Azure OpenAI provider.

        Args:
            azure_config_path: Path to Azure configuration JSON file
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("azure")

        # Check if azure config path is provided
        azure_config_path = (
            kwargs.pop("azure_config_path", None)
            or os.environ.get("AZURE_OPENAI_CONFIG_PATH")
        )

        if not azure_config_path:
            # Try default location
            azure_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "azure.json")

        if not os.path.exists(azure_config_path):
            raise AuthenticationError(
                "Azure configuration file not found. "
                "Set it via environment variable AZURE_OPENAI_CONFIG_PATH "
                "or provide azure_config_path parameter.",
                provider="azure",
            )

        # Load Azure configuration
        with open(azure_config_path, 'r') as f:
            self.azure_config = json.load(f)

        # Store relevant configuration
        self.key1 = self.azure_config.get("key1")
        self.key2 = self.azure_config.get("key2")
        self.region = self.azure_config.get("region")
        self.endpoint = self.azure_config.get("endpoint", "").rstrip("/")
        self.deployments = self.azure_config.get("deployment", {})

        # Default configuration
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        self.api_version = kwargs.get("api_version", "2024-12-01-preview")

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

        # Check for required configuration
        if not self.key1 and not self.key2:
            raise AuthenticationError(
                "Azure OpenAI API key is required in the configuration file.",
                provider="azure",
            )

    def _get_deployment_config(self, model: str) -> Dict[str, Any]:
        """
        Get deployment configuration for a specific model.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "o4-mini")

        Returns:
            Deployment configuration dictionary
        """
        # Check if model has specific deployment configuration
        if model in self.deployments:
            return self.deployments[model]

        # Fallback to default configuration
        return {
            "endpoint": self.endpoint,
            "deployment": model,
            "subscription_key": self.key1 or self.key2,
            "api_version": self.api_version
        }

    def _get_headers(self, deployment_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Get the headers for API requests.

        Args:
            deployment_config: Deployment configuration

        Returns:
            Dict of headers
        """
        # Azure uses api-key header instead of Authorization Bearer
        headers = {
            "Content-Type": "application/json",
            "api-key": deployment_config.get("subscription_key", self.key1 or self.key2),
        }

        return headers

    def _get_url(self, deployment_config: Dict[str, Any], path: str) -> str:
        """
        Get the full URL for an API request.

        Args:
            deployment_config: Deployment configuration
            path: API path

        Returns:
            Full URL
        """
        endpoint = deployment_config.get("endpoint", self.endpoint).rstrip("/")
        deployment = deployment_config.get("deployment")
        api_version = deployment_config.get("api_version", self.api_version)

        # Azure OpenAI URL format
        base_path = f"/openai/deployments/{deployment}{path}"
        url = f"{endpoint}{base_path}?api-version={api_version}"

        return url

    async def _make_request(
        self,
        method: str,
        path: str,
        model: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Make a request to the Azure OpenAI API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            model: Model/deployment name
            data: Request data
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            files: Files to upload

        Returns:
            Response data or streaming response

        Raises:
            MuxiLLMError: On API errors
        """
        # Get deployment configuration
        deployment_config = self._get_deployment_config(model)

        # Construct the full URL
        url = self._get_url(deployment_config, path)

        # Use provided timeout or fall back to default
        timeout = timeout or self.timeout

        # Get authentication headers
        headers = self._get_headers(deployment_config)

        # Handle file uploads
        if files:
            # Need to use multipart/form-data for file uploads
            headers.pop("Content-Type", None)  # Remove Content-Type for multipart form
            form_data = aiohttp.FormData()

            # Add file data to the form
            for key, file_info in files.items():
                form_data.add_field(
                    key,
                    file_info["data"],
                    filename=file_info.get("filename", "file"),
                    content_type=file_info.get(
                        "content_type", "application/octet-stream"
                    ),
                )

            # Add other fields to the form
            if data:
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        # Convert complex objects to JSON strings
                        form_data.add_field(
                            key, json.dumps(value), content_type="application/json"
                        )
                    else:
                        # Add simple values as strings
                        form_data.add_field(key, str(value))

            body = form_data
        else:
            # For regular JSON requests, serialize the data
            body = json.dumps(data) if data else None

        async def execute_request():
            """Inner function to execute the HTTP request with proper error handling"""
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout,
                ) as response:
                    if stream:
                        # For streaming responses, return a generator
                        return self._handle_streaming_response(response)
                    else:
                        # For regular responses, parse JSON and handle errors
                        return await self._handle_response(response)

        # Use retry mechanism for non-streaming requests
        if not stream:
            return await retry_async(execute_request, config=self.retry_config)
        else:
            # Streaming requests don't use retry mechanism
            return await execute_request()

    async def _handle_response(
        self, response: aiohttp.ClientResponse
    ) -> Dict[str, Any]:
        """
        Handle an API response.

        Args:
            response: API response

        Returns:
            Response data

        Raises:
            MuxiLLMError: On API errors
        """
        # Parse the JSON response
        response_data = await response.json()

        # Check for error status codes
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
        # Check for error status codes
        if response.status != 200:
            error_data = await response.json()
            self._handle_error_response(response.status, error_data)

        # Process the stream line by line
        async for line in response.content:
            line = line.decode("utf-8").strip()
            # Azure OpenAI's streaming format prefixes each JSON chunk with "data: "
            if line.startswith("data: "):
                line = line[6:]  # Remove 'data: ' prefix

                # Check for the stream end marker
                if line == "[DONE]":
                    break

                try:
                    # Parse the JSON chunk
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
        # Extract error details from the response
        error = response_data.get("error", {})
        message = error.get("message", "Unknown error")

        # Map HTTP status codes to appropriate error types
        if status_code == 401:
            raise AuthenticationError(
                message, provider="azure", status_code=status_code
            )
        elif status_code == 403:
            raise PermissionError(message, provider="azure", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(
                message, provider="azure", status_code=status_code
            )
        elif status_code == 429:
            raise RateLimitError(message, provider="azure", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(
                message, provider="azure", status_code=status_code
            )
        elif status_code == 500:
            raise ServiceUnavailableError(
                message, provider="azure", status_code=status_code
            )
        elif status_code == 502:
            raise BadGatewayError(message, provider="azure", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="azure", status_code=status_code)
        else:
            # Generic error for unhandled status codes
            raise APIError(
                f"Azure OpenAI API error: {message} (status code: {status_code})",
                provider="azure",
                status_code=status_code,
                error_data=error,
            )

    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion with Azure OpenAI.

        Args:
            messages: List of messages in the conversation
            model: Model/deployment name
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Process messages for vision models if needed
        processed_messages = self._process_messages_for_vision(messages, model)

        # Set up the request data (remove model since Azure uses deployment name in URL)
        data = {
            "messages": processed_messages,
            "stream": stream,
            **kwargs,
        }

        # Make the request
        if stream:
            # Handle streaming response
            async def chunk_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
                """Generator function to process streaming chunks"""
                async for chunk in await self._make_request(
                    method="POST", path="/chat/completions", model=model, data=data, stream=True
                ):
                    if chunk:
                        # Skip empty chunks
                        if "choices" not in chunk or not chunk["choices"]:
                            continue

                        # Transform choices into our model format
                        choices = []
                        for choice_data in chunk["choices"]:
                            # Extract delta data from the chunk
                            delta_data = choice_data.get("delta", {})
                            # Create a ChoiceDelta object
                            delta = ChoiceDelta(
                                content=delta_data.get("content"),
                                role=delta_data.get("role"),
                                function_call=delta_data.get("function_call"),
                                tool_calls=delta_data.get("tool_calls"),
                                finish_reason=choice_data.get("finish_reason"),
                            )
                            # Create a StreamingChoice object
                            choice = StreamingChoice(
                                delta=delta,
                                finish_reason=choice_data.get("finish_reason"),
                                index=choice_data.get("index", 0),
                            )
                            choices.append(choice)

                        # Create the chunk response object
                        chunk_resp = ChatCompletionChunk(
                            id=chunk.get("id", ""),
                            object=chunk.get("object", "chat.completion.chunk"),
                            created=chunk.get("created", int(time.time())),
                            model=chunk.get("model", model),
                            choices=choices,
                            system_fingerprint=chunk.get("system_fingerprint"),
                        )
                        yield chunk_resp

            return chunk_generator()
        else:
            # Handle non-streaming response
            response_data = await self._make_request(
                method="POST", path="/chat/completions", model=model, data=data
            )

            # Transform choices into our model format
            choices = []
            for choice_data in response_data.get("choices", []):
                choice = Choice(
                    message=choice_data.get("message", {}),
                    finish_reason=choice_data.get("finish_reason"),
                    index=choice_data.get("index", 0),
                )
                choices.append(choice)

            # Create the response object
            response = ChatCompletionResponse(
                id=response_data.get("id", ""),
                object=response_data.get("object", "chat.completion"),
                created=response_data.get("created", int(time.time())),
                model=response_data.get("model", model),
                choices=choices,
                usage=response_data.get("usage"),
                system_fingerprint=response_data.get("system_fingerprint"),
            )
            return response

    def _process_messages_for_vision(
        self, messages: List[Message], model: str
    ) -> List[Message]:
        """
        Process messages to ensure they're compatible with vision models if needed.

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
        # List of models known to support vision capabilities
        vision_models = {
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4v",
        }
        # Check if the model supports vision
        model_supports_vision = any(vm in model.lower() for vm in vision_models)

        # Raise error if trying to use images with a non-vision model
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
                    if item.get("type") == "image_url" and isinstance(
                        item.get("image_url"), dict
                    ):
                        image_url = item["image_url"]
                        # Ensure url field exists
                        if "url" not in image_url:
                            raise InvalidRequestError(
                                "Image URL must contain a 'url' field"
                            )

                        # Ensure detail field is valid if present
                        if "detail" in image_url and image_url["detail"] not in [
                            "auto",
                            "low",
                            "high",
                        ]:
                            # Default to "auto" if an invalid detail level is provided
                            image_url["detail"] = "auto"

                        processed_content.append(item)
                    # Handle other content types
                    else:
                        processed_content.append(item)

                processed_message["content"] = processed_content

            processed_messages.append(processed_message)

        return processed_messages

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Args:
            prompt: Text prompt to complete
            model: Model/deployment name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Prepare request data
        request_data = {"prompt": prompt, "stream": stream, **kwargs}

        if stream:
            # Use streaming API
            raw_generator = await self._make_request(
                method="POST", path="/completions", model=model, data=request_data, stream=True
            )

            # Just return the raw chunks for now - further processing can be added as needed
            return raw_generator
        else:
            # Use non-streaming API
            response_data = await self._make_request(
                method="POST", path="/completions", model=model, data=request_data
            )

            # Convert API response to CompletionResponse model
            choices = []
            for choice_data in response_data.get("choices", []):
                choice = CompletionChoice(
                    text=choice_data.get("text", ""),
                    index=choice_data.get("index", 0),
                    logprobs=choice_data.get("logprobs"),
                    finish_reason=choice_data.get("finish_reason"),
                )
                choices.append(choice)

            # Create and return the structured response object
            return CompletionResponse(
                id=response_data.get("id", ""),
                object=response_data.get("object", "text_completion"),
                created=response_data.get("created", int(time.time())),
                model=response_data.get("model", model),
                choices=choices,
                usage=response_data.get("usage"),
                system_fingerprint=response_data.get("system_fingerprint"),
            )

    async def create_embedding(
        self, input: Union[str, List[str]], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Args:
            input: Text or list of texts to embed
            model: Model/deployment name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Prepare request data
        request_data = {"input": input, **kwargs}

        # Make API request to embeddings endpoint
        response_data = await self._make_request(
            method="POST", path="/embeddings", model=model, data=request_data
        )

        # Convert API response to EmbeddingResponse model
        embedding_data = []
        for data in response_data.get("data", []):
            embedding = EmbeddingData(
                embedding=data.get("embedding", []),
                index=data.get("index", 0),
                object=data.get("object", "embedding"),
            )
            embedding_data.append(embedding)

        # Create and return the structured response object
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=embedding_data,
            model=response_data.get("model", model),
            usage=response_data.get("usage"),
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to Azure OpenAI.

        Note: Azure OpenAI may not support file uploads in the same way as OpenAI.
        This is included for interface compatibility.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file
        """
        raise InvalidRequestError(
            "Azure OpenAI does not support file uploads through the standard API. "
            "Files are typically handled through Azure Blob Storage or included in requests.",
            provider="azure"
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from Azure OpenAI.

        Note: Azure OpenAI may not support file downloads in the same way as OpenAI.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file
        """
        raise InvalidRequestError(
            "Azure OpenAI does not support file downloads through the standard API.",
            provider="azure"
        )

    async def create_transcription(
        self, file: Union[str, bytes, IO[bytes]], model: str = "whisper-1", **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using Azure OpenAI's Whisper model.

        Args:
            file: Audio file to transcribe (path, bytes, or file-like object)
            model: Model/deployment name (default: whisper-1)
            **kwargs: Additional parameters

        Returns:
            Transcription result
        """
        # Process the file to get binary data and filename
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data
        request_data = {
            **{k: v for k, v in kwargs.items() if k not in ["filename"]},
        }

        # Set up files dictionary for multipart upload
        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": self._guess_audio_content_type(filename),
            }
        }

        # Make the API request to transcriptions endpoint
        response_data = await self._make_request(
            method="POST", path="/audio/transcriptions", model=model, data=request_data, files=files
        )

        # Process the response based on the requested format
        response_format = kwargs.get("response_format", "json")
        if isinstance(response_format, str) and response_format != "json":
            # For non-JSON formats, return a simplified result with just the text
            return TranscriptionResult(text=response_data)

        # For JSON or default format, parse the structured response
        return TranscriptionResult(
            text=response_data.get("text", ""),
            task=response_data.get("task"),
            language=response_data.get("language"),
            duration=response_data.get("duration"),
            segments=response_data.get("segments"),
            words=response_data.get("words"),
        )

    async def create_translation(
        self, file: Union[str, bytes, IO[bytes]], model: str = "whisper-1", **kwargs
    ) -> TranscriptionResult:
        """
        Translate audio to English text using Azure OpenAI's Whisper model.

        Args:
            file: Audio file to translate (path, bytes, or file-like object)
            model: Model/deployment name (default: whisper-1)
            **kwargs: Additional parameters

        Returns:
            Translation result
        """
        # Process the file to get binary data and filename
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data
        request_data = {
            **{k: v for k, v in kwargs.items() if k not in ["filename"]},
        }

        # Set up files dictionary for multipart upload
        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": self._guess_audio_content_type(filename),
            }
        }

        # Make the API request to translations endpoint
        response_data = await self._make_request(
            method="POST", path="/audio/translations", model=model, data=request_data, files=files
        )

        # Process the response based on the requested format
        response_format = kwargs.get("response_format", "json")
        if isinstance(response_format, str) and response_format != "json":
            # For non-JSON formats, return a simplified result with just the text
            return TranscriptionResult(text=response_data)

        # For JSON or default format, parse the structured response
        # Note: Translations are always to English
        return TranscriptionResult(
            text=response_data.get("text", ""),
            task="translation",
            language="en",  # Translations are always to English
            duration=response_data.get("duration"),
            segments=response_data.get("segments"),
            words=response_data.get("words"),
        )

    def _process_audio_file(
        self, file: Union[str, bytes, IO[bytes]], filename: Optional[str] = None
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
        # Handle different input types for the audio file
        if isinstance(file, str):
            # File path - read the file from disk
            with open(file, "rb") as f:
                file_data = f.read()
            filename = filename or file.split("/")[-1]  # Use provided or extract from path
        elif isinstance(file, bytes):
            # Bytes data
            file_data = file
            filename = filename or "audio.mp3"  # Default filename for byte data
        elif hasattr(file, "read"):
            # File-like object
            file_data = file.read() if callable(file.read) else file.read
            filename = filename or getattr(file, "name", "audio.mp3")
        else:
            # Invalid input type
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
            MIME type for the audio file (defaults to audio/mpeg if unknown)
        """
        # Map file extensions to MIME types
        mime_types = {
            ".mp3": "audio/mpeg",
            ".mp4": "audio/mp4",
            ".mpeg": "audio/mpeg",
            ".mpga": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
        }

        # Get the file extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        # Return the MIME type or a default
        return mime_types.get(ext, "audio/mpeg")

    async def create_speech(
        self, input: str, model: str = "tts-1", voice: str = "alloy", **kwargs
    ) -> bytes:
        """
        Generate audio from text using Azure OpenAI's text-to-speech models.

        Args:
            input: Text to convert to speech
            model: Model/deployment name (default: tts-1)
            voice: Voice to use (default: alloy)
            **kwargs: Additional parameters

        Returns:
            Audio data as bytes
        """
        # Validate parameters
        if not input or not isinstance(input, str):
            raise InvalidRequestError("Input text is required and must be a string")

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
            raise InvalidRequestError("Speed must be a number between 0.25 and 4.0")

        # Prepare request data
        request_data = {
            "input": input,
            "voice": voice,
            **{k: v for k, v in kwargs.items() if k in ["response_format", "speed"]},
        }

        # Make the API request with raw binary response
        return await self._make_request_raw(
            method="POST", path="/audio/speech", model=model, data=request_data
        )

    async def _make_request_raw(
        self,
        method: str,
        path: str,
        model: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """
        Make a request to the Azure OpenAI API and return raw binary data.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            model: Model/deployment name
            data: Request data
            timeout: Request timeout in seconds

        Returns:
            Raw binary response data

        Raises:
            MuxiLLMError: On API errors
        """
        # Get deployment configuration
        deployment_config = self._get_deployment_config(model)

        # Construct the full URL
        url = self._get_url(deployment_config, path)

        timeout = timeout or self.timeout
        headers = self._get_headers(deployment_config)
        body = json.dumps(data) if data else None

        async def execute_request():
            """Inner function to execute the HTTP request with error handling"""
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout,
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
                                f"Azure OpenAI API error: {error_text} "
                                f"(status code: {response.status})",
                                provider="azure",
                                status_code=response.status,
                            )

                    # Return the raw binary data
                    return await response.read()

        # Use retry mechanism for resilience
        return await retry_async(execute_request, config=self.retry_config)

    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt using Azure OpenAI's DALL-E models.

        Args:
            prompt: Text description of the desired image
            model: Model/deployment name (default: dall-e-3)
            n: Number of images to generate (default: 1)
            size: Size of the generated images (default: 1024x1024)
            **kwargs: Additional parameters

        Returns:
            Image generation result
        """
        # Validate parameters
        if not prompt or not isinstance(prompt, str):
            raise InvalidRequestError("Prompt is required and must be a string")

        # Check size based on model
        supported_sizes = {
            "dall-e-2": {"256x256", "512x512", "1024x1024"},
            "dall-e-3": {"1024x1024", "1792x1024", "1024x1792"},
        }

        # Determine which DALL-E version based on deployment name
        dalle_version = "dall-e-3" if "3" in model else "dall-e-2"

        if size not in supported_sizes[dalle_version]:
            raise InvalidRequestError(
                f"Size '{size}' is not supported for {dalle_version}. "
                f"Use one of: {', '.join(supported_sizes[dalle_version])}"
            )

        # Check n (number of images)
        # DALL-E 3 only supports n=1
        if dalle_version == "dall-e-3" and n > 1:
            raise InvalidRequestError(
                "DALL-E 3 only supports generating one image at a time (n=1)"
            )

        # For DALL-E 2, n can be between 1 and 10
        if dalle_version == "dall-e-2" and (not isinstance(n, int) or n < 1 or n > 10):
            raise InvalidRequestError(
                "For DALL-E 2, the number of images (n) must be between 1 and 10"
            )

        # Check quality (DALL-E 3 only)
        quality = kwargs.get("quality")
        if dalle_version == "dall-e-3" and quality is not None:
            supported_qualities = {"standard", "hd"}
            if quality not in supported_qualities:
                raise InvalidRequestError(
                    f"Quality '{quality}' is not supported. "
                    f"Use one of: {', '.join(supported_qualities)}"
                )

        # Check style (DALL-E 3 only)
        style = kwargs.get("style")
        if dalle_version == "dall-e-3" and style is not None:
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
            "n": n,
            "size": size,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ["quality", "style", "response_format", "user"]
            },
        }

        # Make the API request
        response_data = await self._make_request(
            method="POST", path="/images/generations", model=model, data=request_data
        )

        # Convert response to ImageGenerationResult object
        return ImageGenerationResult(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", []),
        )


# Register the Azure provider
register_provider("azure", AzureProvider)

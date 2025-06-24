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
OpenAI provider implementation for OneLLM.

This module implements the OpenAI provider adapter, supporting all OpenAI API
endpoints including chat completions, completions, embeddings, and file operations.
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


class OpenAIProvider(Provider):
    """OpenAI provider implementation."""

    # Set capability flags
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True          # GPT-4V, GPT-4 Turbo, GPT-4o support images
    audio_input_support = False    # No direct audio in chat (only via transcription)
    video_input_support = False    # No video support

    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False       # No realtime API yet

    def __init__(self, **kwargs):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: Optional API key
            organization_id: Optional organization ID
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("openai")

        # Extract credential parameters
        api_key = kwargs.pop("api_key", None)
        organization_id = kwargs.pop("organization_id", None)

        # Filter out any other credential parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["api_key", "organization_id"]
        }

        # Update non-credential configuration
        self.config.update(filtered_kwargs)

        # Apply credentials explicitly provided to the constructor
        if api_key:
            self.config["api_key"] = api_key
        if organization_id:
            self.config["organization_id"] = organization_id

        # Check for required configuration
        if not self.config.get("api_key"):
            raise AuthenticationError(
                "OpenAI API key is required. Set it via environment variable OPENAI_API_KEY "
                "or with onellm.openai_api_key = 'your-key'.",
                provider="openai",
            )

        # Store relevant configuration as instance variables
        self.api_key = self.config["api_key"]
        self.api_base = self.config["api_base"]
        self.organization_id = self.config.get("organization_id")
        self.timeout = self.config["timeout"]
        self.max_retries = self.config["max_retries"]

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests.

        Returns:
            Dict of headers
        """
        # Create standard headers with auth token
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Add organization ID if provided
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
        files: Optional[Dict[str, Any]] = None,
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
        # Construct the full URL by joining the base URL with the path
        url = f"{self.api_base}/{path.lstrip('/')}"
        # Use provided timeout or fall back to default
        timeout = timeout or self.timeout
        # Get authentication and content-type headers
        headers = self._get_headers()

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
            # OpenAI's streaming format prefixes each JSON chunk with "data: "
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
        if isinstance(error, str):
            message = error
        else:
            message = error.get("message", "Unknown error")

        # Map HTTP status codes to appropriate error types
        if status_code == 401:
            raise AuthenticationError(
                message, provider="openai", status_code=status_code
            )
        elif status_code == 403:
            raise PermissionError(message, provider="openai", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(
                message, provider="openai", status_code=status_code
            )
        elif status_code == 429:
            raise RateLimitError(message, provider="openai", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(
                message, provider="openai", status_code=status_code
            )
        elif status_code == 500:
            raise ServiceUnavailableError(
                message, provider="openai", status_code=status_code
            )
        elif status_code == 502:
            raise BadGatewayError(message, provider="openai", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="openai", status_code=status_code)
        else:
            # Generic error for unhandled status codes
            raise APIError(
                f"OpenAI API error: {message} (status code: {status_code})",
                provider="openai",
                status_code=status_code,
                error_data=error,
            )

    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
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
        # Process messages for vision models if needed
        processed_messages = self._process_messages_for_vision(messages, model)

        # Set up the request data
        data = {
            "model": model,
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
                    method="POST", path="chat/completions", data=data, stream=True
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
                method="POST", path="chat/completions", data=data
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
        # List of models known to support vision capabilities
        vision_models = {
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
        }
        # Extract the base model name for broader matching
        model_base = model.split("-")[0]
        # Check if the model is in our vision models list or has a matching base name
        model_supports_vision = (
            any(vm in model for vm in vision_models) or model_base == "gpt4o"
        )

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
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Prepare request data with all parameters
        request_data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

        if stream:
            # Use streaming API
            raw_generator = await self._make_request(
                method="POST", path="/completions", data=request_data, stream=True
            )

            # Just return the raw chunks for now - further processing can be added as needed
            return raw_generator
        else:
            # Use non-streaming API
            response_data = await self._make_request(
                method="POST", path="/completions", data=request_data
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
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Prepare request data with all parameters
        request_data = {"model": model, "input": input, **kwargs}

        # Make API request to embeddings endpoint
        response_data = await self._make_request(
            method="POST", path="/embeddings", data=request_data
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
        Upload a file to OpenAI.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file
        """
        # Prepare file data based on the input type
        if isinstance(file, str):
            # File path - read the file from disk
            with open(file, "rb") as f:
                file_data = f.read()
            filename = os.path.basename(file)  # Extract filename from path
        elif isinstance(file, bytes):
            # Bytes data - use directly
            file_data = file
            filename = kwargs.get("filename", "file.dat")  # Use provided or default name
        elif hasattr(file, "read"):
            # File-like object - read the content
            file_data = file.read()
            filename = getattr(file, "name", "file.dat")  # Try to get name from object
        else:
            # Invalid input type
            error_msg = (
                "Invalid file type. Expected file path, bytes, or file-like object."
            )
            raise InvalidRequestError(error_msg)

        # Prepare request data, excluding filename which is handled separately
        request_data = {
            "purpose": purpose,
            **{k: v for k, v in kwargs.items() if k != "filename"},
        }

        # Set up file data for multipart upload
        files = {
            "file": {
                "data": file_data,
                "filename": filename,
                "content_type": "application/octet-stream",
            }
        }

        # Make API request to files endpoint
        response_data = await self._make_request(
            method="POST", path="/files", data=request_data, files=files
        )

        # Convert API response to FileObject model
        return FileObject(
            id=response_data.get("id", ""),
            object=response_data.get("object", "file"),
            bytes=response_data.get("bytes", 0),
            created_at=response_data.get("created_at", int(time.time())),
            filename=response_data.get("filename", filename),
            purpose=response_data.get("purpose", purpose),
            status=response_data.get("status"),
            status_details=response_data.get("status_details"),
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from OpenAI.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file
        """
        # Construct the URL for file download
        url = f"{self.api_base}/files/{file_id}/content"
        # Use provided timeout or fall back to default
        timeout = kwargs.get("timeout", self.timeout)
        # Get authentication headers
        headers = self._get_headers()

        async def execute_request():
            """Inner function to execute the download request with proper error handling"""
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=url,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    # Check for error status codes
                    if response.status != 200:
                        error_data = await response.json()
                        self._handle_error_response(response.status, error_data)

                    # Return the raw file content
                    return await response.read()

        # Use retry mechanism for reliability
        return await retry_async(execute_request, config=self.retry_config)

    async def list_files(self, **kwargs) -> Dict[str, Any]:
        """
        List files available to the user.

        Args:
            **kwargs: Additional parameters like 'purpose' to filter files

        Returns:
            Dictionary containing list of files
        """
        # Extract additional parameters
        data = {}
        # Add purpose filter if provided
        if "purpose" in kwargs:
            data["purpose"] = kwargs["purpose"]

        # Make the API request to list files
        return await self._make_request(
            method="GET",
            path="/files",
            data=data
        )

    async def delete_file(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            file_id: ID of the file to delete
            **kwargs: Additional parameters

        Returns:
            Dictionary with deletion status
        """
        # Make the API request to delete the file
        return await self._make_request(
            method="DELETE",
            path=f"/files/{file_id}"
        )

    async def create_transcription(
        self, file: Union[str, bytes, IO[bytes]], model: str = "whisper-1", **kwargs
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
        # Process the file to get binary data and filename
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data, excluding filename which is handled separately
        request_data = {
            "model": model,
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
            method="POST", path="/audio/transcriptions", data=request_data, files=files
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
        # Process the file to get binary data and filename
        file_data, filename = self._process_audio_file(file, kwargs.get("filename"))

        # Prepare the request data, excluding filename which is handled separately
        request_data = {
            "model": model,
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
            method="POST", path="/audio/translations", data=request_data, files=files
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

        This helper method handles different input types for audio files and converts them
        to a consistent format for API requests. It supports file paths, byte data, and
        file-like objects.

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

        This helper method determines the appropriate MIME type for an audio file
        based on its file extension, which is required for proper multipart uploads.

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

    async def _make_request_raw(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """
        Make a request to the OpenAI API and return raw binary data.

        Unlike the standard _make_request method which returns parsed JSON,
        this method returns the raw binary response, which is necessary for
        endpoints that return non-JSON data like audio files.

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
        url = f"{self.api_base}/{path.lstrip('/')}"  # Ensure path starts without slash
        timeout = timeout or self.timeout  # Use provided timeout or default
        headers = self._get_headers()  # Get authentication and other headers
        body = json.dumps(data) if data else None  # Serialize data to JSON if provided

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
                                f"OpenAI API error: {error_text} (status code: {response.status})",
                                provider="openai",
                                status_code=response.status,
                            )

                    # Return the raw binary data
                    return await response.read()

        # Use retry mechanism for resilience
        return await retry_async(execute_request, config=self.retry_config)

    async def create_speech(
        self, input: str, model: str = "tts-1", voice: str = "alloy", **kwargs
    ) -> bytes:
        """
        Generate audio from text using OpenAI's text-to-speech models.

        This method converts text to spoken audio using OpenAI's TTS models.
        It supports different voices, speeds, and output formats.

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
            raise InvalidRequestError("Speed must be a number between 0.25 and 4.0")

        # Prepare request data
        request_data = {
            "input": input,
            "model": model,
            "voice": voice,
            **{k: v for k, v in kwargs.items() if k in ["response_format", "speed"]},
        }

        # Make the API request with raw binary response
        return await self._make_request_raw(
            method="POST", path="/audio/speech", data=request_data
        )

    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt using OpenAI's DALL-E models.

        This method creates images based on text descriptions using OpenAI's DALL-E models.
        Different models support different sizes, quality levels, and styles.

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

        # Check size based on model - different models support different sizes
        supported_sizes = {
            "dall-e-2": {"256x256", "512x512", "1024x1024"},
            "dall-e-3": {"1024x1024", "1792x1024", "1024x1792"},
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

        # Prepare request data - only include supported parameters
        request_data = {
            "prompt": prompt,
            "model": model,
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
            method="POST", path="/images/generations", data=request_data
        )

        # Convert response to ImageGenerationResult object
        return ImageGenerationResult(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", []),
        )


# Register the OpenAI provider
register_provider("openai", OpenAIProvider)

#!/usr/bin/env python3
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
Mistral AI provider implementation for OneLLM.

This module implements the Mistral AI provider adapter, supporting all OpenAI-compatible
endpoints including chat completions, completions, embeddings, and fine-tuning operations.
Mistral AI is a European AI company providing efficient, high-performance language models.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from ..config import get_provider_config
from ..errors import (
    APIError,
    AuthenticationError,
    BadGatewayError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    TimeoutError,
)
from ..models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    CompletionChoice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    FileObject,
    StreamingChoice,
)
from ..types import Message
from ..utils.retry import RetryConfig, retry_async
from .base import Provider, register_provider


class MistralProvider(Provider):
    """Mistral AI provider implementation."""

    # Set capability flags
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True  # Pixtral models support images
    audio_input_support = False  # No audio support
    video_input_support = False  # No video support

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    def __init__(self, **kwargs):
        """
        Initialize the Mistral AI provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("mistral")

        # Extract credential parameters
        api_key = kwargs.pop("api_key", None)

        # Filter out any other credential parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key"]}

        # Update non-credential configuration
        self.config.update(filtered_kwargs)

        # Apply credentials explicitly provided to the constructor
        if api_key:
            self.config["api_key"] = api_key

        # Check for required configuration
        if not self.config.get("api_key"):
            raise AuthenticationError(
                "Mistral API key is required. Set it via environment variable MISTRAL_API_KEY "
                "or with onellm.mistral_api_key = 'your-key'.",
                provider="mistral",
            )

        # Store relevant configuration as instance variables
        self.api_key = self.config["api_key"]
        self.api_base = self.config.get("api_base", "https://api.mistral.ai/v1")
        self.timeout = self.config.get("timeout", 30.0)
        self.max_retries = self.config.get("max_retries", 3)

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    def _get_headers(self) -> dict[str, str]:
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

        return headers

    async def _make_request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Make a request to the Mistral AI API.

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
                    content_type=file_info.get("content_type", "application/octet-stream"),
                )

            # Add other fields to the form
            if data:
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        # Convert complex objects to JSON strings
                        form_data.add_field(key, json.dumps(value), content_type="application/json")
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

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
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
    ) -> AsyncGenerator[dict[str, Any], None]:
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
            # Mistral's streaming format prefixes each JSON chunk with "data: "
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

    def _handle_error_response(self, status_code: int, response_data: dict[str, Any]) -> None:
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
            raise AuthenticationError(message, provider="mistral", status_code=status_code)
        elif status_code == 403:
            raise PermissionError(message, provider="mistral", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="mistral", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="mistral", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="mistral", status_code=status_code)
        elif status_code == 500:
            raise ServiceUnavailableError(message, provider="mistral", status_code=status_code)
        elif status_code == 502:
            raise BadGatewayError(message, provider="mistral", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="mistral", status_code=status_code)
        else:
            # Generic error for unhandled status codes
            raise APIError(
                f"Mistral API error: {message} (status code: {status_code})",
                provider="mistral",
                status_code=status_code,
                error_data=error,
            )

    async def create_chat_completion(
        self, messages: list[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion with Mistral AI.

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

    def _process_messages_for_vision(self, messages: list[Message], model: str) -> list[Message]:
        """
        Process messages to ensure they're compatible with vision models if needed.

        This checks for image content items and formats them correctly for the Mistral API.
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
        # List of models known to support vision capabilities (Pixtral models)
        vision_models = {
            "pixtral-12b-2409",
            "pixtral-large-latest",
        }
        # Check if the model is in our vision models list or contains "pixtral"
        model_supports_vision = model in vision_models or "pixtral" in model.lower()

        # Raise error if trying to use images with a non-vision model
        if not model_supports_vision:
            raise InvalidRequestError(
                f"Model '{model}' does not support vision inputs. "
                f"Use a vision-capable model like 'pixtral-12b-2409' or 'pixtral-large-latest'."
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
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
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
        self, input: str | list[str], model: str, **kwargs
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
        Upload a file to Mistral AI.

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
            filename = file.split("/")[-1]  # Extract filename from path
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
            error_msg = "Invalid file type. Expected file path, bytes, or file-like object."
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
        Download a file from Mistral AI.

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


# Register the Mistral provider
register_provider("mistral", MistralProvider)

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
Anthropic provider implementation for OneLLM.

This module implements the Anthropic provider adapter, supporting Claude models through
their native API format. Anthropic is known for their safety-focused approach and offers
advanced models including Claude 4, Claude 3.7, and Claude 3.5 with unique features like
extended thinking, prompt caching, and comprehensive tool use capabilities.
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
    EmbeddingResponse,
    FileObject,
    StreamingChoice,
)
from ..types import Message
from ..utils.retry import RetryConfig, retry_async
from .base import Provider, register_provider


class AnthropicProvider(Provider):
    """Anthropic provider implementation."""

    # Set capability flags
    json_mode_support = False  # Anthropic doesn't have explicit JSON mode

    # Multi-modal capabilities
    vision_support = True  # Claude models support images and PDFs
    audio_input_support = False  # No audio support
    video_input_support = False  # No video support

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    def __init__(self, **kwargs):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("anthropic")

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
                "Anthropic API key is required. Set it via environment variable ANTHROPIC_API_KEY "
                "or with onellm.anthropic_api_key = 'your-key'.",
                provider="anthropic",
            )

        # Store relevant configuration as instance variables
        self.api_key = self.config["api_key"]
        self.api_base = self.config.get("api_base", "https://api.anthropic.com/v1")
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
        # Create standard headers with Anthropic's specific auth header
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
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
        Make a request to the Anthropic API.

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
            # Anthropic's streaming format prefixes each JSON chunk with "data: "
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
            raise AuthenticationError(message, provider="anthropic", status_code=status_code)
        elif status_code == 403:
            raise PermissionError(message, provider="anthropic", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="anthropic", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="anthropic", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="anthropic", status_code=status_code)
        elif status_code == 500:
            raise ServiceUnavailableError(message, provider="anthropic", status_code=status_code)
        elif status_code == 502:
            raise BadGatewayError(message, provider="anthropic", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="anthropic", status_code=status_code)
        else:
            # Generic error for unhandled status codes
            raise APIError(
                f"Anthropic API error: {message} (status code: {status_code})",
                provider="anthropic",
                status_code=status_code,
                error_data=error,
            )

    def _convert_openai_to_anthropic_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """
        Convert OpenAI-style messages to Anthropic's native format.

        Args:
            messages: OpenAI-style messages

        Returns:
            Anthropic-style messages
        """
        anthropic_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Convert role mapping
            if role == "system":
                # Anthropic system messages go in a separate 'system' parameter
                continue
            elif role == "assistant":
                anthropic_role = "assistant"
            else:
                anthropic_role = "user"

            # Handle different content types
            if isinstance(content, str):
                # Simple text content
                anthropic_content = content
            elif isinstance(content, list):
                # Complex content with images, text, etc.
                anthropic_content = []
                for item in content:
                    if item.get("type") == "text":
                        anthropic_content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Convert OpenAI image_url format to Anthropic image format
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")

                        # Handle base64 encoded images
                        if url.startswith("data:"):
                            # Extract media type and base64 data
                            header, data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]

                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
                        else:
                            # For non-base64 URLs, we'd need to fetch and convert
                            # For now, just add as text description
                            anthropic_content.append(
                                {"type": "text", "text": f"[Image URL: {url}]"}
                            )
            else:
                # Fallback to string representation
                anthropic_content = str(content)

            anthropic_messages.append({"role": anthropic_role, "content": anthropic_content})

        return anthropic_messages

    def _extract_system_message(self, messages: list[Message]) -> str | None:
        """
        Extract system message from OpenAI-style messages.

        Args:
            messages: OpenAI-style messages

        Returns:
            System message content or None
        """
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Extract text from complex content
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    return " ".join(text_parts)
        return None

    def _convert_anthropic_to_openai_response(
        self, anthropic_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Anthropic response to OpenAI format.

        Args:
            anthropic_response: Native Anthropic response
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        # Extract content from Anthropic response
        content = anthropic_response.get("content", [])
        message_content = ""

        # Combine all text content
        for item in content:
            if item.get("type") == "text":
                message_content += item.get("text", "")

        # Create choice in OpenAI format
        choice = Choice(
            message={"role": "assistant", "content": message_content},
            finish_reason=anthropic_response.get("stop_reason", "stop"),
            index=0,
        )

        # Create usage information
        usage = None
        if "usage" in anthropic_response:
            usage = {
                "prompt_tokens": anthropic_response["usage"].get("input_tokens", 0),
                "completion_tokens": anthropic_response["usage"].get("output_tokens", 0),
                "total_tokens": (
                    anthropic_response["usage"].get("input_tokens", 0)
                    + anthropic_response["usage"].get("output_tokens", 0)
                ),
            }

        # Create the response object
        return ChatCompletionResponse(
            id=anthropic_response.get("id", ""),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
        )

    async def create_chat_completion(
        self, messages: list[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion with Anthropic.

        Args:
            messages: List of messages in the conversation
            model: Model name (without provider prefix)
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Convert OpenAI messages to Anthropic format
        anthropic_messages = self._convert_openai_to_anthropic_messages(messages)
        system_message = self._extract_system_message(messages)

        # Set up the request data in Anthropic's native format
        data = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),  # Required parameter
            "stream": stream,
        }

        # Add system message if present
        if system_message:
            data["system"] = system_message

        # Add other supported parameters
        if "temperature" in kwargs:
            data["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            data["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        # Handle unique Anthropic features
        if "thinking" in kwargs:
            data["thinking"] = kwargs["thinking"]

        # Make the request
        if stream:
            # Handle streaming response
            async def chunk_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
                """Generator function to process streaming chunks"""
                async for chunk in await self._make_request(
                    method="POST", path="messages", data=data, stream=True
                ):
                    if chunk:
                        # Convert Anthropic streaming format to OpenAI format
                        if chunk.get("type") == "content_block_delta":
                            delta_data = chunk.get("delta", {})
                            if delta_data.get("type") == "text_delta":
                                # Create a ChoiceDelta object
                                delta = ChoiceDelta(
                                    content=delta_data.get("text", ""),
                                    role=None,
                                    function_call=None,
                                    tool_calls=None,
                                    finish_reason=None,
                                )
                                # Create a StreamingChoice object
                                choice = StreamingChoice(
                                    delta=delta,
                                    finish_reason=None,
                                    index=0,
                                )

                                # Create the chunk response object
                                chunk_resp = ChatCompletionChunk(
                                    id=chunk.get("id", ""),
                                    object="chat.completion.chunk",
                                    created=int(time.time()),
                                    model=model,
                                    choices=[choice],
                                    system_fingerprint=None,
                                )
                                yield chunk_resp
                        elif chunk.get("type") == "message_stop":
                            # Send final chunk with finish_reason
                            delta = ChoiceDelta(
                                content=None,
                                role=None,
                                function_call=None,
                                tool_calls=None,
                                finish_reason="stop",
                            )
                            choice = StreamingChoice(
                                delta=delta,
                                finish_reason="stop",
                                index=0,
                            )

                            chunk_resp = ChatCompletionChunk(
                                id=chunk.get("id", ""),
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[choice],
                                system_fingerprint=None,
                            )
                            yield chunk_resp

            return chunk_generator()
        else:
            # Handle non-streaming response
            response_data = await self._make_request(method="POST", path="messages", data=data)

            # Convert Anthropic response to OpenAI format
            return self._convert_anthropic_to_openai_response(response_data, model)

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion.

        Note: Anthropic doesn't have a direct completion endpoint, so we convert
        this to a chat completion with the prompt as a user message.

        Args:
            prompt: Text prompt to complete
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Convert completion to chat completion
        messages = [{"role": "user", "content": prompt}]

        if stream:
            # Handle streaming case
            async def completion_generator():
                async for chunk in await self.create_chat_completion(
                    messages, model, stream=True, **kwargs
                ):
                    # Convert chat completion chunk to completion format
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {
                            "id": chunk.id,
                            "object": "text_completion",
                            "created": chunk.created,
                            "model": chunk.model,
                            "choices": [
                                {
                                    "text": chunk.choices[0].delta.content,
                                    "index": 0,
                                    "finish_reason": chunk.choices[0].finish_reason,
                                }
                            ],
                        }

            return completion_generator()
        else:
            # Handle non-streaming case
            chat_response = await self.create_chat_completion(
                messages, model, stream=False, **kwargs
            )

            # Convert chat completion to text completion
            choice = CompletionChoice(
                text=chat_response.choices[0].message.get("content", ""),
                index=0,
                logprobs=None,
                finish_reason=chat_response.choices[0].finish_reason,
            )

            return CompletionResponse(
                id=chat_response.id,
                object="text_completion",
                created=chat_response.created,
                model=chat_response.model,
                choices=[choice],
                usage=chat_response.usage,
                system_fingerprint=chat_response.system_fingerprint,
            )

    async def create_embedding(
        self, input: str | list[str], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input.

        Note: Anthropic doesn't provide embedding models, so this will raise an error.

        Args:
            input: Text or list of texts to embed
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings

        Raises:
            InvalidRequestError: Anthropic doesn't support embeddings
        """
        raise InvalidRequestError(
            "Anthropic does not provide embedding models. "
            "Use a different provider like OpenAI for embeddings.",
            provider="anthropic",
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to Anthropic.

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
        Download a file from Anthropic.

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


# Register the Anthropic provider
register_provider("anthropic", AnthropicProvider)

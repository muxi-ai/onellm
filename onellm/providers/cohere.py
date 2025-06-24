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
Cohere provider implementation for OneLLM.

This module implements the Cohere provider adapter, supporting their native API
for text generation, embeddings, and reranking. Cohere specializes in enterprise
NLP with advanced RAG capabilities and multilingual support.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

import aiohttp

from ..config import get_provider_config
from ..errors import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ResourceNotFoundError,
    ServiceUnavailableError,
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
from ..utils.retry import RetryConfig
from .base import Provider, register_provider


class CohereProvider(Provider):
    """Cohere provider implementation."""

    # Set capability flags
    json_mode_support = False  # No explicit JSON mode

    # Multi-modal capabilities
    vision_support = False  # Currently no vision support in chat
    audio_input_support = False  # No audio support
    video_input_support = False  # No video support

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    # Additional capabilities
    function_calling_support = True  # Advanced tool use support

    def __init__(self, **kwargs):
        """
        Initialize the Cohere provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        # Get configuration
        self.config = get_provider_config("cohere")

        # Extract credential parameters
        api_key = kwargs.pop("api_key", None)

        # Update configuration
        self.config.update(kwargs)

        # Apply credentials explicitly provided
        if api_key:
            self.config["api_key"] = api_key

        # Check for required configuration
        if not self.config.get("api_key"):
            raise AuthenticationError(
                "Cohere API key is required. Set it via environment variable COHERE_API_KEY "
                "or with onellm.cohere_api_key = 'your-key'.",
                provider="cohere",
            )

        # Store configuration
        self.api_key = self.config["api_key"]
        self.api_base = self.config.get("api_base", "https://api.cohere.com/v2")
        self.timeout = self.config.get("timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Dict of headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "onellm",
        }

    def _convert_messages_to_cohere(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI-style messages to Cohere format.

        Cohere API v2 format:
        - System messages become system parameter
        - User/assistant messages become messages array

        Args:
            messages: OpenAI-style messages

        Returns:
            Tuple of (system, messages)
        """
        system = None
        cohere_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages become system parameter
                if system:
                    system += "\n\n" + content
                else:
                    system = content
            else:
                # Convert role names
                cohere_role = "user" if role == "user" else "assistant"

                # Handle content
                if isinstance(content, str):
                    cohere_messages.append({"role": cohere_role, "content": content})
                elif isinstance(content, list):
                    # Extract text content
                    text_content = ""
                    for item in content:
                        if item.get("type") == "text":
                            text_content += item.get("text", "") + "\n"

                    if text_content:
                        cohere_messages.append(
                            {"role": cohere_role, "content": text_content.strip()}
                        )
                else:
                    cohere_messages.append({"role": cohere_role, "content": str(content)})

        return system, cohere_messages

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        """
        Make a request to the Cohere API.

        Args:
            method: HTTP method
            path: API path
            data: Request data
            stream: Whether to stream the response
            timeout: Request timeout

        Returns:
            Response data or async generator for streaming
        """
        url = f"{self.api_base}/{path}"
        headers = self._get_headers()

        timeout_obj = aiohttp.ClientTimeout(total=timeout or self.timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                ) as response:
                    if response.status not in (200, 201):
                        try:
                            error_data = await response.json()
                        except Exception:
                            error_data = {"message": await response.text()}
                        self._handle_error_response(response.status, error_data)

                    if stream:
                        return self._handle_streaming_response(response)
                    else:
                        return await response.json()

            except aiohttp.ClientError as e:
                raise ServiceUnavailableError(
                    f"Failed to connect to Cohere API: {str(e)}", provider="cohere"
                )

    async def _handle_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a streaming API response.

        Args:
            response: API response

        Yields:
            Parsed JSON chunks
        """
        async for line in response.content:
            line = line.decode("utf-8").strip()
            if line:
                try:
                    # Cohere sends JSON objects directly
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _handle_error_response(self, status_code: int, response_data: Dict[str, Any]) -> None:
        """
        Handle an error response.

        Args:
            status_code: HTTP status code
            response_data: Error response data

        Raises:
            Appropriate error based on status code
        """
        message = response_data.get("message", "Unknown error")

        if status_code == 401:
            raise AuthenticationError(message, provider="cohere", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="cohere", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="cohere", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="cohere", status_code=status_code)
        else:
            raise APIError(
                f"Cohere API error: {message} (status code: {status_code})",
                provider="cohere",
                status_code=status_code,
            )

    def _convert_cohere_to_openai_response(
        self, cohere_response: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Cohere response to OpenAI format.

        Args:
            cohere_response: Response from Cohere API
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        # Extract message content
        message = cohere_response.get("message", {})
        content = message.get("content", [])

        # Combine content blocks
        text = ""
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text += block.get("text", "")
        elif isinstance(content, str):
            text = content

        # Get finish reason
        finish_reason = cohere_response.get("finish_reason", "complete")
        if finish_reason == "complete":
            finish_reason = "stop"
        elif finish_reason == "max_tokens":
            finish_reason = "length"

        choice = Choice(
            message={"role": "assistant", "content": text},
            finish_reason=finish_reason,
            index=0,
        )

        # Get token usage
        usage = cohere_response.get("usage", {})
        usage_dict = {
            "prompt_tokens": usage.get("billed_units", {}).get("input_tokens", 0),
            "completion_tokens": usage.get("billed_units", {}).get("output_tokens", 0),
            "total_tokens": 0,
        }
        usage_dict["total_tokens"] = usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]

        return ChatCompletionResponse(
            id=cohere_response.get("id", f"cohere-{int(time.time())}"),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage_dict,
            system_fingerprint=None,
        )

    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion with Cohere.

        Args:
            messages: List of messages
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ChatCompletionResponse or async generator
        """
        # Convert messages to Cohere format
        system, cohere_messages = self._convert_messages_to_cohere(messages)

        # Build request data
        data = {"model": model, "messages": cohere_messages}

        # Only add stream parameter when streaming
        if stream:
            data["stream"] = True

        # Add system message if present
        if system:
            data["system"] = system

        # Add optional parameters
        if "max_tokens" in kwargs:
            data["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            data["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            data["p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            data["k"] = kwargs["top_k"]
        if "stop" in kwargs:
            data["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        # Make request
        if stream:
            # Handle streaming
            async def stream_generator():
                response_text = ""
                async for chunk in await self._make_request(
                    method="POST",
                    path="chat",
                    data=data,
                    stream=True,
                ):
                    event_type = chunk.get("type")

                    if event_type == "content-delta":
                        delta = chunk.get("delta", {})
                        message = delta.get("message", {})
                        content = message.get("content", {})

                        if content.get("type") == "text":
                            text = content.get("text", "")
                            response_text += text

                            delta_obj = ChoiceDelta(
                                content=text,
                                role=None,
                                function_call=None,
                                tool_calls=None,
                            )

                            choice = StreamingChoice(
                                delta=delta_obj,
                                finish_reason=None,
                                index=0,
                            )

                            chunk_obj = ChatCompletionChunk(
                                id=f"cohere-{int(time.time())}",
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[choice],
                                system_fingerprint=None,
                            )

                            yield chunk_obj

                    elif event_type == "message-end":
                        # Final chunk with finish reason
                        delta_obj = ChoiceDelta(
                            content=None,
                            role=None,
                            function_call=None,
                            tool_calls=None,
                        )

                        choice = StreamingChoice(
                            delta=delta_obj,
                            finish_reason="stop",
                            index=0,
                        )

                        chunk_obj = ChatCompletionChunk(
                            id=f"cohere-{int(time.time())}",
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[choice],
                            system_fingerprint=None,
                        )

                        yield chunk_obj

            return stream_generator()
        else:
            # Non-streaming request
            response = await self._make_request(
                method="POST",
                path="chat",
                data=data,
            )

            return self._convert_cohere_to_openai_response(response, model)

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion.

        Args:
            prompt: Text prompt
            model: Model name
            stream: Whether to stream
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator
        """
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]

        # Use chat completion
        if stream:

            async def completion_generator():
                async for chunk in await self.create_chat_completion(
                    messages, model, stream=True, **kwargs
                ):
                    # Convert chat chunk to completion format
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
            chat_response = await self.create_chat_completion(
                messages, model, stream=False, **kwargs
            )

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
                system_fingerprint=None,
            )

    async def create_embedding(
        self, input: str | List[str], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings.

        Args:
            input: Text or list of texts
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse
        """
        # Ensure input is a list
        texts = [input] if isinstance(input, str) else input

        # Build request data
        data = {
            "model": model,
            "texts": texts,
            "input_type": kwargs.get("input_type", "search_document"),
            "embedding_types": ["float"],
        }

        # Make request
        response = await self._make_request(
            method="POST",
            path="embed",
            data=data,
        )

        # Extract embeddings
        embeddings_data = response.get("embeddings", {})
        float_embeddings = embeddings_data.get("float", [])

        embeddings = []
        for i, embedding in enumerate(float_embeddings):
            embedding_obj = EmbeddingData(
                object="embedding",
                embedding=embedding,
                index=i,
            )
            embeddings.append(embedding_obj)

        # Get token usage
        meta = response.get("meta", {})
        usage = meta.get("billed_units", {})

        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=model,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0),
            },
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file.

        Note: Cohere doesn't support file uploads.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject

        Raises:
            InvalidRequestError: Not supported
        """
        raise InvalidRequestError(
            "File upload is not supported by Cohere. " "Use text content directly in messages.",
            provider="cohere",
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file.

        Args:
            file_id: ID of the file
            **kwargs: Additional parameters

        Returns:
            File bytes

        Raises:
            InvalidRequestError: Not supported
        """
        raise InvalidRequestError("File download is not supported by Cohere.", provider="cohere")


# Register the provider
register_provider("cohere", CohereProvider)

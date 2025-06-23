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


class CohereProvider(Provider):
    """Cohere provider implementation."""

    # Set capability flags
    json_mode_support = False  # No explicit JSON mode

    # Multi-modal capabilities
    vision_support = True          # Aya Vision model supports images
    audio_input_support = False    # No audio support
    video_input_support = False    # No video support

    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = True  # Advanced tool use support

    def __init__(self, **kwargs):
        """
        Initialize the Cohere provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("cohere")

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
                "Cohere API key is required. Set it via environment variable COHERE_API_KEY "
                "or with onellm.cohere_api_key = 'your-key'.",
                provider="cohere",
            )

        # Store relevant configuration as instance variables
        self.api_key = self.config["api_key"]
        self.api_base = self.config.get("api_base", "https://api.cohere.com")
        self.timeout = self.config.get("timeout", 60.0)
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
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Client-Name": "onellm"
        }

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
        Make a request to the Cohere API.

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
        # Construct the full URL
        url = f"{self.api_base}/{path.lstrip('/')}"
        timeout = timeout or self.timeout
        headers = self._get_headers()

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
                    ssl=None,  # Use default SSL settings
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
            if line:
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
        message = response_data.get("message", "Unknown error")

        # Map HTTP status codes to appropriate error types
        if status_code == 401:
            raise AuthenticationError(message, provider="cohere", status_code=status_code)
        elif status_code == 403:
            raise PermissionError(message, provider="cohere", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="cohere", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="cohere", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="cohere", status_code=status_code)
        elif status_code == 500:
            raise ServiceUnavailableError(message, provider="cohere", status_code=status_code)
        elif status_code == 502:
            raise BadGatewayError(message, provider="cohere", status_code=status_code)
        elif status_code == 504:
            raise TimeoutError(message, provider="cohere", status_code=status_code)
        else:
            # Generic error for unhandled status codes
            raise APIError(
                f"Cohere API error: {message} (status code: {status_code})",
                provider="cohere",
                status_code=status_code,
                error_data=response_data,
            )

    def _convert_openai_to_cohere_messages(self, messages: list[Message]) -> tuple[list[dict], str | None]:
        """
        Convert OpenAI-style messages to Cohere's format.

        Args:
            messages: OpenAI-style messages

        Returns:
            Tuple of (cohere_messages, system_prompt)
        """
        cohere_messages = []
        system_prompt = None
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Cohere uses a separate system parameter
                system_prompt = content
            elif role == "assistant":
                cohere_messages.append({
                    "role": "assistant",
                    "message": content
                })
            else:  # user
                cohere_messages.append({
                    "role": "user",
                    "message": content
                })
        
        return cohere_messages, system_prompt

    def _convert_cohere_to_openai_response(
        self, cohere_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Cohere response to OpenAI format.

        Args:
            cohere_response: Native Cohere response
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        # Extract text from Cohere response
        text = cohere_response.get("text", "")
        
        # Create choice in OpenAI format
        choice = Choice(
            message={
                "role": "assistant",
                "content": text
            },
            finish_reason=cohere_response.get("finish_reason", "complete"),
            index=0,
        )
        
        # Create usage information
        usage = None
        if "usage" in cohere_response:
            usage = {
                "prompt_tokens": cohere_response["usage"].get("input_tokens", 0),
                "completion_tokens": cohere_response["usage"].get("output_tokens", 0),
                "total_tokens": cohere_response["usage"].get("total_tokens", 0)
            }
        
        # Create the response object
        return ChatCompletionResponse(
            id=cohere_response.get("id", f"cohere-{int(time.time())}"),
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
        Create a chat completion with Cohere.

        Args:
            messages: List of messages in the conversation
            model: Model name (without provider prefix)
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Convert OpenAI messages to Cohere format
        cohere_messages, system_prompt = self._convert_openai_to_cohere_messages(messages)
        
        # Set up the request data in Cohere's format
        data = {
            "model": model,
            "messages": cohere_messages,
            "stream": stream,
        }
        
        # Add system prompt if present
        if system_prompt:
            data["system"] = system_prompt
        
        # Add other supported parameters
        if "temperature" in kwargs:
            data["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            data["max_tokens"] = kwargs["max_tokens"]
        if "stop" in kwargs:
            data["stop_sequences"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
        
        # Make the request
        if stream:
            # Handle streaming response
            async def chunk_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
                """Generator function to process streaming chunks"""
                async for chunk in await self._make_request(
                    method="POST", path="v2/chat", data=data, stream=True
                ):
                    if chunk:
                        # Convert Cohere streaming format to OpenAI format
                        event_type = chunk.get("event_type")
                        
                        if event_type == "text-generation":
                            # Create a ChoiceDelta object
                            delta = ChoiceDelta(
                                content=chunk.get("text", ""),
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
                        elif event_type == "stream-end":
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
            response_data = await self._make_request(
                method="POST", path="v2/chat", data=data
            )

            # Convert Cohere response to OpenAI format
            return self._convert_cohere_to_openai_response(response_data, model)

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion.

        Note: Cohere's v2 API primarily uses chat format, so we convert
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
                            "choices": [{
                                "text": chunk.choices[0].delta.content,
                                "index": 0,
                                "finish_reason": chunk.choices[0].finish_reason,
                            }]
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

        Args:
            input: Text or list of texts to embed
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Ensure input is a list
        texts = input if isinstance(input, list) else [input]
        
        # Prepare request data
        request_data = {
            "model": model,
            "texts": texts,
            **{k: v for k, v in kwargs.items() if k in ["input_type", "truncate"]}
        }

        # Make API request to embeddings endpoint
        response_data = await self._make_request(
            method="POST", path="v2/embed", data=request_data
        )

        # Convert API response to EmbeddingResponse model
        embedding_data = []
        for idx, embedding in enumerate(response_data.get("embeddings", [])):
            embed = EmbeddingData(
                embedding=embedding,
                index=idx,
                object="embedding",
            )
            embedding_data.append(embed)

        # Create and return the structured response object
        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=response_data.get("model", model),
            usage=response_data.get("usage"),
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to Cohere.

        Note: Cohere doesn't have a general file upload API like OpenAI.
        This method is provided for compatibility but will raise an error.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: Cohere doesn't support file uploads
        """
        raise InvalidRequestError(
            "Cohere does not support file uploads through the API. "
            "Files should be processed locally and sent as part of the request.",
            provider="cohere"
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from Cohere.

        Note: Cohere doesn't have a file storage API.
        This method is provided for compatibility but will raise an error.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file

        Raises:
            InvalidRequestError: Cohere doesn't support file downloads
        """
        raise InvalidRequestError(
            "Cohere does not support file downloads through the API.",
            provider="cohere"
        )


# Register the Cohere provider
register_provider("cohere", CohereProvider)
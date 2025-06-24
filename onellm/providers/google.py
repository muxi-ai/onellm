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
Google AI Studio (Gemini) provider implementation for OneLLM.

Google AI Studio provides access to Gemini models through their native API.
This is different from Vertex AI which is Google Cloud's enterprise offering.
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


class GoogleProvider(Provider):
    """Google AI Studio (Gemini) provider implementation."""

    # Provider configuration
    provider_name = "google"

    # Set capability flags
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True  # Gemini models support vision
    audio_input_support = True  # Gemini models support audio
    video_input_support = True  # Gemini models support video

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    # Additional capabilities
    function_calling_support = True  # Supports function calling

    # Google-specific features
    thinking_mode_support = True  # Gemini 2.0 Flash supports thinking mode

    def __init__(self, **kwargs):
        """
        Initialize the Google provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        # Get configuration
        self.config = get_provider_config("google")

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
                "Google API key is required. Set it via environment variable GOOGLE_API_KEY "
                "or with onellm.google_api_key = 'your-key'.",
                provider="google",
            )

        # Store configuration
        self.api_key = self.config["api_key"]
        self.api_base = self.config.get(
            "api_base", "https://generativelanguage.googleapis.com/v1beta"
        )
        self.timeout = self.config.get("timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    def _convert_messages_to_gemini(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses a different format where:
        - System messages become system_instruction
        - User/assistant messages become contents with parts

        Args:
            messages: OpenAI-style messages

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages become system_instruction
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            else:
                # Convert to Gemini format
                gemini_role = "user" if role == "user" else "model"

                # Handle different content types
                if isinstance(content, str):
                    parts = [{"text": content}]
                elif isinstance(content, list):
                    # Handle multi-modal content
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image content
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            # Google expects base64 data or GCS URLs
                            if url.startswith("data:"):
                                # Extract base64 data
                                mime_type, data = url.split(",", 1)
                                mime_type = mime_type.split(";")[0].split(":")[1]
                                parts.append(
                                    {"inline_data": {"mime_type": mime_type, "data": data}}
                                )
                            else:
                                parts.append({"file_data": {"file_uri": url}})
                else:
                    parts = [{"text": str(content)}]

                contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        """
        Make a request to the Google API.

        Args:
            method: HTTP method
            path: API path
            data: Request data
            stream: Whether to stream the response
            timeout: Request timeout

        Returns:
            Response data or async generator for streaming
        """
        # Build URL without API key
        url = f"{self.api_base}/{path}"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,  # Pass API key in header instead
        }

        timeout_obj = aiohttp.ClientTimeout(total=timeout or self.timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        self._handle_error_response(response.status, error_data)

                    if stream:
                        return self._handle_streaming_response(response)
                    else:
                        return await response.json()

            except aiohttp.ClientError as e:
                raise ServiceUnavailableError(
                    f"Failed to connect to Google API: {str(e)}", provider="google"
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
            if line and line.startswith("data: "):
                line = line[6:]  # Remove 'data: ' prefix

                if line == "[DONE]":
                    break

                try:
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
        error = response_data.get("error", {})
        message = error.get("message", "Unknown error")

        if status_code == 401:
            raise AuthenticationError(message, provider="google", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="google", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="google", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="google", status_code=status_code)
        else:
            raise APIError(
                f"Google API error: {message} (status code: {status_code})",
                provider="google",
                status_code=status_code,
            )

    def _convert_gemini_to_openai_response(
        self, gemini_response: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Gemini response to OpenAI format.

        Args:
            gemini_response: Response from Gemini API
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        candidates = gemini_response.get("candidates", [])

        choices = []
        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Combine all text parts
            text = ""
            for part in parts:
                if "text" in part:
                    text += part["text"]

            # Get finish reason
            finish_reason = candidate.get("finishReason", "stop").lower()
            if finish_reason == "stop":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"
            elif finish_reason == "safety":
                finish_reason = "content_filter"

            choice = Choice(
                message={"role": "assistant", "content": text},
                finish_reason=finish_reason,
                index=i,
            )
            choices.append(choice)

        # Get token usage if available
        usage_metadata = gemini_response.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

        return ChatCompletionResponse(
            id=f"gemini-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
            system_fingerprint=None,
        )

    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a chat completion with Google Gemini.

        Args:
            messages: List of messages
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ChatCompletionResponse or async generator
        """
        # Convert messages to Gemini format
        system_instruction, contents = self._convert_messages_to_gemini(messages)

        # Build request data
        data = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 1.0),
                "topP": kwargs.get("top_p", 1.0),
                "topK": kwargs.get("top_k", 40),
            },
        }

        # Add system instruction if present
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Add response format if specified
        if kwargs.get("response_format", {}).get("type") == "json_object":
            data["generationConfig"]["responseMimeType"] = "application/json"

        # Determine endpoint
        endpoint = f"models/{model}:generateContent"
        if stream:
            endpoint = f"models/{model}:streamGenerateContent"
            data["generationConfig"]["candidateCount"] = 1

        # Make request
        if stream:
            # Handle streaming
            async def stream_generator():
                async for chunk in await self._make_request(
                    method="POST",
                    path=endpoint,
                    data=data,
                    stream=True,
                ):
                    # Convert Gemini streaming format to OpenAI format
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        candidate = candidates[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])

                        for part in parts:
                            if "text" in part:
                                delta = ChoiceDelta(
                                    content=part["text"],
                                    role=None,
                                    function_call=None,
                                    tool_calls=None,
                                )

                                choice = StreamingChoice(
                                    delta=delta,
                                    finish_reason=None,
                                    index=0,
                                )

                                chunk_obj = ChatCompletionChunk(
                                    id=f"gemini-{int(time.time())}",
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
                path=endpoint,
                data=data,
            )

            return self._convert_gemini_to_openai_response(response, model)

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

        embeddings = []

        for i, text in enumerate(texts):
            # Build request data
            data = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}

            # Make request
            response = await self._make_request(
                method="POST",
                path=f"models/{model}:embedContent",
                data=data,
            )

            # Extract embedding
            embedding_data = response.get("embedding", {})
            values = embedding_data.get("values", [])

            embedding = EmbeddingData(
                object="embedding",
                embedding=values,
                index=i,
            )
            embeddings.append(embedding)

        # Calculate token usage (approximate)
        total_tokens = sum(len(text.split()) for text in texts)

        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file.

        Note: Google uses a different file upload mechanism.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject

        Raises:
            InvalidRequestError: Not supported in this implementation
        """
        raise InvalidRequestError(
            "File upload is not directly supported for Google AI Studio. "
            "Use base64 encoded images in messages instead.",
            provider="google",
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
        raise InvalidRequestError(
            "File download is not supported for Google AI Studio.", provider="google"
        )


# Register the provider
register_provider("google", GoogleProvider)

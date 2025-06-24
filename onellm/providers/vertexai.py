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
Vertex AI provider implementation for OneLLM.

This module implements the Vertex AI provider adapter, supporting Google's
enterprise AI platform with Gemini and other foundation models.
"""

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    import aiohttp

    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

from ..config import get_provider_config
from ..errors import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    InvalidConfigurationError,
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


class VertexAIProvider(Provider):
    """Vertex AI provider implementation."""

    # Set capability flags
    json_mode_support = True  # Via response_mime_type

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

    def __init__(self, **kwargs):
        """
        Initialize the Vertex AI provider.

        Args:
            service_account_json: Path to service account JSON file
            project_id: Google Cloud project ID
            location: Google Cloud region (default: us-central1)
            **kwargs: Additional configuration options
        """
        if not VERTEX_AI_AVAILABLE:
            raise InvalidConfigurationError(
                "Vertex AI provider requires google-auth library. "
                "Install it with: pip install google-auth google-auth-httplib2",
                provider="vertexai",
            )

        # Get configuration
        self.config = get_provider_config("vertexai")

        # Extract credential parameters
        service_account_json = kwargs.pop("service_account_json", None)
        project_id = kwargs.pop("project_id", None)
        location = kwargs.pop("location", None)

        # Update configuration
        self.config.update(kwargs)

        # Apply credentials explicitly provided
        if service_account_json:
            self.config["service_account_json"] = service_account_json
        if project_id:
            self.config["project_id"] = project_id
        if location:
            self.config["location"] = location

        # Check for required configuration
        if not self.config.get("service_account_json"):
            raise AuthenticationError(
                "Vertex AI service account JSON is required. "
                "Set it via environment variable GOOGLE_APPLICATION_CREDENTIALS "
                "or provide service_account_json parameter.",
                provider="vertexai",
            )

        # Load service account data
        self.service_account_path = self.config["service_account_json"]
        if os.path.exists(self.service_account_path):
            with open(self.service_account_path, "r") as f:
                self.service_account_data = json.load(f)
                # Extract project ID from service account if not provided
                if not self.config.get("project_id"):
                    self.config["project_id"] = self.service_account_data.get("project_id")
        else:
            raise AuthenticationError(
                f"Service account file not found: {self.service_account_path}",
                provider="vertexai",
            )

        # Validate project ID
        if not self.config.get("project_id"):
            raise AuthenticationError(
                "Project ID is required for Vertex AI. "
                "Set it via project_id parameter or ensure it's in the service account JSON.",
                provider="vertexai",
            )

        # Store configuration
        self.project_id = self.config["project_id"]
        self.location = self.config.get("location", "us-central1")
        self.timeout = self.config.get("timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)

        # Create credentials
        self.credentials = service_account.Credentials.from_service_account_info(
            self.service_account_data, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # API endpoint
        self.api_base = f"https://{self.location}-aiplatform.googleapis.com/v1"

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    async def _get_access_token(self) -> str:
        """
        Get access token for authentication.

        Returns:
            Access token string
        """
        # Refresh token if needed
        if not self.credentials.valid:
            request = Request()
            self.credentials.refresh(request)

        return self.credentials.token

    def _convert_messages_to_vertex(
        self, messages: List[Message]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert OpenAI-style messages to Vertex AI format.

        Args:
            messages: OpenAI-style messages

        Returns:
            Tuple of (vertex_contents, system_instruction)
        """
        vertex_contents = []
        system_instruction = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Vertex AI uses a separate systemInstruction field
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            else:
                # Convert role names
                vertex_role = "model" if role == "assistant" else "user"

                # Handle different content types
                if isinstance(content, str):
                    vertex_contents.append({"role": vertex_role, "parts": [{"text": content}]})
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

                            # Handle base64 images
                            if url.startswith("data:"):
                                mime_type, data = url.split(",", 1)
                                mime_type = mime_type.split(";")[0].split(":")[1]
                                parts.append(
                                    {"inline_data": {"mime_type": mime_type, "data": data}}
                                )
                            else:
                                # GCS URLs
                                parts.append({"file_data": {"file_uri": url}})

                    if parts:
                        vertex_contents.append({"role": vertex_role, "parts": parts})
                else:
                    vertex_contents.append({"role": vertex_role, "parts": [{"text": str(content)}]})

        return vertex_contents, system_instruction

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        """
        Make a request to the Vertex AI API.

        Args:
            method: HTTP method
            path: API path
            data: Request data
            stream: Whether to stream the response
            timeout: Request timeout

        Returns:
            Response data or async generator for streaming
        """
        # Get access token
        token = await self._get_access_token()

        # Build URL
        url = f"{self.api_base}/{path}"

        # Headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
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
                    f"Failed to connect to Vertex AI: {str(e)}", provider="vertexai"
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
            raise AuthenticationError(message, provider="vertexai", status_code=status_code)
        elif status_code == 404:
            raise ResourceNotFoundError(message, provider="vertexai", status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, provider="vertexai", status_code=status_code)
        elif status_code == 400:
            raise InvalidRequestError(message, provider="vertexai", status_code=status_code)
        else:
            raise APIError(
                f"Vertex AI error: {message} (status code: {status_code})",
                provider="vertexai",
                status_code=status_code,
            )

    def _convert_vertex_to_openai_response(
        self, vertex_response: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Vertex AI response to OpenAI format.

        Args:
            vertex_response: Response from Vertex AI
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        candidates = vertex_response.get("candidates", [])

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
            finish_reason = candidate.get("finishReason", "STOP").lower()
            if finish_reason == "stop":
                finish_reason = "stop"
            elif finish_reason in ["max_tokens", "token_limit"]:
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
        usage_metadata = vertex_response.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

        return ChatCompletionResponse(
            id=f"vertex-{int(time.time())}",
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
        Create a chat completion with Vertex AI.

        Args:
            messages: List of messages in the conversation
            model: Model name (e.g., 'gemini-pro', 'gemini-1.5-pro')
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or async generator of chunks
        """
        # Strip provider prefix if present
        if "/" in model:
            _, model = model.split("/", 1)

        # Convert messages to Vertex format
        vertex_contents, system_instruction = self._convert_messages_to_vertex(messages)

        # Build request
        data = {
            "contents": vertex_contents,
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

        # Build endpoint path
        endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model}:generateContent"  # noqa: E501
        if stream:
            endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model}:streamGenerateContent"  # noqa: E501

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
                    # Convert Vertex streaming format to OpenAI format
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
                                    id=f"vertex-{int(time.time())}",
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

            return self._convert_vertex_to_openai_response(response, model)

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion with Vertex AI.

        Args:
            prompt: Text prompt to complete
            model: Model name (e.g., 'text-bison')
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Strip provider prefix if present
        if "/" in model:
            _, model = model.split("/", 1)

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
        Create embeddings with Vertex AI.

        Args:
            input: Text or list of texts to embed
            model: Model name (e.g., 'text-embedding-004')
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Strip provider prefix if present
        if "/" in model:
            _, model = model.split("/", 1)

        # Ensure input is a list
        texts = [input] if isinstance(input, str) else input

        embeddings = []

        for i, text in enumerate(texts):
            # Build request
            data = {"instances": [{"content": text}]}

            # Build endpoint
            endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model}:predict"  # noqa: E501

            # Make request
            response = await self._make_request(
                method="POST",
                path=endpoint,
                data=data,
            )

            # Extract embeddings
            predictions = response.get("predictions", [])
            if predictions:
                embedding_values = predictions[0].get("embeddings", {}).get("values", [])

                embedding = EmbeddingData(
                    object="embedding",
                    embedding=embedding_values,
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
        Upload a file to Vertex AI.

        Note: Vertex AI doesn't have a direct file upload API like OpenAI.
        Files are typically uploaded to GCS.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject

        Raises:
            InvalidRequestError: Not directly supported
        """
        raise InvalidRequestError(
            "Direct file upload is not supported by Vertex AI. "
            "Upload files to Google Cloud Storage and use GCS URLs.",
            provider="vertexai",
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from Vertex AI.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file

        Raises:
            InvalidRequestError: Not supported
        """
        raise InvalidRequestError(
            "File download is not supported by Vertex AI.", provider="vertexai"
        )


# Register the provider
register_provider("vertexai", VertexAIProvider)

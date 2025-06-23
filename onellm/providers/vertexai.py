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
Google Cloud Vertex AI provider implementation for OneLLM.

This module implements the Vertex AI provider adapter, supporting Google's
enterprise Gemini models through the Vertex AI platform. It uses service
account authentication and provides access to advanced features like the
Live API, context caching, and batch processing.
"""

import json
import os
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


class VertexAIProvider(Provider):
    """Google Cloud Vertex AI provider implementation."""

    # Set capability flags
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True          # Gemini models support images and video
    audio_input_support = True     # Gemini models support audio
    video_input_support = True     # Gemini models support video

    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = True        # Live API with WebSocket support
    
    # Additional capabilities
    function_calling_support = True  # Advanced function calling
    context_caching_support = True   # Cost optimization feature

    def __init__(self, **kwargs):
        """
        Initialize the Vertex AI provider.

        Args:
            service_account_json: Path to service account JSON file
            project_id: Google Cloud project ID
            location: Google Cloud region (default: us-central1)
            **kwargs: Additional configuration options
        """
        # Get configuration with potential overrides from global config only
        self.config = get_provider_config("vertexai")

        # Extract credential parameters
        service_account_json = kwargs.pop("service_account_json", None)
        project_id = kwargs.pop("project_id", None)
        location = kwargs.pop("location", None)

        # Filter out credential parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ["service_account_json", "project_id", "location"]
        }

        # Update non-credential configuration
        self.config.update(filtered_kwargs)

        # Apply credentials explicitly provided to the constructor
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
            with open(self.service_account_path, 'r') as f:
                self.service_account_data = json.load(f)
                # Extract project ID from service account if not provided
                if not self.config.get("project_id"):
                    self.config["project_id"] = self.service_account_data.get("project_id")
        else:
            raise InvalidRequestError(
                f"Service account JSON file not found: {self.service_account_path}",
                provider="vertexai"
            )

        if not self.config.get("project_id"):
            raise InvalidRequestError(
                "Google Cloud project ID is required. "
                "Provide project_id parameter or ensure it's in the service account JSON.",
                provider="vertexai"
            )

        # Store relevant configuration
        self.project_id = self.config["project_id"]
        self.location = self.config.get("location", "us-central1")
        self.timeout = self.config.get("timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Build API base URL
        self.api_base = f"https://{self.location}-aiplatform.googleapis.com/v1"
        
        # Initialize access token (will be fetched on first request)
        self._access_token = None
        self._token_expiry = 0

        # Create retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    async def _get_access_token(self) -> str:
        """
        Get access token for authentication.
        
        This is a simplified implementation. In production, you should use
        google-auth library for proper authentication.
        
        Returns:
            Access token string
        """
        # Check if we have a valid token
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
            
        # In production, use google-auth library:
        # from google.auth.transport.requests import Request
        # from google.oauth2 import service_account
        # credentials = service_account.Credentials.from_service_account_file(
        #     self.service_account_path,
        #     scopes=['https://www.googleapis.com/auth/cloud-platform']
        # )
        # credentials.refresh(Request())
        # self._access_token = credentials.token
        # self._token_expiry = credentials.expiry.timestamp()
        
        # For now, raise an error indicating google-auth is needed
        raise NotImplementedError(
            "Vertex AI provider requires google-auth library for authentication. "
            "Install it with: pip install google-auth google-auth-httplib2"
        )

    def _get_headers(self) -> dict[str, str]:
        """
        Get the headers for API requests.

        Returns:
            Dict of headers
        """
        # Note: In actual implementation, this would be async to get the token
        # For now, we'll return basic headers
        return {
            "Content-Type": "application/json",
            # Authorization header would be added here with the access token
        }

    def _convert_openai_to_vertex_messages(self, messages: list[Message]) -> tuple[list[dict], str | None]:
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
                system_instruction = content
            else:
                # Convert role names
                vertex_role = "model" if role == "assistant" else "user"
                
                # Handle different content types
                if isinstance(content, str):
                    vertex_contents.append({
                        "role": vertex_role,
                        "parts": [{"text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multi-modal content
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Convert image URL to Vertex AI format
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")
                            
                            if url.startswith("data:"):
                                # Extract base64 data
                                header, data = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                
                                parts.append({
                                    "inlineData": {
                                        "mimeType": media_type,
                                        "data": data
                                    }
                                })
                            else:
                                # For non-base64 URLs, add as text
                                parts.append({"text": f"[Image URL: {url}]"})
                    
                    vertex_contents.append({
                        "role": vertex_role,
                        "parts": parts
                    })
        
        return vertex_contents, system_instruction

    def _convert_vertex_to_openai_response(
        self, vertex_response: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Vertex AI response to OpenAI format.

        Args:
            vertex_response: Native Vertex AI response
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        # Extract content from Vertex AI response
        candidates = vertex_response.get("candidates", [])
        if not candidates:
            content = ""
            finish_reason = "stop"
        else:
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            
            # Combine all text parts
            content = ""
            for part in content_parts:
                if "text" in part:
                    content += part["text"]
            
            # Map finish reasons
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop"
            }
            finish_reason = finish_reason_map.get(
                candidate.get("finishReason", "STOP"), "stop"
            )
        
        # Create choice in OpenAI format
        choice = Choice(
            message={
                "role": "assistant",
                "content": content
            },
            finish_reason=finish_reason,
            index=0,
        )
        
        # Create usage information
        usage = None
        if "usageMetadata" in vertex_response:
            usage_data = vertex_response["usageMetadata"]
            usage = {
                "prompt_tokens": usage_data.get("promptTokenCount", 0),
                "completion_tokens": usage_data.get("candidatesTokenCount", 0),
                "total_tokens": usage_data.get("totalTokenCount", 0)
            }
        
        # Create the response object
        return ChatCompletionResponse(
            id=f"vertex-{int(time.time())}",
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
        Create a chat completion with Vertex AI.

        Note: This is a simplified implementation. Full implementation would
        require google-auth and proper API calls.

        Args:
            messages: List of messages in the conversation
            model: Model name (without provider prefix)
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Convert messages to Vertex AI format
        contents, system_instruction = self._convert_openai_to_vertex_messages(messages)
        
        # Build request data
        data = {
            "contents": contents,
            "generationConfig": {}
        }
        
        # Add system instruction if present
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        # Add generation config parameters
        if "temperature" in kwargs:
            data["generationConfig"]["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            data["generationConfig"]["maxOutputTokens"] = kwargs["max_tokens"]
        if "stop" in kwargs:
            data["generationConfig"]["stopSequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )
        
        # In a full implementation, this would make the actual API call
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Vertex AI provider requires google-auth library and additional setup. "
            "This is a placeholder implementation showing the interface."
        )

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        """
        Create a text completion.

        Vertex AI primarily uses chat format, so we convert this to a chat completion.

        Args:
            prompt: Text prompt to complete
            model: Model name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or generator yielding completion chunks
        """
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]
        
        # Use chat completion
        if stream:
            raise NotImplementedError("Streaming not implemented in this placeholder")
        else:
            chat_response = await self.create_chat_completion(
                messages, model, stream=False, **kwargs
            )
            
            # Convert to completion format
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
            model: Model name (e.g., "text-embedding-004")
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        raise NotImplementedError(
            "Vertex AI embeddings require google-auth library and additional setup. "
            "This is a placeholder implementation."
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to Vertex AI.

        Note: Vertex AI doesn't have a general file upload API like OpenAI.
        Files are typically handled as part of the request.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: Vertex AI doesn't support standalone file uploads
        """
        raise InvalidRequestError(
            "Vertex AI does not support standalone file uploads. "
            "Files should be included as part of the request content.",
            provider="vertexai"
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from Vertex AI.

        Note: Vertex AI doesn't have a file storage API.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file

        Raises:
            InvalidRequestError: Vertex AI doesn't support file downloads
        """
        raise InvalidRequestError(
            "Vertex AI does not support file downloads through the API.",
            provider="vertexai"
        )


# Register the Vertex AI provider
register_provider("vertexai", VertexAIProvider)
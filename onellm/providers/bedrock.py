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
AWS Bedrock provider implementation for OneLLM.

This module implements the AWS Bedrock provider adapter, supporting foundation models from
multiple providers including Anthropic, Meta, Mistral, Amazon, AI21 Labs, Cohere, and
Stability AI through AWS's managed service. Bedrock provides enterprise features like
guardrails, batch processing, and cross-region inference.
"""

import base64
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
except ImportError:
    raise ImportError(
        "AWS SDK (boto3) is required for Bedrock provider. " "Install it with: pip install boto3"
    )

from ..config import get_provider_config
from ..errors import (
    APIError,
    AuthenticationError,
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
from .base import Provider, register_provider


class BedrockProvider(Provider):
    """AWS Bedrock provider implementation."""

    # Set capability flags
    json_mode_support = False  # Use tool calling instead

    # Multi-modal capabilities (model-dependent)
    vision_support = True  # Claude, Nova, and some Llama models
    audio_input_support = False  # Not currently supported
    video_input_support = False  # Not currently supported

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    # Model ID mappings from OneLLM format to Bedrock format
    MODEL_MAPPINGS = {
        # Anthropic Claude models
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-2.1": "anthropic.claude-v2:1",
        "claude-2": "anthropic.claude-v2",
        "claude-instant": "anthropic.claude-instant-v1",
        # Meta Llama models
        "llama3-2-90b": "meta.llama3-2-90b-instruct-v1:0",
        "llama3-2-11b": "meta.llama3-2-11b-instruct-v1:0",
        "llama3-2-3b": "meta.llama3-2-3b-instruct-v1:0",
        "llama3-2-1b": "meta.llama3-2-1b-instruct-v1:0",
        "llama3-1-405b": "meta.llama3-1-405b-instruct-v1:0",
        "llama3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama3-1-8b": "meta.llama3-1-8b-instruct-v1:0",
        "llama3-70b": "meta.llama3-70b-instruct-v1",
        "llama3-8b": "meta.llama3-8b-instruct-v1",
        # Amazon Nova models
        "nova-pro": "amazon.nova-pro-v1:0",
        "nova-lite": "amazon.nova-lite-v1:0",
        "nova-micro": "amazon.nova-micro-v1:0",
        # Amazon Titan models
        "titan-text-express": "amazon.titan-text-express-v1",
        "titan-text-lite": "amazon.titan-text-lite-v1",
        "titan-embed-text": "amazon.titan-embed-text-v1",
        "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
        # Mistral models
        "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
        "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
        # Cohere models
        "command-r": "cohere.command-r-v1:0",
        "command-r-plus": "cohere.command-r-plus-v1:0",
        "command": "cohere.command-text-v14",
        "command-light": "cohere.command-light-text-v14",
        "embed-english": "cohere.embed-english-v3",
        "embed-multilingual": "cohere.embed-multilingual-v3",
        # AI21 Labs models
        "jamba-1-5-large": "ai21.jamba-1-5-large-v1:0",
        "jamba-1-5-mini": "ai21.jamba-1-5-mini-v1:0",
        "jurassic-2-ultra": "ai21.j2-ultra-v1",
        "jurassic-2-mid": "ai21.j2-mid-v1",
    }

    def __init__(self, **kwargs):
        """
        Initialize the AWS Bedrock provider.

        Args:
            profile: AWS profile name (default: from bedrock.json or environment)
            region: AWS region (default: from bedrock.json or 'us-east-1')
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token for temporary credentials (optional)
            **kwargs: Additional configuration options
        """
        # Load bedrock.json configuration if it exists
        bedrock_config = {}
        bedrock_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "bedrock.json")
        if os.path.exists(bedrock_json_path):
            with open(bedrock_json_path, "r") as f:
                bedrock_config = json.load(f)

        # Get configuration with potential overrides
        self.config = get_provider_config("bedrock")

        # Extract AWS-specific parameters
        self.profile = kwargs.pop("profile", bedrock_config.get("profile"))
        self.region = kwargs.pop("region", bedrock_config.get("region", "us-east-1"))
        self.aws_access_key_id = kwargs.pop(
            "aws_access_key_id", bedrock_config.get("aws_access_key_id")
        )
        self.aws_secret_access_key = kwargs.pop(
            "aws_secret_access_key", bedrock_config.get("aws_secret_access_key")
        )
        self.aws_session_token = kwargs.pop(
            "aws_session_token", bedrock_config.get("aws_session_token")
        )

        # Update non-credential configuration
        self.config.update(kwargs)

        # Store relevant configuration as instance variables
        self.timeout = self.config.get("timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)

        # Initialize boto3 client
        try:
            self._init_boto3_client()
        except Exception as e:
            raise AuthenticationError(
                f"Failed to initialize AWS Bedrock client: {str(e)}. "
                "Ensure you have valid AWS credentials configured via environment variables, "
                "AWS profile, or IAM role.",
                provider="bedrock",
            )

    def _init_boto3_client(self):
        """Initialize the boto3 Bedrock runtime client."""
        # Prepare boto3 session arguments
        session_kwargs = {}
        if self.profile:
            session_kwargs["profile_name"] = self.profile

        # Create boto3 session
        session = boto3.Session(**session_kwargs)

        # Prepare client arguments
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": self.region,
        }

        # Add explicit credentials if provided
        if self.aws_access_key_id and self.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                client_kwargs["aws_session_token"] = self.aws_session_token

        # Create the Bedrock runtime client
        self.client = session.client(**client_kwargs)

        # Also create a Bedrock client for model listing (if needed)
        self.bedrock_client = session.client("bedrock", region_name=self.region)

    def _map_model_name(self, model: str) -> str:
        """
        Map OneLLM model name to Bedrock model ID.

        Args:
            model: Model name (e.g., "claude-3-opus")

        Returns:
            Bedrock model ID (e.g., "anthropic.claude-3-opus-20240229-v1:0")
        """
        # Check if it's already a full Bedrock model ID
        if "." in model and ":" in model:
            return model

        # Try to map from our predefined mappings
        if model in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model]

        # If not found, assume it's a direct model ID
        return model

    def _convert_openai_to_bedrock_messages(
        self, messages: List[Message], model_id: str
    ) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Convert OpenAI-style messages to Bedrock Converse API format.

        Args:
            messages: OpenAI-style messages
            model_id: Bedrock model ID

        Returns:
            Tuple of (bedrock_messages, system_messages)
        """
        bedrock_messages = []
        system_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages go in a separate parameter
                if isinstance(content, str):
                    system_messages.append({"text": content})
                elif isinstance(content, list):
                    # Extract text from complex content
                    for item in content:
                        if item.get("type") == "text":
                            system_messages.append({"text": item.get("text", "")})
                continue

            # Convert role
            bedrock_role = "assistant" if role == "assistant" else "user"

            # Convert content
            if isinstance(content, str):
                # Simple text content
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                # Complex content with images, text, etc.
                bedrock_content = []
                for item in content:
                    if item.get("type") == "text":
                        bedrock_content.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Convert OpenAI image_url format to Bedrock image format
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")

                        # Handle base64 encoded images
                        if url.startswith("data:"):
                            # Extract media type and base64 data
                            header, data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]

                            # Decode base64 to bytes
                            image_bytes = base64.b64decode(data)

                            bedrock_content.append(
                                {
                                    "image": {
                                        "format": media_type.split("/")[1],  # e.g., "jpeg", "png"
                                        "source": {"bytes": image_bytes},
                                    }
                                }
                            )
                        else:
                            # For non-base64 URLs, we'd need to fetch and convert
                            # For now, just add as text description
                            bedrock_content.append({"text": f"[Image URL: {url}]"})
            else:
                # Fallback to string representation
                bedrock_content = [{"text": str(content)}]

            bedrock_messages.append({"role": bedrock_role, "content": bedrock_content})

        return bedrock_messages, system_messages if system_messages else None

    def _convert_bedrock_to_openai_response(
        self, bedrock_response: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """
        Convert Bedrock Converse API response to OpenAI format.

        Args:
            bedrock_response: Native Bedrock response
            model: Model name

        Returns:
            OpenAI-compatible response
        """
        # Extract output message
        output = bedrock_response.get("output", {})
        message = output.get("message", {})

        # Extract content
        content_items = message.get("content", [])
        message_content = ""

        for item in content_items:
            if "text" in item:
                message_content += item["text"]
            elif "toolUse" in item:
                # Handle tool use responses
                tool_use = item["toolUse"]
                # For now, append as text (could be enhanced to support function calling)
                message_content += (
                    f"\n[Tool Use: {tool_use['name']} with input {json.dumps(tool_use['input'])}]"
                )

        # Create choice
        choice = Choice(
            message={"role": "assistant", "content": message_content},
            finish_reason=bedrock_response.get("stopReason", "stop"),
            index=0,
        )

        # Create usage information
        usage = None
        if "usage" in bedrock_response:
            bedrock_usage = bedrock_response["usage"]
            usage = {
                "prompt_tokens": bedrock_usage.get("inputTokens", 0),
                "completion_tokens": bedrock_usage.get("outputTokens", 0),
                "total_tokens": bedrock_usage.get("totalTokens", 0),
            }

        # Create response ID
        response_id = bedrock_response.get("$metadata", {}).get(
            "requestId", f"bedrock-{int(time.time())}"
        )

        return ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
        )

    def _handle_bedrock_error(self, error: Exception) -> None:
        """
        Handle Bedrock-specific errors and convert to OneLLM errors.

        Args:
            error: The exception from boto3

        Raises:
            MuxiLLMError: Appropriate error based on the Bedrock error
        """
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]
            status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)

            if error_code == "AccessDeniedException":
                if "not supported for inference in your account" in error_message:
                    raise PermissionError(
                        f"Model access denied: {error_message}. "
                        "Please request access to this model in the AWS Bedrock console.",
                        provider="bedrock",
                        status_code=403,
                    )
                else:
                    raise AuthenticationError(error_message, provider="bedrock", status_code=403)
            elif error_code == "ValidationException":
                raise InvalidRequestError(error_message, provider="bedrock", status_code=400)
            elif error_code == "ResourceNotFoundException":
                raise ResourceNotFoundError(error_message, provider="bedrock", status_code=404)
            elif error_code == "ThrottlingException":
                raise RateLimitError(error_message, provider="bedrock", status_code=429)
            elif error_code == "ServiceQuotaExceededException":
                raise RateLimitError(
                    f"Service quota exceeded: {error_message}", provider="bedrock", status_code=429
                )
            elif error_code == "ModelTimeoutException":
                raise TimeoutError(error_message, provider="bedrock", status_code=504)
            elif error_code == "InternalServerException":
                raise ServiceUnavailableError(error_message, provider="bedrock", status_code=500)
            else:
                raise APIError(
                    f"AWS Bedrock error: {error_message} (code: {error_code})",
                    provider="bedrock",
                    status_code=status_code,
                )
        elif isinstance(error, BotoCoreError):
            # Handle general boto3 errors
            raise APIError(f"AWS SDK error: {str(error)}", provider="bedrock")
        else:
            # Re-raise unknown errors
            raise error

    async def create_chat_completion(
        self, messages: List[Message], model: str, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion with AWS Bedrock using the Converse API.

        Args:
            messages: List of messages in the conversation
            model: Model name (without provider prefix)
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects
        """
        # Map model name to Bedrock model ID
        model_id = self._map_model_name(model)

        # Convert messages to Bedrock format
        bedrock_messages, system_messages = self._convert_openai_to_bedrock_messages(
            messages, model_id
        )

        # Set up inference configuration
        inference_config = {
            "maxTokens": kwargs.get("max_tokens", 1000),
        }

        # Add optional parameters
        if "temperature" in kwargs:
            inference_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs:
            stop_sequences = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )
            inference_config["stopSequences"] = stop_sequences

        # Prepare the request
        request_params = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config,
        }

        # Add system messages if present
        if system_messages:
            request_params["system"] = system_messages

        # Add tool configuration if provided
        if "tools" in kwargs:
            request_params["toolConfig"] = {"tools": kwargs["tools"]}

        # Add guardrail configuration if provided
        if "guardrail_config" in kwargs:
            request_params["guardrailConfig"] = kwargs["guardrail_config"]

        try:
            if stream:
                # Handle streaming response
                async def chunk_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
                    """Generator function to process streaming chunks"""
                    # Use converse_stream for streaming
                    response = self.client.converse_stream(**request_params)

                    # Process the event stream
                    for event in response.get("stream", []):
                        if "contentBlockStart" in event:
                            # Beginning of a content block
                            continue
                        elif "contentBlockDelta" in event:
                            # Content delta with text
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                # Create a ChoiceDelta object
                                choice_delta = ChoiceDelta(
                                    content=delta["text"],
                                    role=None,
                                    function_call=None,
                                    tool_calls=None,
                                    finish_reason=None,
                                )
                                # Create a StreamingChoice object
                                choice = StreamingChoice(
                                    delta=choice_delta,
                                    finish_reason=None,
                                    index=0,
                                )

                                # Create the chunk response object
                                chunk_resp = ChatCompletionChunk(
                                    id=f"bedrock-stream-{int(time.time())}",
                                    object="chat.completion.chunk",
                                    created=int(time.time()),
                                    model=model,
                                    choices=[choice],
                                    system_fingerprint=None,
                                )
                                yield chunk_resp
                        elif "contentBlockStop" in event:
                            # End of a content block
                            continue
                        elif "messageStop" in event:
                            # End of the message
                            stop_reason = event["messageStop"].get("stopReason", "stop")

                            # Send final chunk with finish_reason
                            choice_delta = ChoiceDelta(
                                content=None,
                                role=None,
                                function_call=None,
                                tool_calls=None,
                                finish_reason=stop_reason,
                            )
                            choice = StreamingChoice(
                                delta=choice_delta,
                                finish_reason=stop_reason,
                                index=0,
                            )

                            chunk_resp = ChatCompletionChunk(
                                id=f"bedrock-stream-{int(time.time())}",
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[choice],
                                system_fingerprint=None,
                            )
                            yield chunk_resp
                        elif "metadata" in event:
                            # Metadata about usage, etc.
                            continue

                return chunk_generator()
            else:
                # Handle non-streaming response
                response = self.client.converse(**request_params)

                # Convert Bedrock response to OpenAI format
                return self._convert_bedrock_to_openai_response(response, model)

        except Exception as e:
            self._handle_bedrock_error(e)

    async def create_completion(
        self, prompt: str, model: str, stream: bool = False, **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Note: Bedrock doesn't have a direct completion endpoint, so we convert
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
        self, input: Union[str, List[str]], model: str, **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the provided input using Amazon Titan Embeddings models.

        Args:
            input: Text or list of texts to embed
            model: Model name
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        # Map model name to Bedrock model ID
        model_id = self._map_model_name(model)

        # Ensure input is a list
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input

        # Check if this is a Titan embedding model
        if "titan-embed" not in model_id and "embed" not in model_id.lower():
            raise InvalidRequestError(
                f"Model '{model_id}' does not support embeddings. "
                "Use Amazon Titan Embeddings models (e.g., 'titan-embed-text-v1') or "
                "Cohere embedding models (e.g., 'embed-english').",
                provider="bedrock",
            )

        embeddings = []
        total_tokens = 0

        try:
            for i, text in enumerate(texts):
                # Prepare request based on model type
                if "titan" in model_id:
                    # Amazon Titan format
                    request_body = {
                        "inputText": text,
                        **{k: v for k, v in kwargs.items() if k in ["dimensions"]},
                    }
                elif "cohere" in model_id:
                    # Cohere format
                    request_body = {
                        "texts": [text],
                        "input_type": kwargs.get("input_type", "search_document"),
                        **{k: v for k, v in kwargs.items() if k in ["truncate"]},
                    }
                else:
                    # Generic format
                    request_body = {"inputText": text}

                # Make the request
                response = self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )

                # Parse response
                response_body = json.loads(response["body"].read())

                # Extract embedding based on model type
                if "titan" in model_id:
                    embedding = response_body.get("embedding", [])
                    token_count = response_body.get("inputTextTokenCount", 0)
                elif "cohere" in model_id:
                    embedding = response_body.get("embeddings", [[]])[0]
                    token_count = len(text.split())  # Approximate
                else:
                    embedding = response_body.get("embedding", [])
                    token_count = len(text.split())  # Approximate

                embeddings.append(EmbeddingData(index=i, embedding=embedding, object="embedding"))
                total_tokens += token_count

        except Exception as e:
            self._handle_bedrock_error(e)

        # Create usage info
        usage = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=model,
            usage=usage,
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file to AWS Bedrock.

        Note: Bedrock doesn't have a direct file upload API. Files are typically
        included directly in requests as base64-encoded data.

        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            **kwargs: Additional parameters

        Returns:
            FileObject representing the uploaded file

        Raises:
            InvalidRequestError: Bedrock doesn't support file uploads
        """
        raise InvalidRequestError(
            "AWS Bedrock does not support file uploads. "
            "Include images directly in messages as base64-encoded data.",
            provider="bedrock",
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file from AWS Bedrock.

        Note: Bedrock doesn't have a file storage system.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters

        Returns:
            Bytes content of the file

        Raises:
            InvalidRequestError: Bedrock doesn't support file downloads
        """
        raise InvalidRequestError(
            "AWS Bedrock does not support file downloads. " "Files are not stored in Bedrock.",
            provider="bedrock",
        )


# Register the Bedrock provider
register_provider("bedrock", BedrockProvider)

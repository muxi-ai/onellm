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
Response and request model definitions for OneLLM.

This module contains the data models used for responses and requests
across different API endpoints and providers.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from .types import Message, UsageInfo
from .utils.text_cleaner import clean_unicode_artifacts


def _clean_message_content(
    content: Union[str, List[Dict[str, Any]], None]
) -> Union[str, List[Dict[str, Any]], None]:
    """
    Clean Unicode artifacts from message content.

    Handles both simple string content and multi-modal content lists.

    Args:
        content: The message content to clean

    Returns:
        Cleaned content with Unicode artifacts removed
    """
    if content is None:
        return None

    if isinstance(content, str):
        return clean_unicode_artifacts(content)

    if isinstance(content, list):
        # Handle multi-modal content - clean text in ContentItem objects
        cleaned_content = []
        for item in content:
            if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                # Clean the text field in ContentItem
                cleaned_item = item.copy()
                cleaned_item['text'] = clean_unicode_artifacts(item['text'])
                cleaned_content.append(cleaned_item)
            else:
                cleaned_content.append(item)
        return cleaned_content

    return content


@dataclass
class ChoiceDelta:
    """
    Represents a chunk of a streaming response.

    This class is used to model incremental updates in streaming responses,
    containing partial content or other response elements.

    Attributes:
        content: The text content of the delta
        role: The role associated with this delta (e.g., 'user', 'assistant')
        function_call: Details of a function call if present
        tool_calls: List of tool calls if present
        finish_reason: Reason why the response finished (e.g., 'stop', 'length')
    """

    content: Optional[str] = None
    role: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None

    def __post_init__(self):
        """Clean Unicode artifacts from content after initialization."""
        if self.content is not None:
            self.content = clean_unicode_artifacts(self.content)


@dataclass
class Choice:
    """
    Represents a single completion choice in a response.

    In many LLM APIs, multiple alternative completions can be generated
    for a single request. This class represents one such completion.

    Attributes:
        message: The message content and metadata
        finish_reason: Reason why the response finished (e.g., 'stop', 'length')
        index: Position of this choice in the list of choices
    """

    message: Message
    finish_reason: Optional[str] = None
    index: int = 0

    def __init__(
        self,
        message: Optional[Message] = None,
        finish_reason: Optional[str] = None,
        index: int = 0,
        **kwargs
    ):
        """
        Initialize a Choice object.

        Args:
            message: The message content and metadata
            finish_reason: Reason why the response finished
            index: Position of this choice in the list of choices
            **kwargs: Additional keyword arguments for future compatibility
        """
        self.message = message or {}  # Default to empty dict if None
        self.finish_reason = finish_reason
        self.index = index

        # Clean Unicode artifacts from message content
        if self.message and 'content' in self.message:
            self.message['content'] = _clean_message_content(self.message['content'])


@dataclass
class StreamingChoice:
    """
    Represents a single streaming choice in a response.

    Similar to Choice, but specifically for streaming responses where
    content is delivered incrementally.

    Attributes:
        delta: The incremental update in this chunk
        finish_reason: Reason why the response finished (e.g., 'stop', 'length')
        index: Position of this choice in the list of choices
    """

    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    index: int = 0

    def __init__(
        self,
        delta: Optional[ChoiceDelta] = None,
        finish_reason: Optional[str] = None,
        index: int = 0,
        **kwargs
    ):
        """
        Initialize a StreamingChoice object.

        Args:
            delta: The incremental update in this chunk
            finish_reason: Reason why the response finished
            index: Position of this choice in the list of choices
            **kwargs: Additional keyword arguments for future compatibility
        """
        self.delta = delta or ChoiceDelta()  # Default to empty ChoiceDelta if None
        self.finish_reason = finish_reason
        self.index = index


@dataclass
class ChatCompletionResponse:
    """
    Response from a chat completion request.

    This class models the complete response from a chat completion API call,
    containing metadata about the request and the generated completions.

    Attributes:
        id: Unique identifier for this completion
        object: Type of object (typically 'chat.completion')
        created: Unix timestamp of when the completion was created
        model: The model used for completion
        choices: List of completion choices
        usage: Token usage information
        system_fingerprint: System identifier for the model version
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

    def __init__(
        self,
        id: str,
        object: str,
        created: int,
        model: str,
        choices: List[Choice],
        usage: Optional[UsageInfo] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a ChatCompletionResponse object.

        Args:
            id: Unique identifier for this completion
            object: Type of object (typically 'chat.completion')
            created: Unix timestamp of when the completion was created
            model: The model used for completion
            choices: List of completion choices
            usage: Token usage information
            system_fingerprint: System identifier for the model version
            **kwargs: Additional keyword arguments for future compatibility
        """
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = system_fingerprint


@dataclass
class ChatCompletionChunk:
    """
    Chunk of a streaming chat completion response.

    This class represents a single chunk in a streaming response,
    containing partial completion data.

    Attributes:
        id: Unique identifier for this completion
        object: Type of object (typically 'chat.completion.chunk')
        created: Unix timestamp of when the chunk was created
        model: The model used for completion
        choices: List of streaming choices in this chunk
        system_fingerprint: System identifier for the model version
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamingChoice]
    system_fingerprint: Optional[str] = None

    def __init__(
        self,
        id: str,
        object: str,
        created: int,
        model: str,
        choices: List[StreamingChoice],
        system_fingerprint: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a ChatCompletionChunk object.

        Args:
            id: Unique identifier for this completion
            object: Type of object (typically 'chat.completion.chunk')
            created: Unix timestamp of when the chunk was created
            model: The model used for completion
            choices: List of streaming choices in this chunk
            system_fingerprint: System identifier for the model version
            **kwargs: Additional keyword arguments for future compatibility
        """
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.system_fingerprint = system_fingerprint


@dataclass
class CompletionChoice:
    """
    Represents a single text completion choice in a response.

    This class is used for traditional text completion (non-chat) responses.

    Attributes:
        text: The generated text content
        index: Position of this choice in the list of choices
        logprobs: Log probabilities for token predictions if requested
        finish_reason: Reason why the response finished (e.g., 'stop', 'length')
    """

    text: str
    index: int = 0
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    def __post_init__(self):
        """Clean Unicode artifacts from text after initialization."""
        if self.text:
            self.text = clean_unicode_artifacts(self.text)


@dataclass
class CompletionResponse:
    """
    Response from a text completion request.

    This class models the complete response from a text completion API call,
    containing metadata about the request and the generated completions.

    Attributes:
        id: Unique identifier for this completion
        object: Type of object (typically 'text_completion')
        created: Unix timestamp of when the completion was created
        model: The model used for completion
        choices: List of completion choices
        usage: Token usage information
        system_fingerprint: System identifier for the model version
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


@dataclass
class EmbeddingData:
    """
    Represents a single embedding in a response.

    Embeddings are vector representations of text that capture semantic meaning.

    Attributes:
        embedding: Vector of floating point numbers representing the embedding
        index: Position of this embedding in the list of embeddings
        object: Type of object (typically 'embedding')
    """

    embedding: List[float]
    index: int = 0
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    """
    Response from an embedding request.

    This class models the complete response from an embedding API call,
    containing the generated embeddings and metadata.

    Attributes:
        object: Type of object (typically 'list')
        data: List of embedding data objects
        model: The model used to generate embeddings
        usage: Token usage information
    """

    object: str
    data: List[EmbeddingData]
    model: str
    usage: Optional[UsageInfo] = None


@dataclass
class FileObject:
    """
    Represents a file stored with the provider.

    This class models metadata about files that have been uploaded to the
    provider's storage system.

    Attributes:
        id: Unique identifier for the file
        object: Type of object (typically 'file')
        bytes: Size of the file in bytes
        created_at: Unix timestamp of when the file was created
        filename: Name of the file
        purpose: Purpose of the file (e.g., 'fine-tune', 'assistants')
        status: Current status of the file (e.g., 'processed')
        status_details: Additional details about the file status
    """

    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[str] = None

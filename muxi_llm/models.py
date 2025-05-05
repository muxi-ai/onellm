#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
Response and request model definitions for muxi-llm.

This module contains the data models used for responses and requests
across different API endpoints and providers.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .types import Message, UsageInfo


@dataclass
class ChoiceDelta:
    """Represents a chunk of a streaming response."""
    content: Optional[str] = None
    role: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None


@dataclass
class Choice:
    """Represents a single completion choice in a response."""
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
        self.message = message or {}
        self.finish_reason = finish_reason
        self.index = index


@dataclass
class StreamingChoice:
    """Represents a single streaming choice in a response."""
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
        self.delta = delta or ChoiceDelta()
        self.finish_reason = finish_reason
        self.index = index


@dataclass
class ChatCompletionResponse:
    """Response from a chat completion request."""
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
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = system_fingerprint


@dataclass
class ChatCompletionChunk:
    """Chunk of a streaming chat completion response."""
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
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.system_fingerprint = system_fingerprint


@dataclass
class CompletionChoice:
    """Represents a single text completion choice in a response."""
    text: str
    index: int = 0
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class CompletionResponse:
    """Response from a text completion request."""
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


@dataclass
class EmbeddingData:
    """Represents a single embedding in a response."""
    embedding: List[float]
    index: int = 0
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Optional[UsageInfo] = None


@dataclass
class FileObject:
    """Represents a file stored with the provider."""
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[str] = None

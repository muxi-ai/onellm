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
Common type definitions for muxi-llm.

This module contains standard type definitions used throughout the library.
"""

from typing import Any, Dict, List, Literal, Optional, Union, TypedDict, IO


# Role types for chat messages
Role = Literal["system", "user", "assistant", "tool", "function"]

# Content types for multi-modal messages
ContentType = Literal["text", "image", "image_url"]

# Image URL detail types
ImageUrlDetail = Literal["auto", "low", "high"]

# Provider types
Provider = Literal["openai", "anthropic", "azure", "ollama", "together", "groq"]

# Audio format types
AudioFormat = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

# Audio response format types
AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]

# Text-to-speech voice types
SpeechVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Text-to-speech output format types
SpeechFormat = Literal["mp3", "opus", "aac", "flac"]

# Image size types
ImageSize = Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]

# Image quality types (DALL-E 3)
ImageQuality = Literal["standard", "hd"]

# Image style types (DALL-E 3)
ImageStyle = Literal["natural", "vivid"]

# Image response format types
ImageResponseFormat = Literal["url", "b64_json"]


class ImageUrl(TypedDict, total=False):
    """Image URL details for vision models."""
    url: str
    detail: Optional[ImageUrlDetail]


class ContentItem(TypedDict, total=False):
    """A single content item in a chat message."""
    type: ContentType
    text: Optional[str]
    image_url: Optional[ImageUrl]


class Message(TypedDict, total=False):
    """A chat message."""
    role: Role
    content: Union[str, List[ContentItem]]
    name: Optional[str]
    function_call: Optional[Dict[str, Any]]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


class UsageInfo(TypedDict, total=False):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelParams(TypedDict, total=False):
    """Parameters for model configuration."""
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    stop: Optional[Union[str, List[str]]]
    logit_bias: Optional[Dict[str, float]]
    user: Optional[str]


class ResponseFormat(TypedDict, total=False):
    """Format specification for the response."""
    type: Literal["text", "json_object"]


class AudioTranscriptionParams(TypedDict, total=False):
    """Parameters for audio transcription requests."""
    file: Union[str, bytes, IO[bytes]]
    model: str
    language: Optional[str]
    prompt: Optional[str]
    response_format: Optional[AudioResponseFormat]
    temperature: Optional[float]


class AudioTranslationParams(TypedDict, total=False):
    """Parameters for audio translation requests."""
    file: Union[str, bytes, IO[bytes]]
    model: str
    prompt: Optional[str]
    response_format: Optional[AudioResponseFormat]
    temperature: Optional[float]


class TranscriptionResult(TypedDict, total=False):
    """Result from audio transcription or translation."""
    text: str
    task: Optional[str]
    language: Optional[str]
    duration: Optional[float]
    segments: Optional[List[Dict[str, Any]]]
    words: Optional[List[Dict[str, Any]]]


class SpeechParams(TypedDict, total=False):
    """Parameters for text-to-speech requests."""
    input: str
    model: str
    voice: SpeechVoice
    response_format: Optional[SpeechFormat]
    speed: Optional[float]


class ImageGenerationParams(TypedDict, total=False):
    """Parameters for image generation requests."""
    prompt: str
    model: str
    n: Optional[int]
    size: Optional[ImageSize]
    quality: Optional[ImageQuality]
    style: Optional[ImageStyle]
    response_format: Optional[ImageResponseFormat]
    user: Optional[str]


class ImageData(TypedDict, total=False):
    """Generated image data."""
    url: Optional[str]
    b64_json: Optional[str]
    revised_prompt: Optional[str]
    filepath: Optional[str]  # Added locally when saving images


class ImageGenerationResult(TypedDict, total=False):
    """Result from image generation."""
    created: int
    data: List[ImageData]


# Export everything for convenience
__all__ = [
    "Role",
    "ContentType",
    "Provider",
    "ImageUrlDetail",
    "ImageUrl",
    "ContentItem",
    "Message",
    "UsageInfo",
    "ModelParams",
    "ResponseFormat",
    "AudioFormat",
    "AudioResponseFormat",
    "AudioTranscriptionParams",
    "AudioTranslationParams",
    "TranscriptionResult",
    "SpeechVoice",
    "SpeechFormat",
    "SpeechParams",
    "ImageSize",
    "ImageQuality",
    "ImageStyle",
    "ImageResponseFormat",
    "ImageGenerationParams",
    "ImageData",
    "ImageGenerationResult",
]

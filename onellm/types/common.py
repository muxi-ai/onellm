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
Common type definitions for OneLLM.

This module contains standard type definitions used throughout the library.
These types provide a consistent interface for working with various LLM providers
while maintaining compatibility with the OpenAI API format.
"""

from typing import Any, Dict, List, Literal, Optional, Union, TypedDict, IO


# Role types for chat messages
Role = Literal["system", "user", "assistant", "tool", "function"]
# Defines the possible roles in a conversation with an LLM

# Content types for multi-modal messages
ContentType = Literal["text", "image", "image_url"]
# Specifies the types of content that can be included in messages

# Image URL detail types
ImageUrlDetail = Literal["auto", "low", "high"]
# Controls the level of detail when processing images via URL

# Provider types
Provider = Literal["openai", "anthropic", "azure", "ollama", "together", "groq"]
# Supported LLM providers in the library

# Audio format types
AudioFormat = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
# Supported audio input formats for transcription/translation

# Audio response format types
AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]
# Output formats for audio transcription and translation

# Text-to-speech voice types
SpeechVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
# Available voice options for text-to-speech generation

# Text-to-speech output format types
SpeechFormat = Literal["mp3", "opus", "aac", "flac"]
# Output formats for generated speech audio

# Image size types
ImageSize = Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
# Available dimensions for generated images

# Image quality types (DALL-E 3)
ImageQuality = Literal["standard", "hd"]
# Quality settings for image generation

# Image style types (DALL-E 3)
ImageStyle = Literal["natural", "vivid"]
# Style options for generated images

# Image response format types
ImageResponseFormat = Literal["url", "b64_json"]
# How image generation results are returned


class ImageUrl(TypedDict, total=False):
    """
    Image URL details for vision models.

    Attributes:
        url: The URL of the image to process
        detail: Optional level of detail for image processing
    """

    url: str
    detail: Optional[ImageUrlDetail]


class ContentItem(TypedDict, total=False):
    """
    A single content item in a chat message.

    Used for multi-modal conversations that may include text and images.

    Attributes:
        type: The type of content (text, image, etc.)
        text: Optional text content
        image_url: Optional image URL and detail level
    """

    type: ContentType
    text: Optional[str]
    image_url: Optional[ImageUrl]


class Message(TypedDict, total=False):
    """
    A chat message.

    Represents a single message in a conversation with an LLM.

    Attributes:
        role: The role of the message sender
        content: The message content (text or multi-modal)
        name: Optional name identifier for the message sender
        function_call: Optional function call information
        tool_call_id: Optional ID for tool calls
        tool_calls: Optional list of tool calls
    """

    role: Role
    content: Union[str, List[ContentItem]]
    name: Optional[str]
    function_call: Optional[Dict[str, Any]]
    tool_call_id: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


class UsageInfo(TypedDict, total=False):
    """
    Token usage information.

    Tracks token consumption for billing and monitoring.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used in the request
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelParams(TypedDict, total=False):
    """
    Parameters for model configuration.

    Controls various aspects of model behavior during inference.

    Attributes:
        temperature: Controls randomness (higher = more random)
        top_p: Controls diversity via nucleus sampling
        n: Number of completions to generate
        stream: Whether to stream the response
        max_tokens: Maximum tokens to generate
        presence_penalty: Penalizes new tokens based on presence in text
        frequency_penalty: Penalizes new tokens based on frequency in text
        stop: Sequences where the model should stop generating
        logit_bias: Modifies likelihood of specific tokens
        user: User identifier for the request
    """

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
    """
    Format specification for the response.

    Controls how the model formats its output.

    Attributes:
        type: The format type (text or JSON)
    """

    type: Literal["text", "json_object"]


class AudioTranscriptionParams(TypedDict, total=False):
    """
    Parameters for audio transcription requests.

    Used when converting speech to text.

    Attributes:
        file: The audio file to transcribe
        model: The model to use for transcription
        language: Optional language of the audio
        prompt: Optional text to guide the transcription
        response_format: Optional format for the response
        temperature: Optional temperature setting
    """

    file: Union[str, bytes, IO[bytes]]
    model: str
    language: Optional[str]
    prompt: Optional[str]
    response_format: Optional[AudioResponseFormat]
    temperature: Optional[float]


class AudioTranslationParams(TypedDict, total=False):
    """
    Parameters for audio translation requests.

    Used when translating speech from one language to English.

    Attributes:
        file: The audio file to translate
        model: The model to use for translation
        prompt: Optional text to guide the translation
        response_format: Optional format for the response
        temperature: Optional temperature setting
    """

    file: Union[str, bytes, IO[bytes]]
    model: str
    prompt: Optional[str]
    response_format: Optional[AudioResponseFormat]
    temperature: Optional[float]


class TranscriptionResult(TypedDict, total=False):
    """
    Result from audio transcription or translation.

    Contains the transcribed or translated text and metadata.

    Attributes:
        text: The transcribed or translated text
        task: Optional description of the task performed
        language: Optional detected language
        duration: Optional duration of the audio in seconds
        segments: Optional list of audio segments with timestamps
        words: Optional list of individual words with timestamps
    """

    text: str
    task: Optional[str]
    language: Optional[str]
    duration: Optional[float]
    segments: Optional[List[Dict[str, Any]]]
    words: Optional[List[Dict[str, Any]]]


class SpeechParams(TypedDict, total=False):
    """
    Parameters for text-to-speech requests.

    Used when generating spoken audio from text.

    Attributes:
        input: The text to convert to speech
        model: The model to use for speech generation
        voice: The voice to use for the audio
        response_format: Optional audio format for the response
        speed: Optional speech speed multiplier
    """

    input: str
    model: str
    voice: SpeechVoice
    response_format: Optional[SpeechFormat]
    speed: Optional[float]


class ImageGenerationParams(TypedDict, total=False):
    """
    Parameters for image generation requests.

    Used when creating images from text prompts.

    Attributes:
        prompt: The text prompt to generate images from
        model: The model to use for image generation
        n: Optional number of images to generate
        size: Optional dimensions of the generated images
        quality: Optional quality level for the images
        style: Optional style for the generated images
        response_format: Optional format for the response
        user: Optional user identifier
    """

    prompt: str
    model: str
    n: Optional[int]
    size: Optional[ImageSize]
    quality: Optional[ImageQuality]
    style: Optional[ImageStyle]
    response_format: Optional[ImageResponseFormat]
    user: Optional[str]


class ImageData(TypedDict, total=False):
    """
    Generated image data.

    Contains the output from image generation.

    Attributes:
        url: Optional URL to the generated image
        b64_json: Optional base64-encoded image data
        revised_prompt: Optional revised prompt used for generation
        filepath: Optional local path where the image was saved
    """

    url: Optional[str]
    b64_json: Optional[str]
    revised_prompt: Optional[str]
    filepath: Optional[str]  # Added locally when saving images


class ImageGenerationResult(TypedDict, total=False):
    """
    Result from image generation.

    Contains metadata and the generated images.

    Attributes:
        created: Unix timestamp of when the images were created
        data: List of generated image data
    """

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

"""
Common type definitions for muxi-llm.

This module contains standard type definitions used throughout the library.
"""

from typing import Any, Dict, List, Literal, Optional, Union, TypedDict


# Role types for chat messages
Role = Literal["system", "user", "assistant", "tool", "function"]

# Content types for multi-modal messages
ContentType = Literal["text", "image", "image_url"]

# Image URL detail types
ImageUrlDetail = Literal["auto", "low", "high"]

# Provider types
Provider = Literal["openai", "anthropic", "azure", "ollama", "together", "groq"]


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
]

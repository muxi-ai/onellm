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
Token counting utilities for various models.

This module provides utilities for counting tokens in text
for different language models. It supports both tiktoken-based
counting for OpenAI models and fallback approximations for other models.
"""

import re
from typing import Dict, List, Optional, Union

# Note: tiktoken is an optional dependency
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# Regex pattern for tokenizing text when tiktoken is not available
# This is a simple approximation and not accurate for all languages/models
SIMPLE_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


# Map of OpenAI model names to tiktoken encodings
OPENAI_MODEL_ENCODINGS = {
    # GPT-4 models
    "gpt-4": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-2024-04-09": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-vision-preview": "cl100k_base",
    # GPT-3.5 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    # Base GPT-3 models
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # Embedding models
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}


def get_encoder(model: str) -> Optional["tiktoken.Encoding"]:
    """
    Get the tiktoken encoder for the specified model.

    This function attempts to find the appropriate tiktoken encoder for a given model.
    It first checks if the model has a known encoding in our mapping, then tries to get
    the encoder directly from tiktoken, and finally falls back to cl100k_base as a default.

    Args:
        model: Name of the model to get the encoder for

    Returns:
        Encoding object if available, None otherwise if tiktoken is not installed
        or if no appropriate encoder can be found
    """
    # Return None immediately if tiktoken is not available
    if not TIKTOKEN_AVAILABLE:
        return None

    # Extract base model name if using provider prefix (e.g., "openai/gpt-4")
    if "/" in model:
        _, model = model.split("/", 1)

    try:
        # First check if we have a predefined encoding for this model
        if model in OPENAI_MODEL_ENCODINGS:
            encoding_name = OPENAI_MODEL_ENCODINGS[model]
            return tiktoken.get_encoding(encoding_name)

        # If not in our mapping, try to get encoder directly from tiktoken
        return tiktoken.encoding_for_model(model)
    except (KeyError, ImportError, ValueError):
        # If both methods fail, fall back to cl100k_base which works for most newer models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # If all attempts fail, return None
            return None


def num_tokens_from_string(text: str, model: Optional[str] = None) -> int:
    """
    Count the number of tokens in a string.

    This function counts tokens in a text string using either tiktoken (if available
    and a model is specified) or a simple regex-based approximation as fallback.

    Args:
        text: Text to count tokens for
        model: Optional model name to use for counting with tiktoken

    Returns:
        Number of tokens in the text
    """
    # Handle empty text case
    if not text:
        return 0

    # Try to use tiktoken for accurate counting if available and model is specified
    if TIKTOKEN_AVAILABLE and model:
        encoder = get_encoder(model)
        if encoder:
            return len(encoder.encode(text))

    # Fallback to simple approximation using regex
    # This splits text into words and punctuation as a rough token estimate
    return len(SIMPLE_TOKEN_PATTERN.findall(text))


def num_tokens_from_messages(
    messages: List[Dict[str, Union[str, List]]], model: Optional[str] = None
) -> int:
    """
    Count the number of tokens in a list of chat messages.

    This function handles token counting for chat message formats used by LLMs.
    It accounts for message formatting overhead and handles nested content structures
    like those used in multi-modal conversations.

    Args:
        messages: List of chat messages in the format expected by LLM APIs
        model: Optional model name to use for counting

    Returns:
        Total number of tokens in the messages, including formatting overhead
    """
    # Handle empty messages case
    if not messages:
        return 0

    # Extract base model name if using provider prefix
    if model and "/" in model:
        _, model = model.split("/", 1)

    # Special handling for OpenAI chat models with tiktoken
    # This follows OpenAI's specific token counting methodology for chat formats
    if TIKTOKEN_AVAILABLE and model and model.startswith(("gpt-3.5", "gpt-4")):
        encoder = get_encoder(model)
        if encoder:
            # Constants based on OpenAI's token counting methodology
            tokens_per_message = (
                3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = 1  # If there's a name, the role is omitted

            token_count = 0
            for message in messages:
                # Add base tokens for each message
                token_count += tokens_per_message

                # Process each field in the message
                for key, value in message.items():
                    if isinstance(value, str):
                        # For string values, encode and count directly
                        token_count += len(encoder.encode(value))
                    elif isinstance(value, list):
                        # Handle content list (for multi-modal inputs)
                        for item in value:
                            # Check if this is a content item with text
                            is_dict = isinstance(item, dict)
                            has_text = is_dict and "text" in item
                            is_text_str = has_text and isinstance(item["text"], str)

                            # Only count tokens for text content
                            if is_dict and has_text and is_text_str:
                                token_count += len(encoder.encode(item["text"]))

                    # Add extra token for name field if present
                    if key == "name":
                        token_count += tokens_per_name

            # Add 3 tokens for assistant message formatting (final reply format)
            token_count += 3
            return token_count

    # Fallback to simple approximation for non-OpenAI models or when tiktoken is unavailable
    token_count = 0
    for message in messages:
        # Count tokens in each field that's a string
        for value in message.values():
            if isinstance(value, str):
                # Count tokens in string values
                token_count += num_tokens_from_string(value, model)
            elif isinstance(value, list):
                # Handle content list (for multi-modal inputs)
                for item in value:
                    # Check if this is a content item with text
                    is_dict = isinstance(item, dict)
                    has_text = is_dict and "text" in item
                    is_text_str = has_text and isinstance(item["text"], str)

                    # Only count tokens for text content
                    if is_dict and has_text and is_text_str:
                        token_count += num_tokens_from_string(item["text"], model)

    # Add approximation of formatting overhead (4 tokens per message)
    # This is a rough estimate of the tokens used for message formatting
    token_count += len(messages) * 4
    return token_count

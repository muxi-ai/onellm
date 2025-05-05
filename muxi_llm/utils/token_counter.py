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

    Args:
        model: Name of the model

    Returns:
        Encoding object if available, None otherwise
    """
    if not TIKTOKEN_AVAILABLE:
        return None

    # Extract base model name if using provider prefix
    if "/" in model:
        _, model = model.split("/", 1)

    try:
        # Check for exact match
        if model in OPENAI_MODEL_ENCODINGS:
            encoding_name = OPENAI_MODEL_ENCODINGS[model]
            return tiktoken.get_encoding(encoding_name)

        # Try to get by model name directly
        return tiktoken.encoding_for_model(model)
    except (KeyError, ImportError, ValueError):
        # Fallback to cl100k_base for most newer models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def num_tokens_from_string(text: str, model: Optional[str] = None) -> int:
    """
    Count the number of tokens in a string.

    Args:
        text: Text to count tokens for
        model: Optional model name to use for counting

    Returns:
        Number of tokens
    """
    if not text:
        return 0

    # Try to use tiktoken if available
    if TIKTOKEN_AVAILABLE and model:
        encoder = get_encoder(model)
        if encoder:
            return len(encoder.encode(text))

    # Fallback to simple approximation
    return len(SIMPLE_TOKEN_PATTERN.findall(text))


def num_tokens_from_messages(
    messages: List[Dict[str, Union[str, List]]], model: Optional[str] = None
) -> int:
    """
    Count the number of tokens in a list of messages.

    Args:
        messages: List of chat messages
        model: Optional model name to use for counting

    Returns:
        Number of tokens
    """
    if not messages:
        return 0

    # Extract base model name if using provider prefix
    if model and "/" in model:
        _, model = model.split("/", 1)

    # Special handling for OpenAI chat models with tiktoken
    if TIKTOKEN_AVAILABLE and model and model.startswith(("gpt-3.5", "gpt-4")):
        encoder = get_encoder(model)
        if encoder:
            # Format based on OpenAI's token counting methodology
            tokens_per_message = (
                3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = 1  # If there's a name, the role is omitted

            token_count = 0
            for message in messages:
                token_count += tokens_per_message
                for key, value in message.items():
                    if isinstance(value, str):
                        token_count += len(encoder.encode(value))
                    elif isinstance(value, list):
                        # Handle content list (for multi-modal inputs)
                        for item in value:
                            is_dict = isinstance(item, dict)
                            has_text = is_dict and "text" in item
                            is_text_str = has_text and isinstance(item["text"], str)

                            if is_dict and has_text and is_text_str:
                                token_count += len(encoder.encode(item["text"]))

                    if key == "name":
                        token_count += tokens_per_name

            # Add 3 tokens for assistant message formatting
            token_count += 3
            return token_count

    # Fallback to simple approximation
    token_count = 0
    for message in messages:
        # Count tokens in each field that's a string
        for value in message.values():
            if isinstance(value, str):
                token_count += num_tokens_from_string(value, model)
            elif isinstance(value, list):
                # Handle content list (for multi-modal inputs)
                for item in value:
                    is_dict = isinstance(item, dict)
                    has_text = is_dict and "text" in item
                    is_text_str = has_text and isinstance(item["text"], str)

                    if is_dict and has_text and is_text_str:
                        token_count += num_tokens_from_string(item["text"], model)

    # Approximation of formatting overhead
    token_count += len(messages) * 4
    return token_count

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
Type definitions for OneLLM.

This module provides type definitions and data structures used throughout the OneLLM
package. These types ensure consistent interfaces when working with different LLM
providers and handling various content formats.

The types defined here include:
- Role: Enumeration of possible message roles (system, user, assistant)
- ContentType: Enumeration of content types (text, image, etc.)
- Provider: Enumeration of supported LLM providers
- ContentItem: Structure for multi-modal content items
- Message: Structure for messages in a conversation
- UsageInfo: Structure for token usage statistics
- ModelParams: Configuration parameters for model requests
- ResponseFormat: Structure for specifying response format preferences
"""

# Import all type definitions from the common module
from .common import (
    Role,                # Defines possible message roles (system, user, assistant)
    ContentType,         # Defines content types (text, image, audio, etc.)
    Provider,            # Enumerates supported LLM providers
    ContentItem,         # Structure for multi-modal content items
    Message,             # Structure for messages in a conversation
    UsageInfo,           # Structure for token usage statistics
    ModelParams,         # Configuration parameters for model requests
    ResponseFormat,      # Structure for specifying response format preferences
)

# Define the public API for this module
__all__ = [
    "Role",
    "ContentType",
    "Provider",
    "ContentItem",
    "Message",
    "UsageInfo",
    "ModelParams",
    "ResponseFormat",
]

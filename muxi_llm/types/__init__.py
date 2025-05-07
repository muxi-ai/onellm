#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi-llm
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
Type definitions for muxi-llm.

This module provides type definitions and data structures used throughout the muxi-llm
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

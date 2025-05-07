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
Utility functions and classes for muxi-llm.

This module provides various utility functions and classes used throughout the muxi-llm
package to handle common tasks such as:
- Asynchronous retry mechanisms for API calls
- Streaming response handling
- Error management for streaming operations

These utilities help ensure robust API interactions and proper handling of
streaming responses when working with different LLM providers.
"""

# Import retry-related utilities for handling transient API failures
from .retry import retry_async, RetryConfig

# Import streaming utilities for handling real-time response processing
from .streaming import stream_generator, StreamingError

# Define the public API for this module
__all__ = [
    "retry_async",       # Async function decorator that implements retry logic
    "RetryConfig",       # Configuration class for customizing retry behavior
    "stream_generator",  # Generator function for processing streaming responses
    "StreamingError",    # Exception class for streaming-related errors
]

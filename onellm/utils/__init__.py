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
Utility functions and classes for OneLLM.

This module provides various utility functions and classes used throughout the OneLLM
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

# Import text cleaning utilities for processing AI model responses
from .text_cleaner import clean_unicode_artifacts

# Define the public API for this module
__all__ = [
    "retry_async",            # Async function decorator that implements retry logic
    "RetryConfig",            # Configuration class for customizing retry behavior
    "stream_generator",       # Generator function for processing streaming responses
    "StreamingError",         # Exception class for streaming-related errors
    "clean_unicode_artifacts",  # Function for cleaning Unicode artifacts from text
]

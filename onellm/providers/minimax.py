#!/usr/bin/env python3
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
MiniMax provider implementation for OneLLM.

MiniMax provides AI models through an Anthropic-compatible API. They offer the MiniMax-M2 
model series with advanced reasoning capabilities, tool use, and interleaved thinking.

Supported models:
- MiniMax-M2: Agentic capabilities with advanced reasoning
- MiniMax-M2-Stable: High concurrency and commercial use

For more information, see: https://platform.minimax.io/docs/api-reference/text-anthropic-api
"""

from .base import register_provider
from .anthropic_compatible import AnthropicCompatibleProvider


class MinimaxProvider(AnthropicCompatibleProvider):
    """MiniMax provider implementation using Anthropic-compatible API."""

    # Provider configuration
    provider_name = "minimax"
    default_api_base = "https://api.minimax.io/anthropic"

    # Set capability flags (inherited from Anthropic but documented here for clarity)
    json_mode_support = False  # Anthropic doesn't have explicit JSON mode

    # Multi-modal capabilities
    vision_support = False  # MiniMax M2 doesn't support vision through Anthropic API yet
    audio_input_support = False  # No audio support
    video_input_support = False  # No video support

    # Streaming capabilities
    streaming_support = True  # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False  # No realtime API

    # Additional capabilities
    thinking_support = True  # Supports interleaved thinking
    tool_calling_support = True  # Supports function/tool calling


# Register the MiniMax provider
register_provider("minimax", MinimaxProvider)

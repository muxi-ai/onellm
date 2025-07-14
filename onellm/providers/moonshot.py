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
Moonshot provider implementation for OneLLM.

Moonshot AI is a Chinese AI company providing the Kimi large language model series
with OpenAI-compatible APIs, known for their long-context capabilities and strong
performance on coding and reasoning tasks.
"""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot provider implementation."""

    # Provider configuration
    provider_name = "moonshot"
    default_api_base = "https://api.moonshot.ai/v1"

    # Set capability flags
    json_mode_support = True

    # Multi-modal capabilities
    vision_support = True          # Kimi-VL supports vision
    audio_input_support = True     # Kimi-Audio supports audio
    video_input_support = False    # No video support currently

    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming

    # Realtime capabilities
    realtime_support = False       # No realtime API

    # Additional capabilities
    function_calling_support = True  # Supports function calling


# Register the Moonshot provider
register_provider("moonshot", MoonshotProvider)

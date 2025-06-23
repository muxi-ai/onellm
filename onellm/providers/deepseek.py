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
DeepSeek provider implementation for OneLLM.

DeepSeek is a Chinese AI company providing competitive language models with
OpenAI-compatible APIs, known for their cost-effective pricing and strong
performance on coding tasks.
"""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation."""
    
    # Provider configuration
    provider_name = "deepseek"
    default_api_base = "https://api.deepseek.com/v1"
    
    # Set capability flags
    json_mode_support = True
    
    # Multi-modal capabilities
    vision_support = False         # No vision support currently
    audio_input_support = False    # No audio support
    video_input_support = False    # No video support
    
    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming
    
    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = True  # Supports function calling


# Register the DeepSeek provider
register_provider("deepseek", DeepSeekProvider)
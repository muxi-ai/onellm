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
OpenRouter provider implementation for OneLLM.

OpenRouter provides a unified interface to 100+ language models from various providers
including OpenAI, Anthropic, Google, Meta, and many others. It offers smart routing,
fallback options, and competitive pricing.
"""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider implementation."""
    
    # Provider configuration
    provider_name = "openrouter"
    default_api_base = "https://openrouter.ai/api/v1"
    requires_special_headers = True
    
    # Set capability flags (varies by model)
    json_mode_support = True
    
    # Multi-modal capabilities (model-dependent)
    vision_support = True          # Many models support vision
    audio_input_support = False    # No direct audio support
    video_input_support = False    # No video support
    
    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming
    
    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = True  # Many models support function calling
    
    def _get_special_headers(self) -> dict[str, str]:
        """
        Get OpenRouter-specific headers.
        
        OpenRouter recommends setting HTTP-Referer and X-Title headers
        for better analytics and debugging.
        
        Returns:
            Dict of special headers
        """
        return {
            "HTTP-Referer": "https://github.com/muxi-ai/onellm",
            "X-Title": "OneLLM"
        }


# Register the OpenRouter provider
register_provider("openrouter", OpenRouterProvider)
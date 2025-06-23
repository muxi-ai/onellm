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
Perplexity AI provider implementation for OneLLM.

Perplexity AI provides search-augmented language models that can access real-time
information from the internet and provide responses with citations.
"""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class PerplexityProvider(OpenAICompatibleProvider):
    """Perplexity AI provider implementation."""
    
    # Provider configuration
    provider_name = "perplexity"
    default_api_base = "https://api.perplexity.ai"
    
    # Set capability flags
    json_mode_support = False      # No explicit JSON mode
    
    # Multi-modal capabilities
    vision_support = False         # No vision support
    audio_input_support = False    # No audio support
    video_input_support = False    # No video support
    
    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming
    
    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = False  # No function calling
    
    # Perplexity-specific features
    search_augmented = True        # Models have internet access
    provides_citations = True      # Responses include citations


# Register the Perplexity provider
register_provider("perplexity", PerplexityProvider)
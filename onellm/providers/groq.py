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
Groq provider implementation for OneLLM.

Groq provides ultra-fast AI inference through their custom Language Processing Unit (LPU)
technology. They offer OpenAI-compatible APIs with specialized focus on speed and efficiency.
"""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Groq provider implementation."""
    
    # Provider configuration
    provider_name = "groq"
    default_api_base = "https://api.groq.com/openai/v1"
    
    # Set capability flags
    json_mode_support = True
    
    # Multi-modal capabilities
    vision_support = True          # Limited to specific models
    audio_input_support = False    # No direct audio input
    video_input_support = False    # No video support
    
    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming
    
    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = True  # Supports function calling
    
    def _process_messages_for_vision(self, messages: list, model: str) -> list:
        """
        Process messages for vision models.
        
        Groq has limited vision support only for specific models.
        """
        # Check if any message contains images
        has_images = any(
            isinstance(msg.get("content"), list) and 
            any(item.get("type") in ["image_url", "image"] for item in msg.get("content", []))
            for msg in messages
        )
        
        if has_images:
            # Only specific models support vision
            vision_models = {"llava-v1.5-7b-4096-preview"}
            if model not in vision_models:
                from ..errors import InvalidRequestError
                raise InvalidRequestError(
                    f"Model '{model}' does not support vision inputs. "
                    f"Use a vision-capable model like 'llava-v1.5-7b-4096-preview'.",
                    provider=self.provider_name
                )
        
        return messages


# Register the Groq provider
register_provider("groq", GroqProvider)
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
Google AI Studio (Gemini) provider implementation for OneLLM.

Google AI Studio provides access to Gemini models through an OpenAI-compatible API.
This is different from Vertex AI which is Google Cloud's enterprise offering.
"""

from typing import Any

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class GoogleProvider(OpenAICompatibleProvider):
    """Google AI Studio (Gemini) provider implementation."""
    
    # Provider configuration
    provider_name = "google"
    default_api_base = "https://generativelanguage.googleapis.com/v1beta"
    
    # Set capability flags
    json_mode_support = True
    
    # Multi-modal capabilities
    vision_support = True          # Gemini models support vision
    audio_input_support = True     # Gemini models support audio
    video_input_support = True     # Gemini models support video
    
    # Streaming capabilities
    streaming_support = True       # All models support streaming
    token_by_token_support = True  # Provides token-by-token streaming
    
    # Realtime capabilities
    realtime_support = False       # No realtime API
    
    # Additional capabilities
    function_calling_support = True  # Supports function calling
    
    # Google-specific features
    thinking_mode_support = True   # Gemini 2.0 Flash supports thinking mode
    
    async def _make_request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Any:
        """
        Override to handle Google's API key in URL pattern.
        
        Google AI Studio uses API key as a query parameter rather than in headers.
        """
        # Add API key to the path as a query parameter
        separator = "&" if "?" in path else "?"
        path = f"{path}{separator}key={self.api_key}"
        
        # Remove the Authorization header since Google uses URL param
        return await super()._make_request(
            method=method,
            path=path,
            data=data,
            stream=stream,
            timeout=timeout,
            files=files
        )
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get headers for API requests.
        
        Google AI Studio doesn't use Authorization header.
        """
        return {
            "Content-Type": "application/json"
        }


# Register the Google provider
register_provider("google", GoogleProvider)
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
Anyscale Endpoints provider implementation for OneLLM.

This module implements the Anyscale provider adapter, supporting open-source
models like Llama, Mistral, and CodeLlama through an OpenAI-compatible API.
"""

from ..config import get_provider_config
from .openai_compatible import OpenAICompatibleProvider
from .base import register_provider


class AnyscaleProvider(OpenAICompatibleProvider):
    """Anyscale Endpoints provider implementation.

    Anyscale provides scalable inference for open-source models with an
    OpenAI-compatible API. Built by the creators of Ray, it offers simple
    pricing and good performance for models like Llama and Mistral.
    """

    # Set capability flags
    json_mode_support = True  # Supports JSON mode with schema
    vision_support = False  # No image support
    audio_input_support = False  # No audio support
    video_input_support = False  # No video support
    streaming_support = True  # Supports streaming
    token_by_token_support = True  # SSE streaming
    realtime_support = False  # No realtime API
    function_calling_support = True  # Supports function calling (single calls only)

    def __init__(self, **kwargs):
        """
        Initialize the Anyscale provider.

        Args:
            api_key: Anyscale API key (starts with 'esecret_')
            **kwargs: Additional configuration options
        """
        # Get configuration
        config = get_provider_config("anyscale")

        # API key must start with 'esecret_'
        api_key = kwargs.pop("api_key", config.get("api_key"))

        # Ensure we have an API key
        if not api_key:
            import os

            api_key = os.environ.get("ANYSCALE_API_KEY")

        # Update configuration
        config["api_key"] = api_key
        config["api_base"] = config.get("api_base", "https://api.endpoints.anyscale.com/v1")
        config.update(kwargs)

        # Initialize parent with provider configuration
        super().__init__(provider_name="anyscale", config=config, **kwargs)

    def get_headers(self) -> dict:
        """
        Get headers for Anyscale API requests.

        Returns:
            Dictionary of headers including authorization
        """
        headers = super().get_headers()
        # Anyscale uses Bearer token authentication
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def map_model_name(self, model: str) -> str:
        """
        Map model names to Anyscale's naming convention.

        Anyscale uses full model names in the format:
        {organization}/{model-name}

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Mapped model name
        """
        # Remove provider prefix if present
        if "/" in model and model.split("/")[0] == "anyscale":
            model = model.split("/", 1)[1]

        # Common model aliases
        model_aliases = {
            # Llama 3 aliases
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            # Llama 2 aliases
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
            "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
            # Mistral aliases
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            # Code models
            "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
            "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
            "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
            "codellama-70b": "codellama/CodeLlama-70b-Instruct-hf",
            # Other models
            "gemma-7b": "google/gemma-7b-it",
        }

        # Return alias if found, otherwise return as-is
        return model_aliases.get(model, model)


# Register the provider
register_provider("anyscale", AnyscaleProvider)

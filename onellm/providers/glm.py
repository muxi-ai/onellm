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

"""Zhipu GLM provider implementation for OneLLM."""

from .base import register_provider
from .openai_compatible import OpenAICompatibleProvider


class GLMProvider(OpenAICompatibleProvider):
    """Zhipu GLM provider using the OpenAI-compatible API surface."""

    provider_name = "glm"
    default_api_base = "https://api.z.ai/api/paas/v4"

    # Capability flags (model dependent, defaults informed by provider docs)
    json_mode_support = True

    vision_support = True          # GLM models offer image understanding in newer releases
    audio_input_support = False    # No direct audio input in OpenAI-compatible API today
    video_input_support = False    # No video support exposed yet

    streaming_support = True       # Streaming supported for chat/completions endpoint
    token_by_token_support = True  # Uses standard SSE chunk streaming

    realtime_support = False       # No realtime WebSocket API

    function_calling_support = True  # GLM exposes tool/function calling via OpenAI schema


# Register the GLM provider
register_provider("glm", GLMProvider)

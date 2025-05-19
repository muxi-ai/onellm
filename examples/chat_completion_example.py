#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/llm
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
# ============================================================================ #
# MUXI-LLM EXAMPLE: Chat Completion
# ============================================================================ #
#
# This example demonstrates the core ChatCompletion functionality of muxi-llm,
# providing a unified interface for interacting with various LLM providers.
# Key features demonstrated:
#
# - Synchronous and asynchronous API usage
# - Streaming responses for incremental text generation
# - Provider-specific model usage with unified interface
# - Working with response objects and message formats
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages muxi-llm's support for:
# - ChatCompletion API (core functionality)
# - Multiple providers through a unified interface
# - Streaming and non-streaming response handling
# - Response object schema for structured data access
#
# RELATED EXAMPLES:
# ----------------
# - fallback_example.py: Using fallback models for reliability
# - parallel_operation_example.py: Parallel processing with multiple requests
# - json_mode_example.py: Structured JSON responses
# - vision_example.py: Handling multi-modal inputs with images
#
# REQUIREMENTS:
# ------------
# - muxi-llm
# - OpenAI API key
# - Anthropic API key (optional, for Claude examples)
#
# EXPECTED OUTPUT:
# ---------------
# Multiple chat completion responses demonstrating different patterns:
# 1. Synchronous request and response
# 2. Asynchronous request and response
# 3. Streaming tokens for real-time display
# 4. Different provider responses if Anthropic key is available
# ============================================================================ #
"""

import asyncio
import os
from typing import List, Dict, Any, Callable

from muxi_llm import ChatCompletion
from muxi_llm.models import ChatCompletionResponse, ChatCompletionChunk
from muxi_llm.config import set_api_key


def print_chat_response(response: ChatCompletionResponse) -> None:
    """
    Print the response from a chat completion.

    This function formats and displays the key information from a ChatCompletionResponse
    object, including the model used, the content of the response, and usage statistics
    if available.

    Args:
        response: The ChatCompletionResponse object to print
    """
    print("\n--- Chat Completion Response ---")
    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message['content']}")
    if response.usage:
        print(f"Usage: {response.usage}")


async def streaming_demo(
    model: str,
    messages: List[Dict[str, Any]],
    callback: Callable[[str], None] = print
) -> None:
    """
    Demonstrate streaming chat completions.

    This function shows how to use the streaming API to receive responses
    incrementally as they are generated, rather than waiting for the complete response.

    Args:
        model: The model identifier in format "provider/model"
        messages: List of message dictionaries to send to the model
        callback: Function to call with each chunk of text as it arrives (defaults to print)
    """
    print(f"\n--- Streaming Chat Completion ({model}) ---")
    print("Response:", end=" ", flush=True)

    # Call with streaming=True to get a generator
    stream = await ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True
    )

    # Process the streaming response chunks
    async for chunk in stream:
        if isinstance(chunk, ChatCompletionChunk):
            for choice in chunk.choices:
                if choice.delta.content:
                    callback(choice.delta.content)


def non_streaming_demo(model: str, messages: List[Dict[str, Any]]) -> None:
    """
    Demonstrate non-streaming chat completions using the synchronous API.

    This function shows how to use the synchronous API to get a complete
    response in a single call, blocking until the response is ready.

    Args:
        model: The model identifier in format "provider/model"
        messages: List of message dictionaries to send to the model
    """
    # Use the synchronous API
    response = ChatCompletion.create(
        model=model,
        messages=messages
    )
    print_chat_response(response)


async def async_non_streaming_demo(model: str, messages: List[Dict[str, Any]]) -> None:
    """
    Demonstrate asynchronous non-streaming chat completions.

    This function shows how to use the asynchronous API to get a complete
    response without blocking the event loop, allowing other tasks to run
    while waiting for the response.

    Args:
        model: The model identifier in format "provider/model"
        messages: List of message dictionaries to send to the model
    """
    # Use the asynchronous API
    response = await ChatCompletion.acreate(
        model=model,
        messages=messages
    )
    print_chat_response(response)


async def main() -> None:
    """
    Run the example demonstrations of ChatCompletion functionality.

    This function:
    1. Sets up API keys from environment variables
    2. Demonstrates different usage patterns (sync, async, streaming)
    3. Shows how to use different providers (OpenAI, Anthropic)

    The function will skip examples for providers whose API keys are not available.
    """
    # Set API keys from environment variables
    # You can also pass these directly to the API calls via api_key parameter
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Check if OpenAI API key is available and configure it
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment.")
    else:
        set_api_key(openai_api_key, "openai")

    # Check if Anthropic API key is available and configure it
    if not anthropic_api_key:
        print("Note: ANTHROPIC_API_KEY not found in environment. "
              "Anthropic examples will be skipped.")
    else:
        set_api_key(anthropic_api_key, "anthropic")

    # Example messages for a chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]

    # Demonstrate different usage patterns and providers
    if openai_api_key:
        # 1. Using provider prefix (required)
        non_streaming_demo("openai/gpt-3.5-turbo", messages)

        # 2. Asynchronous usage
        await async_non_streaming_demo("openai/gpt-3.5-turbo", messages)

        # 3. Streaming usage
        def print_streaming(text: str) -> None:
            """Helper function to print streaming text without line breaks."""
            print(text, end="", flush=True)

        await streaming_demo("openai/gpt-3.5-turbo", messages, print_streaming)
        print()  # New line after streaming response

    # Demonstrate Anthropic model (if API key is available)
    if anthropic_api_key:
        # Anthropic requires a different message format (no system message)
        anthropic_messages = [
            {"role": "user", "content": "What are the three laws of robotics?"}
        ]

        # 4. Using Anthropic model (Claude)
        print("\n--- Using Anthropic Model ---")
        await async_non_streaming_demo("anthropic/claude-3-opus", anthropic_messages)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

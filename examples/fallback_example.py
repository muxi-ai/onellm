#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
# ============================================================================ #
# MUXI-LLM EXAMPLE: Fallback Mechanisms for Reliability
# ============================================================================ #
#
# This example demonstrates how to use muxi-llm's fallback mechanisms to create
# robust applications that gracefully handle model failures.
# Key features demonstrated:
#
# - Configuring fallback models across different providers
# - Custom fallback configurations and policies
# - Fallback callbacks for monitoring and logging
# - Streaming responses with fallback support
# - Error handling with specific error types
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages muxi-llm's support for:
# - Fallback mechanism in ChatCompletion, Completion, and Embedding APIs
# - Error handling and retry logic
# - Custom callback integrations
# - Cross-provider compatibility
#
# RELATED EXAMPLES:
# ----------------
# - retry_example.py: Retrying with the same model before fallback
# - chat_completion_example.py: Basic chat completions without fallbacks
# - parallel_operation_example.py: Parallel processing with reliability
#
# REQUIREMENTS:
# ------------
# - muxi-llm
# - OpenAI API key
# - Anthropic API key (optional, for more fallback options)
# - Cohere API key (optional, for more fallback options)
#
# EXPECTED OUTPUT:
# ---------------
# Multiple demonstrations of fallback scenarios:
# 1. Successful fallback from non-existent to valid model
# 2. Limited fallback attempts based on configuration
# 3. Fallbacks with different API types (completion, embedding)
# 4. Custom callback execution during fallback events
# 5. Streaming response with fallback support
# ============================================================================ #
"""

import asyncio
import os
import logging

from muxi_llm import ChatCompletion, Completion, Embedding
from muxi_llm.errors import RateLimitError

# Set up logging to see fallback messages
logging.basicConfig(level=logging.INFO)


def set_api_keys_from_env():
    """
    Set API keys from environment variables.

    This function retrieves API keys from environment variables for different providers
    and configures them in the muxi_llm library. It supports OpenAI, Anthropic, and Cohere.
    """
    from muxi_llm import set_api_key

    # Set API keys for different providers
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        set_api_key(openai_key, "openai")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        set_api_key(anthropic_key, "anthropic")

    cohere_key = os.environ.get("COHERE_API_KEY")
    if cohere_key:
        set_api_key(cohere_key, "cohere")


async def demonstrate_chat_completion_fallback():
    """
    Demonstrate fallback for chat completions.

    This function shows two scenarios:
    1. Fallback from a non-existent model to a valid model
    2. Custom fallback configuration with limited fallback attempts

    It demonstrates how the fallback mechanism automatically switches to alternative
    models when the primary model fails.
    """
    print("\n=== Chat Completion Fallback Demo ===")

    messages = [{"role": "user", "content": "What are three interesting facts about the moon?"}]

    try:
        # Scenario 1: Use a non-existent model to force fallback
        print("\nScenario 1: Fallback from non-existent model to valid model")
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            fallback_models=[
                "openai/gpt-3.5-turbo",  # This should work
                "anthropic/claude-3-haiku"  # Backup if the first fallback also fails
            ],
            fallback_config={
                "log_fallbacks": True  # Enable logging of fallback events
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")

    try:
        # Scenario 2: Custom fallback configuration
        print("\nScenario 2: Custom fallback configuration with max_fallbacks=1")
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model-1",  # This will fail
            messages=messages,
            fallback_models=[
                "openai/non-existent-model-2",  # This will also fail
                # This would work but won't be tried due to max_fallbacks=1:
                "openai/gpt-3.5-turbo"
            ],
            fallback_config={
                "max_fallbacks": 1,  # Only try the first fallback
                "log_fallbacks": True  # Enable logging of fallback events
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ Expected failure due to max_fallbacks=1: {e}")


async def demonstrate_completion_fallback():
    """
    Demonstrate fallback for text completions.

    This function shows how to use fallbacks with the Completion API, including
    configuring specific error types (RateLimitError) that should trigger fallbacks.
    """
    print("\n=== Text Completion Fallback Demo ===")

    prompt = "Write a haiku about programming:"

    try:
        response = await Completion.acreate(
            model="openai/non-existent-model",  # This will fail
            prompt=prompt,
            fallback_models=["openai/gpt-3.5-turbo-instruct"],  # This should work
            fallback_config={
                "log_fallbacks": True,
                "retriable_errors": [RateLimitError]  # Only retry on rate limit errors
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Text: {response.choices[0].text}")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def demonstrate_embedding_fallback():
    """
    Demonstrate fallback for embeddings.

    This function shows how to use fallbacks with the Embedding API, which is useful
    for ensuring reliable vector embeddings even when the primary embedding model fails.
    """
    print("\n=== Embedding Fallback Demo ===")

    texts = ["The quick brown fox jumps over the lazy dog"]

    try:
        response = await Embedding.acreate(
            model="openai/non-existent-embedding-model",  # This will fail
            input=texts,
            fallback_models=["openai/text-embedding-ada-002"],  # This should work
            fallback_config={
                "log_fallbacks": True  # Enable logging of fallback events
            }
        )

        print("✅ Success! Got embeddings from fallback model:")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def custom_fallback_callback(primary_model: str, fallback_model: str, error: Exception):
    """
    Example callback function when fallbacks are used.

    This function is called whenever a fallback occurs, providing information about
    the primary model that failed, the fallback model being used, and the error that
    triggered the fallback.

    Args:
        primary_model: The original model that failed
        fallback_model: The fallback model being used instead
        error: The exception that caused the primary model to fail
    """
    print("\n🔄 Fallback callback triggered:")
    print(f"  - Primary model: {primary_model}")
    print(f"  - Fallback model used: {fallback_model}")
    print(f"  - Error from primary model: {type(error).__name__}: {str(error)}")

    # You could send metrics, log to a monitoring system, or take other actions here


async def demonstrate_fallback_callback():
    """
    Demonstrate using a callback when fallbacks occur.

    This function shows how to register a custom callback function that will be
    invoked whenever a fallback is triggered, allowing for custom logging, metrics,
    or other actions.
    """
    print("\n=== Fallback Callback Demo ===")

    messages = [{"role": "user", "content": "What's your favorite programming language?"}]

    try:
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            fallback_models=["openai/gpt-3.5-turbo"],  # This should work
            fallback_config={
                "log_fallbacks": True,
                "fallback_callback": custom_fallback_callback  # Register our custom callback
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def demonstrate_streaming_fallback():
    """
    Demonstrate fallback for streaming responses.

    This function shows how fallbacks work with streaming responses, which is important
    for applications that need to display results incrementally as they are generated.
    """
    print("\n=== Streaming Fallback Demo ===")

    messages = [{"role": "user", "content": "Count from 1 to 5 slowly."}]

    try:
        print("\nStarting streaming with fallback:")
        stream = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            stream=True,  # Important: Enable streaming
            fallback_models=["openai/gpt-3.5-turbo"],  # This should work
            fallback_config={
                "log_fallbacks": True  # Enable logging of fallback events
            }
        )

        # Process the streaming response chunks
        response_text = ""
        async for chunk in stream:
            # Extract content from each chunk if available
            if chunk.choices and chunk.choices[0].delta.get("content"):
                content = chunk.choices[0].delta["content"]
                response_text += content  # Accumulate the full response
                print(content, end="", flush=True)  # Print immediately without newlines

        print("\n✅ Streaming complete!")

    except Exception as e:
        print(f"\n❌ All fallbacks failed: {e}")


async def run_demos():
    """
    Run all the demonstration functions.

    This function serves as the main entry point for the example, setting up API keys
    and executing all the demonstration functions in sequence.
    """
    # Set API keys
    set_api_keys_from_env()

    # Run demos
    await demonstrate_chat_completion_fallback()
    await demonstrate_completion_fallback()
    await demonstrate_embedding_fallback()
    await demonstrate_fallback_callback()
    await demonstrate_streaming_fallback()


if __name__ == "__main__":
    asyncio.run(run_demos())

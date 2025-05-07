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
# MUXI-LLM EXAMPLE: Retry Mechanisms and Error Handling
# ============================================================================ #
#
# This example demonstrates how to configure muxi-llm to automatically retry requests
# with the same model before attempting fallbacks to different models.
# Key features demonstrated:
#
# - Configuring retry attempts for transient failures
# - Combining retries with fallback models
# - Using retries with different API types (ChatCompletion, Completion)
# - Asynchronous retry behavior
# - Error handling strategies
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages muxi-llm's support for:
# - Retry mechanisms in API calls
# - Fallback configurations
# - Both synchronous and asynchronous interfaces
# - Cross-provider compatibility
#
# RELATED EXAMPLES:
# ----------------
# - fallback_example.py: Using different models as fallbacks
# - chat_completion_example.py: Basic API usage without retries
# - parallel_operation_example.py: Parallel processing with reliability
#
# REQUIREMENTS:
# ------------
# - muxi-llm
# - OpenAI API key
# - Anthropic API key (optional, for some examples)
#
# EXPECTED OUTPUT:
# ---------------
# Multiple demonstrations of retry scenarios:
# 1. Retrying the same model multiple times without fallbacks
# 2. Retrying before falling back to alternative models
# 3. Retry behavior with the Completion API
# 4. Asynchronous retry and fallback patterns
# ============================================================================ #
"""

import os
import asyncio
from muxi_llm import ChatCompletion, Completion

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


def demonstrate_chat_completion_retries():
    """
    Demonstrate retries with ChatCompletion.

    This function shows how to use the retries parameter with ChatCompletion
    in two scenarios:
    1. Retrying the same model multiple times without fallbacks
    2. Retrying the same model before falling back to alternative models

    The examples use a simple question about the capital of France to demonstrate
    the retry functionality.
    """
    print("\n=== Using retries with ChatCompletion ===\n")

    # Define messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # Example 1: Using retries without fallbacks
    print("Example 1: Retrying same model 3 times before failing")
    try:
        # Make a request with retries but no fallbacks
        # If the first attempt fails, it will retry up to 3 more times with the same model
        response = ChatCompletion.create(
            model="openai/gpt-4",
            messages=messages,
            retries=3,  # Will try gpt-4 up to 3 additional times if the first attempt fails
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        # This will only execute if all retry attempts fail
        print(f"Error (after 3 retries): {str(e)}")

    # Example 2: Using retries with fallbacks
    print("\nExample 2: Retrying same model 2 times before falling back to alternative models")
    try:
        # Make a request with both retries and fallbacks
        # First tries gpt-4 up to 3 times (initial + 2 retries)
        # Then falls back to gpt-3.5-turbo, then claude-3-haiku if needed
        response = ChatCompletion.create(
            model="openai/gpt-4",
            messages=messages,
            retries=2,  # Will try gpt-4 up to 2 additional times before trying fallbacks
            fallback_models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        # This will only execute if all retries and all fallbacks fail
        print(f"Error (after all attempts): {str(e)}")


def demonstrate_completion_retries():
    """
    Demonstrate retries with Completion.

    This function shows how to use the retries parameter with the Completion API
    in two scenarios:
    1. Retrying the same model multiple times without fallbacks
    2. Retrying the same model before falling back to alternative models

    The examples use a simple story prompt to demonstrate the retry functionality.
    """
    print("\n=== Using retries with Completion ===\n")

    # Define a prompt
    prompt = "Once upon a time in a land far, far away"

    # Example 3: Using retries without fallbacks
    print("Example 3: Retrying same model 3 times before failing")
    try:
        # Make a completion request with retries but no fallbacks
        # If the first attempt fails, it will retry up to 3 more times with the same model
        response = Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=prompt,
            retries=3,  # Will try the model up to 3 additional times if the first attempt fails
            max_tokens=50,
        )
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        # This will only execute if all retry attempts fail
        print(f"Error (after 3 retries): {str(e)}")

    # Example 4: Using retries with fallbacks
    print("\nExample 4: Retrying same model 2 times before falling back to alternative models")
    try:
        # Make a completion request with both retries and fallbacks
        # First tries gpt-3.5-turbo-instruct up to 3 times (initial + 2 retries)
        # Then falls back to text-davinci-003, then claude-instant-1.2 if needed
        response = Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=prompt,
            retries=2,  # Will try the model up to 2 additional times before trying fallbacks
            fallback_models=["openai/text-davinci-003", "anthropic/claude-instant-1.2"],
            max_tokens=50,
        )
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        # This will only execute if all retries and all fallbacks fail
        print(f"Error (after all attempts): {str(e)}")


async def demonstrate_async_retries():
    """
    Demonstrate retries with async API.

    This function shows how to use the retries parameter with the asynchronous
    ChatCompletion API. It demonstrates using both retries and fallbacks in an
    async context.

    The example uses a simple question about the capital of Italy to demonstrate
    the async retry functionality.
    """
    print("\n=== Using retries with async API ===\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Italy?"}
    ]

    # Example 5: Using retries with async API
    print("Example 5: Using retries with async API")
    try:
        # Make an async request with both retries and fallbacks
        # First tries gpt-4 up to 3 times (initial + 2 retries)
        # Then falls back to gpt-3.5-turbo if needed
        response = await ChatCompletion.acreate(
            model="openai/gpt-4",
            messages=messages,
            retries=2,  # Will try gpt-4 up to 2 additional times if the first attempt fails
            fallback_models=["openai/gpt-3.5-turbo"],
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        # This will only execute if all retries and all fallbacks fail
        print(f"Error (after all attempts): {str(e)}")


if __name__ == "__main__":
    # Run the synchronous examples
    demonstrate_chat_completion_retries()
    demonstrate_completion_retries()

    # Run the async example
    asyncio.run(demonstrate_async_retries())

    print("\nAll examples completed!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the use of retries parameter with muxi-llm.

This example shows how to configure automatic retries with the same model
before falling back to alternative models.
"""

import os
import asyncio
from muxi_llm import ChatCompletion, Completion

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


def demonstrate_chat_completion_retries():
    """Demonstrate retries with ChatCompletion."""
    print("\n=== Using retries with ChatCompletion ===\n")

    # Define messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # Example 1: Using retries without fallbacks
    print("Example 1: Retrying same model 3 times before failing")
    try:
        response = ChatCompletion.create(
            model="openai/gpt-4",
            messages=messages,
            retries=3,  # Will try gpt-4 up to 3 additional times if the first attempt fails
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Error (after 3 retries): {str(e)}")

    # Example 2: Using retries with fallbacks
    print("\nExample 2: Retrying same model 2 times before falling back to alternative models")
    try:
        response = ChatCompletion.create(
            model="openai/gpt-4",
            messages=messages,
            retries=2,  # Will try gpt-4 up to 2 additional times before trying fallbacks
            fallback_models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Error (after all attempts): {str(e)}")


def demonstrate_completion_retries():
    """Demonstrate retries with Completion."""
    print("\n=== Using retries with Completion ===\n")

    # Define a prompt
    prompt = "Once upon a time in a land far, far away"

    # Example 3: Using retries without fallbacks
    print("Example 3: Retrying same model 3 times before failing")
    try:
        response = Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=prompt,
            retries=3,  # Will try the model up to 3 additional times if the first attempt fails
            max_tokens=50,
        )
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        print(f"Error (after 3 retries): {str(e)}")

    # Example 4: Using retries with fallbacks
    print("\nExample 4: Retrying same model 2 times before falling back to alternative models")
    try:
        response = Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt=prompt,
            retries=2,  # Will try the model up to 2 additional times before trying fallbacks
            fallback_models=["openai/text-davinci-003", "anthropic/claude-instant-1.2"],
            max_tokens=50,
        )
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        print(f"Error (after all attempts): {str(e)}")


async def demonstrate_async_retries():
    """Demonstrate retries with async API."""
    print("\n=== Using retries with async API ===\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Italy?"}
    ]

    # Example 5: Using retries with async API
    print("Example 5: Using retries with async API")
    try:
        response = await ChatCompletion.acreate(
            model="openai/gpt-4",
            messages=messages,
            retries=2,  # Will try gpt-4 up to 2 additional times if the first attempt fails
            fallback_models=["openai/gpt-3.5-turbo"],
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Error (after all attempts): {str(e)}")


if __name__ == "__main__":
    # Run the synchronous examples
    demonstrate_chat_completion_retries()
    demonstrate_completion_retries()

    # Run the async example
    asyncio.run(demonstrate_async_retries())

    print("\nAll examples completed!")

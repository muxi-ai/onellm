#!/usr/bin/env python3
"""
Anthropic Provider Example

This example demonstrates how to use the Anthropic provider with OneLLM,
including chat completions, streaming, and Claude's unique features.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Anthropic from the model name)
# Uses ANTHROPIC_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Claude."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "What makes Claude unique compared to other AI assistants?",
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Claude."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Write a haiku about artificial intelligence."}],
        stream=True,
        max_tokens=100,
    )

    print("Streaming haiku: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def multi_turn_conversation():
    """Multi-turn conversation example."""
    print("\n=== Multi-turn Conversation ===")

    messages = [
        {"role": "user", "content": "Hi Claude, I'm learning Python."},
        {
            "role": "assistant",
            "content": (
                "Hello! That's great to hear. Python is an excellent language to learn. "
                "What aspects of Python are you currently focusing on?"
            ),
        },
        {
            "role": "user",
            "content": (
                "I'm trying to understand decorators. Can you explain them simply?"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku-20240307",  # Using Haiku for faster responses
        messages=messages,
        max_tokens=300,
    )

    print(f"Claude's explanation: {response.choices[0].message['content']}")


def json_output_example():
    """JSON structured output example."""
    print("\n=== JSON Output Example ===")

    response = client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
            {
                "role": "user",
                "content": (
                    "Create a JSON object with information about 3 planets in our solar system."
                ),
            },
        ],
        max_tokens=300,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation example."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a Python function that calculates the Fibonacci sequence up to n terms."
                ),
            }
        ],
        max_tokens=400,
        temperature=0.2,  # Lower temperature for more consistent code
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def different_claude_models():
    """Example showing different Claude models."""
    print("\n=== Different Claude Models ===")

    models = [
        ("anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (most capable)"),
        ("anthropic/claude-3-haiku-20240307", "Claude 3 Haiku (fastest)"),
        ("anthropic/claude-3-opus-20240229", "Claude 3 Opus (highly capable)"),
    ]

    prompt = "In one sentence, what is machine learning?"

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=100
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {model_id}: {e}")


def main():
    """Run all examples."""
    print("Anthropic (Claude) Provider Examples\n")

    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    try:
        basic_chat_completion()
        streaming_example()
        multi_turn_conversation()
        json_output_example()
        code_generation_example()
        different_claude_models()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

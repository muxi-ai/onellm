#!/usr/bin/env python3
"""
MiniMax Provider Example

This example demonstrates how to use the MiniMax provider with OneLLM,
including chat completions, streaming, and MiniMax's unique features like
interleaved thinking and advanced reasoning capabilities.

MiniMax provides an Anthropic-compatible API for their M2 model series.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects MiniMax from the model name)
# Uses MINMAX_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with MiniMax-M2."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "What are the key features of the MiniMax-M2 model?",
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with MiniMax-M2."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[{"role": "user", "content": "Write a haiku about artificial intelligence."}],
        stream=True,
        max_tokens=100,
    )

    print("Streaming haiku: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def interleaved_thinking_example():
    """Example demonstrating MiniMax's interleaved thinking capability."""
    print("\n=== Interleaved Thinking Example ===")

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[
            {
                "role": "user",
                "content": (
                    "Solve this problem step by step: "
                    "A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. "
                    "What is the total distance traveled?"
                ),
            }
        ],
        max_tokens=500,
        # Enable interleaved thinking for complex reasoning tasks
        thinking={"enabled": True, "budget_tokens": 20000},
    )

    print(f"Response with reasoning: {response.choices[0].message['content']}")


def multi_turn_conversation():
    """Multi-turn conversation example."""
    print("\n=== Multi-turn Conversation ===")

    messages = [
        {"role": "user", "content": "Hi, I'm learning about machine learning."},
        {
            "role": "assistant",
            "content": (
                "Hello! That's great to hear. Machine learning is a fascinating field. "
                "What specific area are you interested in exploring?"
            ),
        },
        {
            "role": "user",
            "content": (
                "I want to understand neural networks. Can you explain them simply?"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2", messages=messages, max_tokens=300
    )

    print(f"MiniMax's explanation: {response.choices[0].message['content']}")


def reasoning_task_example():
    """Example showing MiniMax's advanced reasoning capabilities."""
    print("\n=== Advanced Reasoning Example ===")

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[
            {
                "role": "user",
                "content": (
                    "Consider a scenario where you have 3 boxes: one contains only apples, "
                    "one contains only oranges, and one contains both. All boxes are labeled "
                    "incorrectly. You can pick one fruit from one box. How do you determine "
                    "the contents of all boxes?"
                ),
            }
        ],
        max_tokens=400,
        temperature=0.2,  # Lower temperature for more logical reasoning
    )

    print("Solution:")
    print(response.choices[0].message["content"])


def code_generation_example():
    """Code generation example with MiniMax-M2."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a Python function that implements binary search on a sorted list."
                ),
            }
        ],
        max_tokens=400,
        temperature=0.2,
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def model_variants_example():
    """Example showing different MiniMax model variants."""
    print("\n=== Different MiniMax Models ===")

    models = [
        ("minimax/MiniMax-M2", "MiniMax-M2 (Agentic capabilities, advanced reasoning)"),
        ("minimax/MiniMax-M2-Stable", "MiniMax-M2-Stable (High concurrency, commercial use)"),
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


def china_endpoint_example():
    """Example showing how to use the China endpoint."""
    print("\n=== China Endpoint Configuration ===")

    # For users in China, you can configure the API base
    # This can be done via environment variable or configuration
    print(
        "To use the China endpoint, set:\n"
        "export ONELLM_PROVIDERS__MINIMAX__API_BASE=https://api.minimaxi.com/anthropic"
    )


def json_output_example():
    """JSON structured output example."""
    print("\n=== JSON Output Example ===")

    response = client.chat.completions.create(
        model="minimax/MiniMax-M2",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs valid JSON.",
            },
            {
                "role": "user",
                "content": (
                    "Create a JSON object with information about 3 programming languages: "
                    "Python, JavaScript, and Rust. Include name, year created, and primary use case."
                ),
            },
        ],
        max_tokens=300,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def main():
    """Run all examples."""
    print("MiniMax Provider Examples\n")

    # Check if API key is set
    if not os.environ.get("MINMAX_API_KEY"):
        print("Please set MINMAX_API_KEY environment variable")
        print("Get your API key from: https://platform.minimax.io/")
        return

    try:
        basic_chat_completion()
        streaming_example()
        interleaved_thinking_example()
        multi_turn_conversation()
        reasoning_task_example()
        code_generation_example()
        json_output_example()
        model_variants_example()
        china_endpoint_example()

        print("\n=== Additional Information ===")
        print("MiniMax Documentation: https://platform.minimax.io/docs/")
        print("Supported Models: MiniMax-M2, MiniMax-M2-Stable")
        print("API Format: Anthropic-compatible")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

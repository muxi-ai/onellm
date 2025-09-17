#!/usr/bin/env python3
"""
Groq Provider Example

This example demonstrates how to use the Groq provider with OneLLM.
Groq offers ultra-fast inference using their LPU (Language Processing Unit).
"""

import os
import time
from onellm import OpenAI

# Initialize the client (it automatically detects Groq from the model name)
# Uses GROQ_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Groq's ultra-fast inference."""
    print("=== Basic Chat Completion ===")

    start_time = time.time()

    response = client.chat.completions.create(
        model="groq/llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What are the benefits of using Groq's LPU for AI inference?",
            },
        ],
        temperature=0.7,
        max_tokens=200,
    )

    end_time = time.time()

    print(f"Response: {response.choices[0].message['content']}")
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    print(f"Tokens/second: {response.usage['completion_tokens'] / (end_time - start_time):.1f}")


def streaming_example():
    """Streaming response showcasing Groq's speed."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="groq/llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 10 with a brief description of each number.",
            }
        ],
        stream=True,
        max_tokens=300,
    )

    print("Streaming response: ")
    start_time = time.time()
    tokens = 0

    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
            tokens += 1

    end_time = time.time()
    print(f"\n\nStreaming speed: ~{tokens / (end_time - start_time):.1f} tokens/second")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")

    response = client.chat.completions.create(
        model="groq/mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {
                "role": "user",
                "content": "Create a JSON object with 3 fast food restaurants and their popular items.",  # noqa: E501
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def different_models_comparison():
    """Compare different models available on Groq."""
    print("\n=== Model Comparison ===")

    models = [
        ("groq/llama3-8b-8192", "Llama 3 8B (fastest)"),
        ("groq/llama3-70b-8192", "Llama 3 70B (more capable)"),
        ("groq/mixtral-8x7b-32768", "Mixtral 8x7B (32K context)"),
        ("groq/gemma-7b-it", "Gemma 7B"),
    ]

    prompt = "Explain in one sentence what makes Groq's LPU technology special."

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            start_time = time.time()

            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5,
            )

            end_time = time.time()
            print(f"Response: {response.choices[0].message['content']}")
            print(f"Time: {end_time - start_time:.3f}s")

        except Exception as e:
            print(f"Error with {model_id}: {e}")


def tool_use_example():
    """Function calling example with Groq."""
    print("\n=== Tool Use Example ===")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_speed",
                "description": "Calculate speed given distance and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance": {"type": "number", "description": "Distance in kilometers"},
                        "time": {"type": "number", "description": "Time in hours"},
                    },
                    "required": ["distance", "time"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="groq/llama3-70b-8192",
        messages=[
            {"role": "user", "content": "If I travel 150 kilometers in 2 hours, what's my speed?"}
        ],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    if message.get("tool_calls"):
        print(f"Function to call: {message['tool_calls'][0]['function']['name']}")
        print(f"Arguments: {message['tool_calls'][0]['function']['arguments']}")
    else:
        print(f"Response: {message['content']}")


def performance_benchmark():
    """Simple performance benchmark."""
    print("\n=== Performance Benchmark ===")

    prompts = [
        "Write a short poem about speed.",
        "List 5 programming languages.",
        "Explain quantum computing in simple terms.",
    ]

    total_time = 0
    total_tokens = 0

    for prompt in prompts:
        start_time = time.time()

        response = client.chat.completions.create(
            model="groq/llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )

        end_time = time.time()
        elapsed = end_time - start_time
        tokens = response.usage["total_tokens"]

        total_time += elapsed
        total_tokens += tokens

        print(f"Prompt: '{prompt[:30]}...' - Time: {elapsed:.3f}s, Tokens: {tokens}")

    print(f"\nAverage response time: {total_time/len(prompts):.3f}s")
    print(f"Average tokens/second: {total_tokens/total_time:.1f}")


def main():
    """Run all examples."""
    print("Groq Provider Examples - Ultra-Fast LPU Inference\n")

    # Check if API key is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY environment variable")
        print("Get your API key from: https://console.groq.com/keys")
        return

    try:
        basic_chat_completion()
        streaming_example()
        json_mode_example()
        different_models_comparison()
        tool_use_example()
        performance_benchmark()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

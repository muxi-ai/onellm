#!/usr/bin/env python3
"""
Fireworks AI Provider Example

This example demonstrates how to use the Fireworks AI provider with OneLLM.
Fireworks provides fast inference for open models with optimizations.
"""

import os
import time
from onellm import OpenAI

# Initialize the client (it automatically detects Fireworks from the model name)
# Uses FIREWORKS_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Fireworks."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What makes Fireworks AI special for model inference?"},  # noqa: E501
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming with Fireworks' optimized inference."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[
            {"role": "user", "content": "Explain the benefits of model optimization in 3 points."},
        ],
        stream=True,
        max_tokens=150,
    )

    print("Streaming response: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def speed_test_example():
    """Test inference speed with different models."""
    print("\n=== Speed Test Example ===")

    models = [
        ("fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct", "Llama 3.1 8B"),
        ("fireworks/accounts/fireworks/models/mixtral-8x7b-instruct", "Mixtral 8x7B"),
        ("fireworks/accounts/fireworks/models/gemma-7b-it", "Gemma 7B"),
    ]

    prompt = "What is 2+2?"

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            start_time = time.time()

            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )

            end_time = time.time()
            tokens = response.usage["total_tokens"]

            print(f"Response: {response.choices[0].message['content']}")
            print(f"Time: {end_time - start_time:.3f}s")
            print(f"Tokens/sec: {tokens / (end_time - start_time):.1f}")

        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")

    response = client.chat.completions.create(
        model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {
                "role": "user",
                "content": (
                    "Create a JSON object describing 3 optimization techniques used in model inference."  # noqa: E501
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=300,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def function_calling_example():
    """Function calling with Fireworks."""
    print("\n=== Function Calling Example ===")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_inference_cost",
                "description": "Calculate the cost of model inference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_size": {
                            "type": "string",
                            "description": "Size of the model (e.g., 7B, 13B, 70B)",
                        },
                        "tokens": {"type": "integer", "description": "Number of tokens to process"},
                        "batch_size": {
                            "type": "integer",
                            "description": "Batch size for inference",
                        },
                    },
                    "required": ["model_size", "tokens"],
                },
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "How much would it cost to process 10,000 tokens with a 13B model?",
                }
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

    except Exception as e:
        print(f"Function calling might not be fully supported: {str(e)[:150]}")


def code_generation_example():
    """Code generation with optimized models."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a Python function that implements a simple LRU cache with get and put methods.",  # noqa: E501
                ),
            }
        ],
        temperature=0.2,
        max_tokens=400,
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def batch_processing_example():
    """Example of processing multiple prompts efficiently."""
    print("\n=== Batch Processing Example ===")

    prompts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?",
        "What is transformer architecture?",
    ]

    print("Processing multiple prompts...")
    total_time = 0

    for i, prompt in enumerate(prompts):
        start_time = time.time()

        response = client.chat.completions.create(
            model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )

        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        print(f"\nPrompt {i+1}: '{prompt}'")
        print(f"Response: {response.choices[0].message['content'][:100]}...")
        print(f"Time: {elapsed:.3f}s")

    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.3f}s")


def multilingual_example():
    """Test multilingual capabilities."""
    print("\n=== Multilingual Example ===")

    languages = {
        "Spanish": "¿Cómo optimiza Fireworks AI los modelos de lenguaje?",
        "French": "Comment Fireworks AI optimise-t-il les modèles de langage?",
        "German": "Wie optimiert Fireworks AI Sprachmodelle?",
    }

    for lang, prompt in languages.items():
        response = client.chat.completions.create(
            model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{"role": "user", "content": f"{prompt} (Please respond in {lang})"}],  # noqa: E501
            max_tokens=100,
        )

        print(f"\n{lang} Question: {prompt}")
        print(f"Response: {response.choices[0].message['content']}")


def main():
    """Run all examples."""
    print("Fireworks AI Provider Examples\n")
    print("Fast, optimized inference for open models!\n")

    # Check if API key is set
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Please set FIREWORKS_API_KEY environment variable")
        print("Get your API key from: https://app.fireworks.ai/")
        return

    try:
        basic_chat_completion()
        streaming_example()
        speed_test_example()
        json_mode_example()
        function_calling_example()
        code_generation_example()
        batch_processing_example()
        multilingual_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

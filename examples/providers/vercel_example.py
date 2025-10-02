#!/usr/bin/env python3
"""
Vercel AI Gateway Provider Example

This example demonstrates how to use the Vercel AI Gateway provider with OneLLM.
Vercel AI Gateway provides access to 100+ models through a unified OpenAI-compatible API.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Vercel from the model name)
# Uses VERCEL_AI_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Vercel AI Gateway."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="vercel/openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Vercel AI Gateway and why is it useful?"},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming with Vercel AI Gateway."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="vercel/openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about cloud computing."}],
        stream=True,
        max_tokens=100,
    )

    print("Streaming haiku: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def model_variety_example():
    """Demonstrate access to various models through Vercel AI Gateway."""
    print("\n=== Model Variety Example ===")

    # Different models available through Vercel AI Gateway
    models = [
        ("vercel/openai/gpt-4o-mini", "GPT-4o Mini"),
        ("vercel/anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        ("vercel/google/gemini-2.0-flash-exp", "Gemini 2.0 Flash"),
        ("vercel/meta/llama-3.3-70b-instruct", "Llama 3.3 70B"),
    ]

    prompt = "What is machine learning in one sentence?"

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5,
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def multi_provider_example():
    """Example showing access to different AI providers."""
    print("\n=== Multi-Provider Example ===")
    print("Access models from OpenAI, Anthropic, Google, and more!")

    providers_and_models = [
        ("vercel/openai/gpt-4o-mini", "OpenAI"),
        ("vercel/anthropic/claude-sonnet-4", "Anthropic"),
        ("vercel/google/gemini-2.0-flash-exp", "Google"),
    ]

    for model_id, provider_name in providers_and_models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": f"In one sentence, what makes {provider_name}'s AI unique?",
                    }
                ],
                max_tokens=100,
            )
            print(f"\n{provider_name}: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {provider_name}: {str(e)[:150]}")


def json_mode_example():
    """JSON mode with Vercel AI Gateway."""
    print("\n=== JSON Mode Example ===")

    response = client.chat.completions.create(
        model="vercel/openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {
                "role": "user",
                "content": "List 3 benefits of using Vercel AI Gateway in JSON format.",
            },
        ],
        max_tokens=200,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation through Vercel AI Gateway."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="vercel/anthropic/claude-sonnet-4",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to calculate the factorial of a number using recursion.",  # noqa: E501
            }
        ],
        temperature=0.2,
        max_tokens=200,
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def multi_language_example():
    """Multi-language support through Vercel AI Gateway."""
    print("\n=== Multi-Language Example ===")

    languages = [
        ("English", "Hello, how are you?"),
        ("Spanish", "¿Cómo estás?"),
        ("French", "Comment allez-vous?"),
        ("German", "Wie geht es dir?"),
    ]

    for lang, text in languages:
        response = client.chat.completions.create(
            model="vercel/openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Translate '{text}' and respond in the same language.",
                },
            ],
            max_tokens=50,
        )
        print(f"\n{lang}: {text}")
        print(f"Response: {response.choices[0].message['content']}")


def vision_example():
    """Vision capabilities through Vercel AI Gateway."""
    print("\n=== Vision Example ===")
    print("Note: This requires a model with vision capabilities.")

    try:
        response = client.chat.completions.create(
            model="vercel/google/gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"  # noqa: E501
                            },
                        },
                    ],
                }
            ],
            max_tokens=150,
        )
        print(f"Vision response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Vision example not available: {str(e)[:150]}")


def main():
    """Run all examples."""
    print("Vercel AI Gateway Provider Examples\n")
    print("Access 100+ models through a unified OpenAI-compatible API!\n")

    # Check if API key is set
    if not os.environ.get("VERCEL_AI_API_KEY"):
        print("Please set VERCEL_AI_API_KEY environment variable")
        print("Get your API key from: https://vercel.com/ai-gateway")
        return

    try:
        basic_chat_completion()
        streaming_example()
        model_variety_example()
        multi_provider_example()
        json_mode_example()
        code_generation_example()
        multi_language_example()
        vision_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

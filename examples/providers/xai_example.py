#!/usr/bin/env python3
"""
X.AI (Grok) Provider Example

This example demonstrates how to use the X.AI provider with OneLLM.
X.AI offers Grok models with a 128K context window.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects X.AI from the model name)
# Uses XAI_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Grok."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {"role": "system", "content": "You are Grok, a helpful AI assistant."},
            {"role": "user", "content": "What makes Grok unique compared to other AI models?"},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Grok."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[{"role": "user", "content": "Tell me a joke about artificial intelligence."}],
        stream=True,
        max_tokens=150,
    )

    print("Streaming response: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def long_context_example():
    """Demonstrate Grok's 128K context window."""
    print("\n=== Long Context Example ===")

    # Create a long context
    long_context = (
        """
    In the year 2045, humanity achieved a breakthrough in quantum computing that changed everything.
    The new quantum processors could solve problems in seconds that would take classical computers
    millions of years.

    This led to rapid advances in medicine, climate modeling, and space exploration.

    Dr. Sarah Chen was at the forefront of this revolution. Her team at the
    Quantum Research Institute had developed the first stable quantum computer that could
    operate at room temperature. This eliminated the need for expensive cooling systems
    and made quantum computing accessible to laboratories around the world.

    The implications were staggering. Drug discovery accelerated by a factor of 1000. Weather
    predictions became accurate up to 6 months in advance. And perhaps most importantly, researchers
    finally cracked the code for efficient fusion energy, solving the world's energy crisis.
    """
        * 3
    )  # Repeat to make it longer

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {
                "role": "user",
                "content": f"Based on this story, what were the three most important breakthroughs mentioned?\n\nStory: {long_context}",  # noqa: E501
            }
        ],
        max_tokens=200,
    )

    print(f"Grok's analysis: {response.choices[0].message['content']}")


def reasoning_example():
    """Complex reasoning with Grok."""
    print("\n=== Reasoning Example ===")

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {
                "role": "user",
                "content": """If all Bloops are Bleeps, and some Bleeps are Blops, and no Blops are Blips,
                can we conclude that some Bloops are not Blips? Explain your reasoning step by step.""",  # noqa: E501
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )

    print("Logical reasoning:")
    print(response.choices[0].message["content"])


def creative_writing_example():
    """Creative writing with Grok."""
    print("\n=== Creative Writing Example ===")

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {"role": "system", "content": "You are a creative writer with a unique perspective."},
            {
                "role": "user",
                "content": "Write the opening paragraph of a science fiction story set on Mars in the year 2150.",  # noqa: E501
            },
        ],
        temperature=0.9,
        max_tokens=200,
    )

    print("Story opening:")
    print(response.choices[0].message["content"])


def code_generation_example():
    """Code generation with Grok."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function that implements a binary search tree with insert, search, and delete operations.",  # noqa: E501
            }
        ],
        temperature=0.2,
        max_tokens=500,
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def analysis_example():
    """Data analysis example."""
    print("\n=== Analysis Example ===")

    data = """
    Sales Q1: $1.2M (up 15% YoY)
    Sales Q2: $1.5M (up 25% YoY)
    Sales Q3: $1.3M (up 8% YoY)
    Sales Q4: $1.8M (up 20% YoY)

    Top products: Widget A (35%), Gadget B (25%), Tool C (20%), Other (20%)
    Top regions: North America (45%), Europe (30%), Asia (20%), Other (5%)
    """

    response = client.chat.completions.create(
        model="xai/grok-2-latest",
        messages=[
            {
                "role": "user",
                "content": f"Analyze this sales data and provide 3 key insights:\n\n{data}",
            }
        ],
        temperature=0.5,
        max_tokens=300,
    )

    print("Analysis:")
    print(response.choices[0].message["content"])


def different_grok_models():
    """Compare different Grok models if available."""
    print("\n=== Grok Models ===")

    models = [
        ("xai/grok-2-latest", "Grok 2 Latest"),
        ("xai/grok-2", "Grok 2"),
        ("xai/grok-1", "Grok 1 (if available)"),
    ]

    prompt = "What is the meaning of life in exactly one sentence?"

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7,
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Note: {model_id} might not be available: {str(e)[:100]}")


def main():
    """Run all examples."""
    print("X.AI (Grok) Provider Examples\n")

    # Check if API key is set
    if not os.environ.get("XAI_API_KEY"):
        print("Please set XAI_API_KEY environment variable")
        print("Get your API key from: https://x.ai/")
        return

    try:
        basic_chat_completion()
        streaming_example()
        long_context_example()
        reasoning_example()
        creative_writing_example()
        code_generation_example()
        analysis_example()
        different_grok_models()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

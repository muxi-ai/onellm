#!/usr/bin/env python3
"""
DeepSeek Provider Example

This example demonstrates how to use the DeepSeek provider with OneLLM.
DeepSeek is a Chinese AI company offering multilingual LLMs.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects DeepSeek from the model name)
# Uses DEEPSEEK_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with DeepSeek."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful multilingual assistant."},
            {"role": "user", "content": "What makes DeepSeek unique in the global AI landscape?"},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with DeepSeek."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "Write a short story about AI helping humanity, in exactly 3 sentences.",
            }
        ],
        stream=True,
        max_tokens=150,
    )

    print("Streaming story: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def multilingual_example():
    """Demonstrate DeepSeek's multilingual capabilities."""
    print("\n=== Multilingual Example ===")

    languages = [
        ("English", "How does artificial intelligence work?"),
        ("Chinese", "人工智能是如何工作的？"),
        ("Spanish", "¿Cómo funciona la inteligencia artificial?"),
        ("Japanese", "人工知能はどのように機能しますか？"),
        ("Korean", "인공지능은 어떻게 작동하나요?"),
    ]

    for lang, question in languages:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": f"{question} (Please answer in {lang})"}],
            max_tokens=100,
            temperature=0.5,
        )

        print(f"\n{lang}: {question}")
        print(f"Response: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation with DeepSeek."""
    print("\n=== Code Generation Example ===")

    response = client.chat.completions.create(
        model="deepseek/deepseek-coder",  # Using coder model if available
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a Python function to convert temperature between "
                    "Celsius, Fahrenheit, and Kelvin."
                ),
            }
        ],
        temperature=0.2,
        max_tokens=400,
    )

    print("Generated code:")
    print(response.choices[0].message["content"])


def chinese_specific_example():
    """Example focusing on Chinese language tasks."""
    print("\n=== Chinese Language Tasks ===")

    tasks = [
        {
            "task": "Translation",
            "prompt": "Translate to Chinese: 'Artificial intelligence is transforming the world.'",
        },
        {
            "task": "Poetry",
            "prompt": "写一首关于春天的五言绝句。(Write a 5-character quatrain about spring)",
        },
        {
            "task": "Explanation",
            "prompt": "用简单的中文解释什么是机器学习。(Explain machine learning in simple Chinese)",
        },
    ]

    for task_info in tasks:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": task_info["prompt"]}],
            max_tokens=200,
        )

        print(f"\n{task_info['task']}:")
        print(f"Prompt: {task_info['prompt']}")
        print(f"Response: {response.choices[0].message['content']}")


def mathematical_reasoning_example():
    """Mathematical problem solving with DeepSeek."""
    print("\n=== Mathematical Reasoning ===")

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": """Solve this problem step by step:

                A train travels from City A to City B at 80 km/h and returns at 120 km/h.
                What is the average speed for the entire round trip?""",
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )

    print("Solution:")
    print(response.choices[0].message["content"])


def json_output_example():
    """JSON structured output with DeepSeek."""
    print("\n=== JSON Output Example ===")

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
            {
                "role": "user",
                "content": (
                    "Create a JSON object with information about 3 major Chinese tech companies."
                ),
            },
        ],
        max_tokens=300,
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def different_deepseek_models():
    """Try different DeepSeek model variants."""
    print("\n=== Different DeepSeek Models ===")

    models = [
        ("deepseek/deepseek-chat", "DeepSeek Chat (general purpose)"),
        ("deepseek/deepseek-coder", "DeepSeek Coder (code generation)"),
        ("deepseek/deepseek-llm-67b-chat", "DeepSeek 67B (if available)"),
    ]

    prompt = "What is the difference between machine learning and deep learning?"

    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=100
            )
            print(f"Response: {response.choices[0].message['content'][:150]}...")
        except Exception as e:
            print(f"Note: {model_id} might not be available: {str(e)[:100]}")


def conversation_example():
    """Multi-turn conversation with context."""
    print("\n=== Multi-turn Conversation ===")

    messages = [
        {"role": "user", "content": "I want to learn about Chinese culture."},
        {
            "role": "assistant",
            "content": (
                "That's wonderful! Chinese culture is rich and diverse, spanning over 5,000 years. "
                "What aspect interests you most - history, philosophy, art, cuisine, or modern culture?"  # noqa: E501
            ),
        },
        {"role": "user", "content": "Tell me about Chinese philosophy, especially Confucianism."},
    ]

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat", messages=messages, max_tokens=300
    )

    print("Conversation about Chinese philosophy:")
    print(response.choices[0].message["content"])


def main():
    """Run all examples."""
    print("DeepSeek Provider Examples\n")
    print("Multilingual AI with strong Chinese language support!\n")

    # Check if API key is set
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Please set DEEPSEEK_API_KEY environment variable")
        print("Get your API key from: https://platform.deepseek.com/")
        return

    try:
        basic_chat_completion()
        streaming_example()
        multilingual_example()
        code_generation_example()
        chinese_specific_example()
        mathematical_reasoning_example()
        json_output_example()
        different_deepseek_models()
        conversation_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

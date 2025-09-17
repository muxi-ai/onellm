#!/usr/bin/env python3
"""
OpenAI Provider Example

This example demonstrates how to use the OpenAI provider with OneLLM,
including chat completions, streaming, function calling, and embeddings.
"""

import os
from onellm import OpenAI

# Initialize the OpenAI client
# Uses OPENAI_API_KEY environment variable by default
client = OpenAI()

# You can also explicitly set the API key
# client = OpenAI(api_key="your-api-key-here")


def basic_chat_completion():
    """Basic chat completion example."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."},
        ],
        temperature=0.7,
        max_tokens=150,
    )

    print(f"Response: {response.choices[0].message['content']}")
    print(f"Tokens used: {response.usage}")


def streaming_example():
    """Streaming response example."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
        stream=True,
    )

    print("Streaming response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print()


def function_calling_example():
    """Function calling example."""
    print("\n=== Function Calling Example ===")

    # Define a function for the model to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather like in New York?"}],
        tools=tools,
        tool_choice="auto",
    )

    # Check if the model wants to call a function
    message = response.choices[0].message
    if message.get("tool_calls"):
        print(f"Model wants to call: {message['tool_calls'][0]['function']['name']}")
        print(f"With arguments: {message['tool_calls'][0]['function']['arguments']}")
    else:
        print(f"Response: {message['content']}")


def embedding_example():
    """Embedding generation example."""
    print("\n=== Embedding Example ===")

    # Generate embeddings for text
    response = client.embeddings.create(
        model="openai/text-embedding-3-small", input=["Hello, world!", "How are you today?"]
    )

    for i, embedding in enumerate(response.data):
        print(f"Embedding {i+1}: {len(embedding['embedding'])} dimensions")
        print(f"First 5 values: {embedding['embedding'][:5]}")


def json_mode_example():
    """JSON mode example for structured output."""
    print("\n=== JSON Mode Example ===")

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": "List 3 programming languages with their key features."},
        ],
        response_format={"type": "json_object"},
    )

    print(f"JSON Response: {response.choices[0].message['content']}")


def main():
    """Run all examples."""
    print("OpenAI Provider Examples\n")

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    try:
        basic_chat_completion()
        streaming_example()
        function_calling_example()
        embedding_example()
        json_mode_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

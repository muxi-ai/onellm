"""
Example demonstrating the use of the ChatCompletion module.

This example shows how to use the ChatCompletion class to interact with
various LLM providers via a unified interface.
"""

import asyncio
import os
from typing import List, Dict, Any, Callable

from muxi_llm import ChatCompletion
from muxi_llm.models import ChatCompletionResponse, ChatCompletionChunk
from muxi_llm.config import set_api_key


def print_chat_response(response: ChatCompletionResponse) -> None:
    """Print the response from a chat completion."""
    print("\n--- Chat Completion Response ---")
    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message['content']}")
    if response.usage:
        print(f"Usage: {response.usage}")


async def streaming_demo(
    model: str,
    messages: List[Dict[str, Any]],
    callback: Callable[[str], None] = print
) -> None:
    """Demonstrate streaming chat completions."""
    print(f"\n--- Streaming Chat Completion ({model}) ---")
    print("Response:", end=" ", flush=True)

    # Call with streaming=True to get a generator
    stream = await ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True
    )

    # Process the streaming response chunks
    async for chunk in stream:
        if isinstance(chunk, ChatCompletionChunk):
            for choice in chunk.choices:
                if choice.delta.content:
                    callback(choice.delta.content)


def non_streaming_demo(model: str, messages: List[Dict[str, Any]]) -> None:
    """Demonstrate non-streaming chat completions."""
    # Use the synchronous API
    response = ChatCompletion.create(
        model=model,
        messages=messages
    )
    print_chat_response(response)


async def async_non_streaming_demo(model: str, messages: List[Dict[str, Any]]) -> None:
    """Demonstrate asynchronous non-streaming chat completions."""
    # Use the asynchronous API
    response = await ChatCompletion.acreate(
        model=model,
        messages=messages
    )
    print_chat_response(response)


async def main() -> None:
    """Run the example."""
    # Set API keys from environment variables
    # You can also pass these directly to the API calls via api_key parameter
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment.")
    else:
        set_api_key(openai_api_key, "openai")

    if not anthropic_api_key:
        print("Note: ANTHROPIC_API_KEY not found in environment. "
              "Anthropic examples will be skipped.")
    else:
        set_api_key(anthropic_api_key, "anthropic")

    # Example messages for a chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]

    # Demonstrate different usage patterns and providers
    if openai_api_key:
        # 1. Using provider prefix (required)
        non_streaming_demo("openai/gpt-3.5-turbo", messages)

        # 2. Asynchronous usage
        await async_non_streaming_demo("openai/gpt-3.5-turbo", messages)

        # 3. Streaming usage
        def print_streaming(text: str) -> None:
            print(text, end="", flush=True)

        await streaming_demo("openai/gpt-3.5-turbo", messages, print_streaming)
        print()  # New line after streaming response

    # Demonstrate Anthropic model (if API key is available)
    if anthropic_api_key:
        anthropic_messages = [
            {"role": "user", "content": "What are the three laws of robotics?"}
        ]

        # 4. Using Anthropic model (Claude)
        print("\n--- Using Anthropic Model ---")
        await async_non_streaming_demo("anthropic/claude-3-opus", anthropic_messages)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

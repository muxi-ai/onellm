"""
Example demonstrating the fallback mechanism in muxi-llm.

This example shows how to use fallbacks to increase reliability when a primary model
is unavailable or experiences errors.
"""

import asyncio
import os
import logging

from muxi.llm import ChatCompletion, Completion, Embedding
from muxi.llm.errors import RateLimitError

# Set up logging to see fallback messages
logging.basicConfig(level=logging.INFO)


def set_api_keys_from_env():
    """Set API keys from environment variables."""
    from muxi.llm import set_api_key

    # Set API keys for different providers
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        set_api_key(openai_key, "openai")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        set_api_key(anthropic_key, "anthropic")

    cohere_key = os.environ.get("COHERE_API_KEY")
    if cohere_key:
        set_api_key(cohere_key, "cohere")


async def demonstrate_chat_completion_fallback():
    """Demonstrate fallback for chat completions."""
    print("\n=== Chat Completion Fallback Demo ===")

    messages = [{"role": "user", "content": "What are three interesting facts about the moon?"}]

    try:
        # Scenario 1: Use a non-existent model to force fallback
        print("\nScenario 1: Fallback from non-existent model to valid model")
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            fallback_models=[
                "openai/gpt-3.5-turbo",  # This should work
                "anthropic/claude-3-haiku"  # Backup if the first fallback also fails
            ],
            fallback_config={
                "log_fallbacks": True
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")

    try:
        # Scenario 2: Custom fallback configuration
        print("\nScenario 2: Custom fallback configuration with max_fallbacks=1")
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model-1",  # This will fail
            messages=messages,
            fallback_models=[
                "openai/non-existent-model-2",  # This will also fail
                # This would work but won't be tried due to max_fallbacks=1:
                "openai/gpt-3.5-turbo"
            ],
            fallback_config={
                "max_fallbacks": 1,  # Only try the first fallback
                "log_fallbacks": True
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ Expected failure due to max_fallbacks=1: {e}")


async def demonstrate_completion_fallback():
    """Demonstrate fallback for text completions."""
    print("\n=== Text Completion Fallback Demo ===")

    prompt = "Write a haiku about programming:"

    try:
        response = await Completion.acreate(
            model="openai/non-existent-model",  # This will fail
            prompt=prompt,
            fallback_models=["openai/gpt-3.5-turbo-instruct"],  # This should work
            fallback_config={
                "log_fallbacks": True,
                "retriable_errors": [RateLimitError]  # Only retry on rate limit errors
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Text: {response.choices[0].text}")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def demonstrate_embedding_fallback():
    """Demonstrate fallback for embeddings."""
    print("\n=== Embedding Fallback Demo ===")

    texts = ["The quick brown fox jumps over the lazy dog"]

    try:
        response = await Embedding.acreate(
            model="openai/non-existent-embedding-model",  # This will fail
            input=texts,
            fallback_models=["openai/text-embedding-ada-002"],  # This should work
            fallback_config={
                "log_fallbacks": True
            }
        )

        print("✅ Success! Got embeddings from fallback model:")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def custom_fallback_callback(primary_model: str, fallback_model: str, error: Exception):
    """Example callback function when fallbacks are used."""
    print("\n🔄 Fallback callback triggered:")
    print(f"  - Primary model: {primary_model}")
    print(f"  - Fallback model used: {fallback_model}")
    print(f"  - Error from primary model: {type(error).__name__}: {str(error)}")

    # You could send metrics, log to a monitoring system, or take other actions here


async def demonstrate_fallback_callback():
    """Demonstrate using a callback when fallbacks occur."""
    print("\n=== Fallback Callback Demo ===")

    messages = [{"role": "user", "content": "What's your favorite programming language?"}]

    try:
        response = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            fallback_models=["openai/gpt-3.5-turbo"],  # This should work
            fallback_config={
                "log_fallbacks": True,
                "fallback_callback": custom_fallback_callback
            }
        )

        print("✅ Success! Response from fallback model:")
        print(f"Content: {response.choices[0].message['content'][:150]}...")

    except Exception as e:
        print(f"❌ All fallbacks failed: {e}")


async def demonstrate_streaming_fallback():
    """Demonstrate fallback for streaming responses."""
    print("\n=== Streaming Fallback Demo ===")

    messages = [{"role": "user", "content": "Count from 1 to 5 slowly."}]

    try:
        print("\nStarting streaming with fallback:")
        stream = await ChatCompletion.acreate(
            model="openai/non-existent-model",  # This will fail
            messages=messages,
            stream=True,  # Important: Enable streaming
            fallback_models=["openai/gpt-3.5-turbo"],  # This should work
            fallback_config={
                "log_fallbacks": True
            }
        )

        # Process the streaming response chunks
        response_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.get("content"):
                content = chunk.choices[0].delta["content"]
                response_text += content
                print(content, end="", flush=True)

        print("\n✅ Streaming complete!")

    except Exception as e:
        print(f"\n❌ All fallbacks failed: {e}")


async def run_demos():
    """Run all the demonstration functions."""
    # Set API keys
    set_api_keys_from_env()

    # Run demos
    await demonstrate_chat_completion_fallback()
    await demonstrate_completion_fallback()
    await demonstrate_embedding_fallback()
    await demonstrate_fallback_callback()
    await demonstrate_streaming_fallback()


if __name__ == "__main__":
    asyncio.run(run_demos())

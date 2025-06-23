#!/usr/bin/env python3
"""
Example of using Ollama provider with OneLLM.

This example demonstrates:
1. Basic usage with default localhost
2. Using different Ollama servers for different models
3. Vision model support
4. Listing available models
"""

import asyncio
from onellm import Client
from onellm.providers import get_provider


async def basic_ollama_example():
    """Basic example using Ollama with default settings."""
    print("=== Basic Ollama Example ===")
    
    client = Client()
    
    # Using default localhost:11434
    response = await client.chat.completions.create(
        model="ollama/llama3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Ollama and why is it useful?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print()


async def multi_server_example():
    """Example using different Ollama servers for different models."""
    print("=== Multi-Server Example ===")
    
    client = Client()
    
    # Simulating different servers (replace with your actual servers)
    models = [
        "ollama/llama3:8b",  # Default localhost:11434
        "ollama/mistral:7b@localhost:11434",  # Explicit localhost
        # "ollama/mixtral:8x7b@gpu-server:11434",  # Remote server example
        # "ollama/llama3:70b-instruct-q4_K_M@10.0.0.5:11434",  # IP address
    ]
    
    for model in models:
        try:
            print(f"Using model: {model}")
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                max_tokens=50
            )
            print(f"Response: {response.choices[0].message.content}")
            print()
        except Exception as e:
            print(f"Error with {model}: {e}")
            print()


async def streaming_example():
    """Example of streaming responses from Ollama."""
    print("=== Streaming Example ===")
    
    client = Client()
    
    print("Streaming response from Ollama:")
    stream = await client.chat.completions.create(
        model="ollama/llama3:8b",
        messages=[{"role": "user", "content": "Write a haiku about AI."}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


async def vision_model_example():
    """Example using vision models (if available)."""
    print("=== Vision Model Example ===")
    
    client = Client()
    
    # Example with a vision model like llava
    try:
        response = await client.chat.completions.create(
            model="ollama/llava:latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }],
            max_tokens=100
        )
        print(f"Vision response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Vision model not available or error: {e}")
    print()


async def list_models_example():
    """Example of listing available models on Ollama servers."""
    print("=== List Models Example ===")
    
    # Direct provider usage for listing models
    ollama = get_provider("ollama")
    
    # List models on default server
    try:
        models = await ollama.list_models()
        print(f"Models available on localhost:11434:")
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # List models on a different server
    # try:
    #     models = await ollama.list_models("http://gpu-server:11434")
    #     print(f"\nModels available on gpu-server:11434:")
    #     for model in models:
    #         print(f"  - {model}")
    # except Exception as e:
    #     print(f"Error listing models on gpu-server: {e}")
    
    print()


async def custom_parameters_example():
    """Example using Ollama-specific parameters."""
    print("=== Custom Parameters Example ===")
    
    client = Client()
    
    # Ollama-specific parameters
    response = await client.chat.completions.create(
        model="ollama/llama3:8b",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=50,
        # Ollama-specific parameters
        num_gpu=1,  # Number of GPU layers
        num_thread=8,  # Number of CPU threads
        num_ctx=2048,  # Context window size
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print()


async def main():
    """Run all examples."""
    print("OneLLM Ollama Provider Examples")
    print("==============================\n")
    
    # Make sure Ollama is running
    print("Note: Make sure Ollama is running (ollama serve)")
    print("Pull models if needed: ollama pull llama3:8b\n")
    
    # Run examples
    await basic_ollama_example()
    await multi_server_example()
    await streaming_example()
    await vision_model_example()
    await list_models_example()
    await custom_parameters_example()


if __name__ == "__main__":
    asyncio.run(main())
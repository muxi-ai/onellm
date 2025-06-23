#!/usr/bin/env python3
"""
Example of using llama.cpp provider with OneLLM.

This example demonstrates:
1. Basic usage with model from default directory
2. Using full path to model
3. GPU acceleration
4. Streaming responses
5. Custom parameters

Note: This requires llama-cpp-python to be installed.
See docs/llama_cpp_tutorial.md for installation instructions.
"""

import asyncio
import os
from pathlib import Path
from onellm import Client
from onellm.providers import get_provider


async def basic_example():
    """Basic example using llama.cpp with default settings."""
    print("=== Basic llama.cpp Example ===")
    
    # Make sure we have a model directory
    model_dir = Path.home() / "llama_models"
    if not model_dir.exists():
        print(f"Note: Create {model_dir} and download GGUF models there")
        print("Example: wget https://huggingface.co/[model-path]/model.gguf")
        return
    
    client = Client()
    
    try:
        # Using model from default directory
        response = await client.chat.completions.create(
            model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is llama.cpp and why is it useful?"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model: {response.model}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a model file in ~/llama_models/")
        print()


async def full_path_example():
    """Example using full path to model."""
    print("=== Full Path Example ===")
    
    client = Client()
    
    # Replace with your actual model path
    model_path = "/path/to/your/model.gguf"
    
    try:
        response = await client.chat.completions.create(
            model=f"llama_cpp/{model_path}",
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            max_tokens=50
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print()
    except Exception as e:
        print(f"Note: Update model_path in the example to your actual model file")
        print(f"Error: {e}")
        print()


async def gpu_example():
    """Example using GPU acceleration."""
    print("=== GPU Acceleration Example ===")
    
    client = Client()
    
    try:
        # Enable GPU acceleration
        response = await client.chat.completions.create(
            model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
            messages=[
                {"role": "user", "content": "What are the benefits of GPU acceleration?"}
            ],
            max_tokens=150,
            n_gpu_layers=32,  # Use GPU for 32 layers
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print("\nNote: GPU acceleration requires llama-cpp-python built with GPU support")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


async def streaming_example():
    """Example of streaming responses."""
    print("=== Streaming Example ===")
    
    client = Client()
    
    try:
        print("Streaming response:")
        stream = await client.chat.completions.create(
            model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
            messages=[{"role": "user", "content": "Write a haiku about local AI."}],
            stream=True,
            max_tokens=100
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}")
        print()


async def custom_parameters_example():
    """Example with custom parameters."""
    print("=== Custom Parameters Example ===")
    
    client = Client()
    
    try:
        response = await client.chat.completions.create(
            model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
            messages=[
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            max_tokens=200,
            # llama.cpp specific parameters
            n_ctx=4096,       # Larger context window
            n_gpu_layers=0,   # CPU only
            n_threads=8,      # Number of CPU threads
            temperature=0.3,  # Lower temperature for focused response
            top_k=40,
            top_p=0.95
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


async def list_models_example():
    """Example of listing available models."""
    print("=== List Available Models ===")
    
    # Direct provider usage
    try:
        provider = get_provider("llama_cpp")
        models = provider.list_available_models()
        
        if models:
            print(f"Available models in {provider.model_dir}:")
            for model in models:
                print(f"  - {model}")
        else:
            print(f"No models found in {provider.model_dir}")
            print("Download GGUF models to this directory to use them")
    except Exception as e:
        print(f"Error: {e}")
    print()


async def main():
    """Run all examples."""
    print("OneLLM llama.cpp Provider Examples")
    print("==================================\n")
    
    # Check if llama-cpp-python is installed
    try:
        import llama_cpp
        print("✓ llama-cpp-python is installed")
    except ImportError:
        print("✗ llama-cpp-python is not installed")
        print("\nInstall it with:")
        print("  pip install llama-cpp-python  # CPU only")
        print("  CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python  # Mac GPU")
        print("  CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python  # NVIDIA GPU")
        print("\nSee docs/llama_cpp_tutorial.md for details")
        return
    
    print(f"Model directory: {os.path.expanduser('~/llama_models')}")
    print()
    
    # Run examples
    await basic_example()
    await full_path_example()
    await gpu_example()
    await streaming_example()
    await custom_parameters_example()
    await list_models_example()


if __name__ == "__main__":
    asyncio.run(main())
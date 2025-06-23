#!/usr/bin/env python3
"""
Example showing how to download and use a GGUF model with OneLLM.

This demonstrates the complete workflow:
1. Download a model using the CLI
2. Use it with the llama.cpp provider
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from onellm import Client


def download_model():
    """Download a model using OneLLM's CLI utility."""
    print("=== Downloading Model ===")
    print("This example will download Phi-3 Mini (small, fast model)")
    print()
    
    # Run the download command
    cmd = [
        sys.executable, "-m", "onellm.cli",
        "download",
        "--repo-id", "microsoft/Phi-3-mini-4k-instruct-gguf",
        "--filename", "Phi-3-mini-4k-instruct-q4.gguf"
    ]
    
    print("Running:", " ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Failed to download model")
        return False
    
    return True


async def use_model():
    """Use the downloaded model."""
    print("\n=== Using Downloaded Model ===")
    
    client = Client()
    
    try:
        # Use the model we just downloaded
        response = await client.chat.completions.create(
            model="llama_cpp/Phi-3-mini-4k-instruct-q4.gguf",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"\nModel used: {response.model}")
        
    except Exception as e:
        print(f"Error using model: {e}")
        print("\nMake sure:")
        print("1. llama-cpp-python is installed")
        print("2. The model was downloaded successfully")
        print("3. The model path is correct")


async def main():
    """Main example flow."""
    print("OneLLM Model Download and Usage Example")
    print("======================================\n")
    
    # Check if model already exists
    model_path = Path.home() / "llama_models" / "Phi-3-mini-4k-instruct-q4.gguf"
    
    if model_path.exists():
        print(f"Model already exists at: {model_path}")
        use_existing = input("Use existing model? (y/n): ").lower()
        if use_existing != 'y':
            if not download_model():
                return
    else:
        print("Model not found locally. Downloading...")
        if not download_model():
            return
    
    # Use the model
    await use_model()
    
    print("\n=== Complete! ===")
    print("You can now use this model in your applications with:")
    print('  model="llama_cpp/Phi-3-mini-4k-instruct-q4.gguf"')


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Example of using Azure OpenAI provider with OneLLM.

This example demonstrates how to use Azure OpenAI through OneLLM's unified interface.
"""

import asyncio
import os
from onellm import Client

async def main():
    # Initialize the client
    # The Azure provider will look for azure.json in the project root by default
    # Or you can specify a custom path via AZURE_OPENAI_CONFIG_PATH environment variable
    client = Client()
    
    # Example 1: Chat completion with o4-mini
    print("Example 1: Chat completion with o4-mini")
    print("-" * 50)
    
    response = await client.chat.completions.create(
        model="azure/o4-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print()
    
    # Example 2: Chat completion with gpt-4o-mini
    print("Example 2: Chat completion with gpt-4o-mini")
    print("-" * 50)
    
    response = await client.chat.completions.create(
        model="azure/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a haiku about Azure cloud computing."}
        ],
        temperature=0.9
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print()
    
    # Example 3: Streaming chat completion
    print("Example 3: Streaming chat completion")
    print("-" * 50)
    
    stream = await client.chat.completions.create(
        model="azure/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ],
        stream=True
    )
    
    print("Streaming response: ", end="")
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
    # Example 4: Using specific deployment configuration
    print("Example 4: Custom deployment configuration")
    print("-" * 50)
    
    # You can also initialize the provider with a custom config path
    from onellm.providers import get_provider
    
    azure_provider = get_provider("azure", azure_config_path="./azure.json")
    
    # Use the provider directly
    response = await azure_provider.create_chat_completion(
        messages=[
            {"role": "user", "content": "What are the benefits of Azure OpenAI?"}
        ],
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=150
    )
    
    print(f"Response: {response.choices[0].message.content}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
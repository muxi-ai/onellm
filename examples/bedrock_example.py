#!/usr/bin/env python3
"""
AWS Bedrock Provider Example

This example demonstrates how to use the AWS Bedrock provider with OneLLM.
It shows chat completions, streaming, and embeddings with various models.

Requirements:
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to AWS Bedrock models (request access in AWS console if needed)
- boto3 package installed: pip install boto3
"""

import asyncio
import os
from onellm import Client

# Initialize the client
client = Client()


async def basic_chat_example():
    """Basic chat completion example with Claude 3.5 Sonnet."""
    print("\n=== Basic Chat Completion Example ===")
    
    try:
        response = await client.chat.completions.create(
            model="bedrock/claude-3-5-sonnet",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the main benefits of using AWS Bedrock?"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message['content']}")
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"Error: {e}")


async def streaming_example():
    """Streaming chat completion example."""
    print("\n=== Streaming Example ===")
    
    try:
        stream = await client.chat.completions.create(
            model="bedrock/claude-3-haiku",  # Using faster model for streaming
            messages=[
                {"role": "user", "content": "Write a haiku about cloud computing."}
            ],
            max_tokens=100,
            stream=True
        )
        
        print("Streaming response: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"Error: {e}")


async def multi_modal_example():
    """Multi-modal example with image input (if you have an image)."""
    print("\n=== Multi-Modal Example ===")
    
    # Note: This requires a base64-encoded image
    # For demonstration, we'll just show the structure
    print("Multi-modal support available for Claude 3 and Nova models")
    print("Example message structure:")
    print("""
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
    """)


async def embedding_example():
    """Embedding example with Titan Embeddings."""
    print("\n=== Embedding Example ===")
    
    try:
        response = await client.embeddings.create(
            model="bedrock/titan-embed-text-v2",
            input="The quick brown fox jumps over the lazy dog."
        )
        
        print(f"Embedding dimension: {len(response.data[0].embedding)}")
        print(f"First 5 values: {response.data[0].embedding[:5]}")
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"Error: {e}")


async def different_models_example():
    """Example using different model providers through Bedrock."""
    print("\n=== Different Models Example ===")
    
    models = [
        ("bedrock/claude-3-haiku", "Anthropic Claude 3 Haiku"),
        ("bedrock/llama3-2-3b", "Meta Llama 3.2 3B"),
        ("bedrock/mistral-7b", "Mistral 7B"),
        ("bedrock/nova-micro", "Amazon Nova Micro"),
    ]
    
    for model_id, model_name in models:
        print(f"\nTrying {model_name}...")
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                max_tokens=50
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")


async def custom_region_example():
    """Example using a specific AWS region."""
    print("\n=== Custom Region Example ===")
    
    # Create a client with custom region
    regional_client = Client()
    
    # You can override the region when creating the provider
    try:
        response = await regional_client.chat.completions.create(
            model="bedrock/claude-3-haiku",
            messages=[{"role": "user", "content": "What AWS region are you running in?"}],
            max_tokens=100,
            # Provider-specific config can be passed
            provider_config={"region": "eu-west-1"}  # Example of overriding region
        )
        print(f"Response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Note: Custom region configuration would be passed during provider initialization")


async def main():
    """Run all examples."""
    print("AWS Bedrock Provider Examples")
    print("=============================")
    
    # Check if AWS credentials are available
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or 
            os.environ.get("AWS_PROFILE") or
            os.path.exists(os.path.expanduser("~/.aws/credentials"))):
        print("\nWarning: AWS credentials not detected!")
        print("Please configure AWS credentials using one of these methods:")
        print("1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("2. Run 'aws configure' to set up AWS CLI")
        print("3. Use IAM role (if running on AWS)")
        return
    
    await basic_chat_example()
    await streaming_example()
    await multi_modal_example()
    await embedding_example()
    await different_models_example()
    await custom_region_example()


if __name__ == "__main__":
    asyncio.run(main())
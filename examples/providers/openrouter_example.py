#!/usr/bin/env python3
"""
OpenRouter Provider Example

This example demonstrates how to use the OpenRouter provider with OneLLM.
OpenRouter provides access to 100+ models through a unified API.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects OpenRouter from the model name)
# Uses OPENROUTER_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with OpenRouter."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is OpenRouter and why is it useful?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming with a free model."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
        messages=[
            {"role": "user", "content": "Write a haiku about cloud computing."}
        ],
        stream=True,
        max_tokens=100
    )
    
    print("Streaming haiku: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def model_variety_example():
    """Demonstrate access to various models through OpenRouter."""
    print("\n=== Model Variety Example ===")
    
    # Different models available through OpenRouter
    models = [
        ("openrouter/meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B (Free)"),
        ("openrouter/google/gemma-7b-it:free", "Gemma 7B (Free)"),
        ("openrouter/mistralai/mistral-7b-instruct:free", "Mistral 7B (Free)"),
        ("openrouter/openchat/openchat-7b:free", "OpenChat 7B (Free)")
    ]
    
    prompt = "What is machine learning in one sentence?"
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def premium_model_example():
    """Example with premium models (requires credits)."""
    print("\n=== Premium Model Example ===")
    print("Note: This example uses premium models that require credits.")
    
    # Example of premium model usage
    try:
        response = client.chat.completions.create(
            model="openrouter/anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "user", "content": "What are the advantages of using OpenRouter for accessing multiple LLMs?"}
            ],
            max_tokens=150
        )
        print(f"Claude's response: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Premium model not available or no credits: {str(e)[:150]}")
        print("\nTrying with a free alternative...")
        
        # Fallback to free model
        response = client.chat.completions.create(
            model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
            messages=[
                {"role": "user", "content": "What are the advantages of using OpenRouter for accessing multiple LLMs?"}
            ],
            max_tokens=150
        )
        print(f"Response: {response.choices[0].message['content']}")


def json_mode_example():
    """JSON mode with OpenRouter."""
    print("\n=== JSON Mode Example ===")
    
    response = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "List 3 benefits of using OpenRouter in JSON format."
            }
        ],
        max_tokens=200
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation through OpenRouter."""
    print("\n=== Code Generation Example ===")
    
    response = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to calculate the factorial of a number using recursion."
            }
        ],
        temperature=0.2,
        max_tokens=200
    )
    
    print("Generated code:")
    print(response.choices[0].message['content'])


def model_routing_example():
    """Demonstrate OpenRouter's automatic model routing."""
    print("\n=== Model Routing Example ===")
    print("OpenRouter can automatically route to the best available model.")
    
    # Using auto routing (if supported)
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",  # Auto routing
            messages=[
                {"role": "user", "content": "Explain quantum entanglement simply."}
            ],
            max_tokens=150
        )
        print(f"Response (auto-routed): {response.choices[0].message['content']}")
    except:
        # Fallback to specific model
        response = client.chat.completions.create(
            model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
            messages=[
                {"role": "user", "content": "Explain quantum entanglement simply."}
            ],
            max_tokens=150
        )
        print(f"Response: {response.choices[0].message['content']}")


def multi_language_example():
    """Multi-language support through OpenRouter."""
    print("\n=== Multi-Language Example ===")
    
    languages = [
        ("English", "Hello, how are you?"),
        ("Spanish", "¿Cómo estás?"),
        ("French", "Comment allez-vous?"),
        ("German", "Wie geht es dir?")
    ]
    
    for lang, text in languages:
        response = client.chat.completions.create(
            model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
            messages=[
                {"role": "user", "content": f"Translate '{text}' and respond in the same language."}
            ],
            max_tokens=50
        )
        print(f"\n{lang}: {text}")
        print(f"Response: {response.choices[0].message['content']}")


def main():
    """Run all examples."""
    print("OpenRouter Provider Examples\n")
    print("Access 100+ models through a unified API!\n")
    
    # Check if API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        print("Get your API key from: https://openrouter.ai/keys")
        print("\nNote: OpenRouter offers free models that don't require credits!")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        model_variety_example()
        premium_model_example()
        json_mode_example()
        code_generation_example()
        model_routing_example()
        multi_language_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mistral Provider Example

This example demonstrates how to use the Mistral provider with OneLLM.
Mistral is a European AI company offering powerful open-weight models.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Mistral from the model name)
# Uses MISTRAL_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Mistral."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="mistral/mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What makes Mistral AI unique in the AI landscape?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Mistral."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="mistral/mistral-small-latest",
        messages=[
            {"role": "user", "content": "Write a short story about a robot learning to paint."}
        ],
        stream=True,
        max_tokens=150
    )
    
    print("Streaming story: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def function_calling_example():
    """Function calling with Mistral."""
    print("\n=== Function Calling Example ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_exchange_rate",
                "description": "Get the exchange rate between two currencies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {
                            "type": "string",
                            "description": "The source currency code (e.g., USD)"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "The target currency code (e.g., EUR)"
                        }
                    },
                    "required": ["from_currency", "to_currency"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="mistral/mistral-small-latest",
        messages=[
            {"role": "user", "content": "How many euros would I get for 100 US dollars?"}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    if message.get("tool_calls"):
        print(f"Function to call: {message['tool_calls'][0]['function']['name']}")
        print(f"Arguments: {message['tool_calls'][0]['function']['arguments']}")
    else:
        print(f"Response: {message['content']}")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")
    
    response = client.chat.completions.create(
        model="mistral/mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Create a JSON object with information about 3 European cities."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def different_mistral_models():
    """Showcase different Mistral models."""
    print("\n=== Different Mistral Models ===")
    
    models = [
        ("mistral/mistral-tiny", "Mistral Tiny (fastest, most affordable)"),
        ("mistral/mistral-small-latest", "Mistral Small (balanced)"),
        ("mistral/mistral-medium-latest", "Mistral Medium (deprecated)"),
        ("mistral/mistral-large-latest", "Mistral Large (most capable)")
    ]
    
    prompt = "Explain the concept of 'emergence' in AI in one sentence."
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Note: {model_id} - {str(e)[:100]}")


def code_generation_example():
    """Code generation with Mistral."""
    print("\n=== Code Generation Example ===")
    
    response = client.chat.completions.create(
        model="mistral/mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to merge two sorted lists into one sorted list."
            }
        ],
        temperature=0.2,
        max_tokens=300
    )
    
    print("Generated code:")
    print(response.choices[0].message['content'])


def multilingual_example():
    """Multilingual capabilities of Mistral."""
    print("\n=== Multilingual Example ===")
    
    languages = [
        ("English", "Hello, how are you?"),
        ("French", "Bonjour, comment allez-vous?"),
        ("Spanish", "Hola, ¿cómo estás?"),
        ("German", "Hallo, wie geht es dir?")
    ]
    
    for language, greeting in languages:
        response = client.chat.completions.create(
            model="mistral/mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": f"Respond to this greeting appropriately: {greeting}"
                }
            ],
            max_tokens=50
        )
        print(f"\n{language}: {greeting}")
        print(f"Response: {response.choices[0].message['content']}")


def main():
    """Run all examples."""
    print("Mistral AI Provider Examples\n")
    
    # Check if API key is set
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Please set MISTRAL_API_KEY environment variable")
        print("Get your API key from: https://console.mistral.ai/api-keys/")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        function_calling_example()
        json_mode_example()
        different_mistral_models()
        code_generation_example()
        multilingual_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
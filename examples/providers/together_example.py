#!/usr/bin/env python3
"""
Together AI Provider Example

This example demonstrates how to use the Together AI provider with OneLLM.
Together AI provides fast inference for open-source models.
"""

import os
import time
from onellm import OpenAI

# Initialize the client (it automatically detects Together from the model name)
# Uses TOGETHER_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Together AI."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What makes Together AI special for running open-source models?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming with Together AI's fast inference."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "Write a short poem about open-source AI."}
        ],
        stream=True,
        max_tokens=150
    )
    
    print("Streaming poem: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def performance_comparison():
    """Compare performance of different models."""
    print("\n=== Performance Comparison ===")
    
    models = [
        ("together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "Llama 3.1 8B Turbo"),
        ("together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Llama 3.1 70B Turbo"),
        ("together/mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B"),
        ("together/togethercomputer/Llama-3-8b-chat-hf-int4", "Llama 3 8B INT4 (quantized)")
    ]
    
    prompt = "What is the capital of France?"
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            
            end_time = time.time()
            print(f"Response: {response.choices[0].message['content']}")
            print(f"Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")
    
    response = client.chat.completions.create(
        model="together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Create a JSON object with information about 3 popular open-source LLMs."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def function_calling_example():
    """Function calling with Together AI."""
    print("\n=== Function Calling Example ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_model_info",
                "description": "Get information about an AI model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the AI model"
                        },
                        "info_type": {
                            "type": "string",
                            "enum": ["parameters", "context_length", "training_data"],
                            "description": "Type of information to retrieve"
                        }
                    },
                    "required": ["model_name", "info_type"]
                }
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "How many parameters does Llama 3 70B have?"}
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
            
    except Exception as e:
        print(f"Function calling might not be supported: {str(e)[:150]}")


def code_generation_example():
    """Code generation with specialized models."""
    print("\n=== Code Generation Example ===")
    
    # Try specialized code models if available
    code_models = [
        "together/codellama/CodeLlama-70b-Instruct-hf",
        "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    ]
    
    for model in code_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes."
                    }
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            print(f"\nGenerated code using {model.split('/')[-1]}:")
            print(response.choices[0].message['content'])
            break
            
        except Exception as e:
            if model == code_models[-1]:
                print(f"Code generation error: {str(e)[:100]}")


def long_context_example():
    """Example with models supporting long context."""
    print("\n=== Long Context Example ===")
    
    # Create a longer context
    context = """
    The history of computing began with mechanical devices like the abacus and evolved through
    various stages including mechanical calculators, vacuum tubes, transistors, and integrated
    circuits. Each advancement brought exponential improvements in speed and capability.
    """ * 5  # Repeat to make longer
    
    response = client.chat.completions.create(
        model="together/mistralai/Mixtral-8x7B-Instruct-v0.1",  # 32K context
        messages=[
            {
                "role": "user",
                "content": f"Based on this text about computing history, what were the major technological transitions mentioned?\n\n{context}"
            }
        ],
        max_tokens=200
    )
    
    print(f"Analysis: {response.choices[0].message['content']}")


def quantized_model_example():
    """Example using quantized models for efficiency."""
    print("\n=== Quantized Model Example ===")
    
    models = [
        ("together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "Full precision"),
        ("together/togethercomputer/Llama-3-8b-chat-hf-int4", "INT4 quantized"),
        ("together/togethercomputer/Llama-3-8b-chat-hf-int8", "INT8 quantized")
    ]
    
    prompt = "Explain the trade-offs between model size and accuracy."
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            end_time = time.time()
            print(f"Response: {response.choices[0].message['content'][:150]}...")
            print(f"Inference time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"Model not available: {str(e)[:100]}")


def main():
    """Run all examples."""
    print("Together AI Provider Examples\n")
    print("Fast inference for open-source models!\n")
    
    # Check if API key is set
    if not os.environ.get("TOGETHER_API_KEY"):
        print("Please set TOGETHER_API_KEY environment variable")
        print("Get your API key from: https://api.together.xyz/")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        performance_comparison()
        json_mode_example()
        function_calling_example()
        code_generation_example()
        long_context_example()
        quantized_model_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
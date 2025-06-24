#!/usr/bin/env python3
"""
Anyscale Provider Example

This example demonstrates how to use the Anyscale provider with OneLLM.
Anyscale provides scalable inference for open-source models with Ray.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Anyscale from the model name)
# Uses ANYSCALE_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Anyscale."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="anyscale/llama3-8b",  # Using alias
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Anyscale and how does it relate to Ray?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Anyscale."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="anyscale/mistral-7b",
        messages=[
            {"role": "user", "content": "Explain distributed computing in simple terms."}
        ],
        stream=True,
        max_tokens=150
    )
    
    print("Streaming explanation: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def different_model_sizes():
    """Compare different model sizes available on Anyscale."""
    print("\n=== Different Model Sizes ===")
    
    models = [
        ("anyscale/llama3-8b", "Llama 3 8B (fastest)"),
        ("anyscale/llama3-70b", "Llama 3 70B (more capable)"),
        ("anyscale/mixtral-8x7b", "Mixtral 8x7B (MoE architecture)"),
        ("anyscale/codellama-34b", "CodeLlama 34B (for code)")
    ]
    
    prompt = "What is Ray framework in one sentence?"
    
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
            print(f"Error with {model_id}: {str(e)[:100]}")


def json_mode_with_schema():
    """Anyscale's extended JSON mode with schema specification."""
    print("\n=== JSON Mode with Schema ===")
    
    response = client.chat.completions.create(
        model="anyscale/llama3-8b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Create information about a distributed computing system."
            }
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "components": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "scalability": {"type": "string"},
                    "use_cases": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "components", "scalability"]
            }
        },
        max_tokens=300
    )
    
    print(f"Structured JSON Response: {response.choices[0].message['content']}")


def function_calling_example():
    """Function calling with Anyscale (single calls only)."""
    print("\n=== Function Calling Example ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scale_cluster",
                "description": "Scale a Ray cluster",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cluster_name": {
                            "type": "string",
                            "description": "Name of the cluster"
                        },
                        "num_nodes": {
                            "type": "integer",
                            "description": "Number of nodes to scale to"
                        },
                        "node_type": {
                            "type": "string",
                            "enum": ["cpu", "gpu"],
                            "description": "Type of nodes"
                        }
                    },
                    "required": ["cluster_name", "num_nodes"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="anyscale/llama3-70b",
        messages=[
            {"role": "user", "content": "I need to scale my ML training cluster to 10 GPU nodes."}
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


def code_generation_example():
    """Code generation with CodeLlama models."""
    print("\n=== Code Generation Example ===")
    
    code_models = [
        ("anyscale/codellama-7b", "CodeLlama 7B"),
        ("anyscale/codellama-34b", "CodeLlama 34B")
    ]
    
    for model_id, description in code_models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": "Write a Python function for parallel processing using Ray."
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            print("Generated code:")
            print(response.choices[0].message['content'])
            break
            
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def rate_limit_aware_example():
    """Example showing Anyscale's unique rate limiting (30 concurrent requests)."""
    print("\n=== Rate Limit Awareness Example ===")
    
    print("Anyscale has a unique rate limit: 30 concurrent requests maximum")
    print("This translates to ~227 queries per minute max\n")
    
    # Single request example
    response = client.chat.completions.create(
        model="anyscale/llama3-8b",
        messages=[
            {"role": "user", "content": "How should I handle Anyscale's concurrent request limit?"}
        ],
        max_tokens=200
    )
    
    print(f"Advice: {response.choices[0].message['content']}")


def open_source_models_showcase():
    """Showcase various open-source models available."""
    print("\n=== Open Source Models Showcase ===")
    
    model_examples = [
        {
            "model": "anyscale/llama2-70b",
            "prompt": "What is the importance of open-source AI models?"
        },
        {
            "model": "anyscale/gemma-7b",
            "prompt": "Explain machine learning optimization techniques."
        },
        {
            "model": "anyscale/mixtral-8x22b",
            "prompt": "What are mixture of experts models?"
        }
    ]
    
    for example in model_examples:
        try:
            print(f"\nModel: {example['model']}")
            print(f"Prompt: {example['prompt']}")
            
            response = client.chat.completions.create(
                model=example['model'],
                messages=[{"role": "user", "content": example['prompt']}],
                max_tokens=100
            )
            
            print(f"Response: {response.choices[0].message['content'][:150]}...")
            
        except Exception as e:
            print(f"Error: {str(e)[:100]}")


def cost_estimation_example():
    """Example showing Anyscale's simple pricing model."""
    print("\n=== Cost Estimation Example ===")
    
    print("Anyscale Pricing: $1.00 per million tokens (both input and output)")
    print("Let's calculate some costs:\n")
    
    examples = [
        {"tokens": 1000, "description": "Small request"},
        {"tokens": 10000, "description": "Medium conversation"},
        {"tokens": 100000, "description": "Long document processing"},
        {"tokens": 1000000, "description": "Extensive usage"}
    ]
    
    for example in examples:
        cost = example["tokens"] / 1_000_000 * 1.00
        print(f"{example['description']} ({example['tokens']:,} tokens): ${cost:.4f}")


def main():
    """Run all examples."""
    print("Anyscale Provider Examples\n")
    print("Scalable open-source model hosting with Ray!\n")
    
    # Check if API key is set
    if not os.environ.get("ANYSCALE_API_KEY"):
        print("Please set ANYSCALE_API_KEY environment variable")
        print("Get your API key from: https://www.anyscale.com/")
        print("Note: API keys start with 'esecret_'")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        different_model_sizes()
        json_mode_with_schema()
        function_calling_example()
        code_generation_example()
        rate_limit_aware_example()
        open_source_models_showcase()
        cost_estimation_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
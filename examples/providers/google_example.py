#!/usr/bin/env python3
"""
Google AI Studio Provider Example

This example demonstrates how to use the Google AI Studio provider with OneLLM,
featuring Gemini models with multimodal capabilities.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Google from the model name)
# Uses GOOGLE_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Gemini."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the key features of Google's Gemini models?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Gemini."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {"role": "user", "content": "Write a creative description of a sunset over the ocean."}
        ],
        stream=True,
        max_tokens=150
    )
    
    print("Streaming description: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def multimodal_example():
    """Multimodal capabilities with images."""
    print("\n=== Multimodal Example (Image Understanding) ===")
    
    # Example with image URL
    response = client.chat.completions.create(
        model="google/gemini-1.5-pro",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    print(f"Gemini's analysis: {response.choices[0].message['content']}")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")
    
    response = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Create a JSON object with information about 3 Google products and their main features."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def different_gemini_models():
    """Compare different Gemini models."""
    print("\n=== Different Gemini Models ===")
    
    models = [
        ("google/gemini-1.5-flash", "Gemini 1.5 Flash (fast, efficient)"),
        ("google/gemini-1.5-pro", "Gemini 1.5 Pro (most capable)"),
        ("google/gemini-pro", "Gemini Pro (previous generation)")
    ]
    
    prompt = "Explain the concept of 'attention mechanism' in transformers in simple terms."
    
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
            print(f"Note: {model_id} might not be available: {str(e)[:100]}")


def long_context_example():
    """Demonstrate Gemini's long context window."""
    print("\n=== Long Context Example ===")
    
    # Create a long context
    long_text = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories, 
    and rumors of artificial beings endowed with intelligence or consciousness by master 
    craftsmen. The seeds of modern AI were planted by classical philosophers who attempted 
    to describe the process of human thinking as the mechanical manipulation of symbols.
    
    This work culminated in the invention of the programmable digital computer in the 1940s, 
    a machine based on the abstract essence of mathematical reasoning. This device and the 
    ideas behind it inspired a handful of scientists to begin seriously discussing the 
    possibility of building an electronic brain.
    
    The field of AI research was founded at a workshop held on the campus of Dartmouth 
    College, USA during the summer of 1956. Those who attended would become the leaders 
    of AI research for decades.
    """
    
    response = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {"role": "user", "content": f"Summarize this text in 2 sentences: {long_text}"}
        ],
        max_tokens=100
    )
    
    print(f"Summary: {response.choices[0].message['content']}")


def safety_settings_example():
    """Example with safety settings (Gemini specific)."""
    print("\n=== Safety Settings Example ===")
    
    response = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {
                "role": "user",
                "content": "Write a children's story about a brave little robot."
            }
        ],
        temperature=0.9,
        max_tokens=200
    )
    
    print(f"Safe content generated: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation with Gemini."""
    print("\n=== Code Generation Example ===")
    
    response = client.chat.completions.create(
        model="google/gemini-1.5-pro",
        messages=[
            {
                "role": "user",
                "content": "Write a Python class for a simple task queue with add, remove, and process methods."
            }
        ],
        temperature=0.2,
        max_tokens=400
    )
    
    print("Generated code:")
    print(response.choices[0].message['content'])


def main():
    """Run all examples."""
    print("Google AI Studio (Gemini) Provider Examples\n")
    
    # Check if API key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        # multimodal_example()  # Uncomment if you want to test image understanding
        json_mode_example()
        different_gemini_models()
        long_context_example()
        safety_settings_example()
        code_generation_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
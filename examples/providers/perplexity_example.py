#!/usr/bin/env python3
"""
Perplexity AI Provider Example

This example demonstrates how to use the Perplexity provider with OneLLM.
Perplexity offers search-augmented AI models with real-time web access.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Perplexity from the model name)
# Uses PERPLEXITY_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Perplexity."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-small-128k-online",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to current information."},
            {"role": "user", "content": "What are the latest developments in AI as of 2024?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def online_search_example():
    """Demonstrate Perplexity's real-time search capabilities."""
    print("\n=== Online Search Example ===")
    
    # Ask about current events
    response = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-large-128k-online",
        messages=[
            {"role": "user", "content": "What is happening in the tech industry this week?"}
        ],
        max_tokens=300
    )
    
    print(f"Current information: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming with search-augmented responses."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-small-128k-online",
        messages=[
            {"role": "user", "content": "What are the current stock prices of major tech companies?"}
        ],
        stream=True,
        max_tokens=200
    )
    
    print("Streaming response: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def different_perplexity_models():
    """Compare different Perplexity models."""
    print("\n=== Different Perplexity Models ===")
    
    models = [
        ("perplexity/llama-3.1-sonar-small-128k-online", "Sonar Small (fast, online)"),
        ("perplexity/llama-3.1-sonar-large-128k-online", "Sonar Large (accurate, online)"),
        ("perplexity/llama-3.1-sonar-small-128k-chat", "Sonar Small Chat (offline)"),
        ("perplexity/llama-3.1-sonar-large-128k-chat", "Sonar Large Chat (offline)")
    ]
    
    prompt = "What is Perplexity AI?"
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            print(f"Response: {response.choices[0].message['content'][:200]}...")
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def fact_checking_example():
    """Use Perplexity for fact-checking with sources."""
    print("\n=== Fact Checking Example ===")
    
    claims = [
        "The Great Wall of China is visible from space",
        "Lightning never strikes the same place twice",
        "Humans only use 10% of their brain"
    ]
    
    for claim in claims:
        response = client.chat.completions.create(
            model="perplexity/llama-3.1-sonar-large-128k-online",
            messages=[
                {"role": "user", "content": f"Is this claim true or false? Explain with sources: '{claim}'"}
            ],
            max_tokens=150
        )
        
        print(f"\nClaim: '{claim}'")
        print(f"Fact check: {response.choices[0].message['content']}")


def research_assistant_example():
    """Use Perplexity as a research assistant."""
    print("\n=== Research Assistant Example ===")
    
    response = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-large-128k-online",
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Provide detailed, sourced information."
            },
            {
                "role": "user",
                "content": "What are the latest breakthroughs in quantum computing? Include recent developments from 2024."
            }
        ],
        max_tokens=400
    )
    
    print("Research findings:")
    print(response.choices[0].message['content'])


def comparison_example():
    """Compare online vs offline models."""
    print("\n=== Online vs Offline Model Comparison ===")
    
    question = "Who won the most recent Nobel Prize in Physics?"
    
    # Try with online model
    print("Online model (with web access):")
    try:
        response_online = client.chat.completions.create(
            model="perplexity/llama-3.1-sonar-small-128k-online",
            messages=[{"role": "user", "content": question}],
            max_tokens=150
        )
        print(f"Response: {response_online.choices[0].message['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try with offline model
    print("\nOffline model (no web access):")
    try:
        response_offline = client.chat.completions.create(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
            messages=[{"role": "user", "content": question}],
            max_tokens=150
        )
        print(f"Response: {response_offline.choices[0].message['content']}")
    except Exception as e:
        print(f"Error: {e}")


def code_search_example():
    """Use Perplexity to search for code examples."""
    print("\n=== Code Search Example ===")
    
    response = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-large-128k-online",
        messages=[
            {
                "role": "user",
                "content": "Find me the latest best practices for implementing authentication in Next.js 14 applications."
            }
        ],
        max_tokens=300
    )
    
    print("Code search results:")
    print(response.choices[0].message['content'])


def multi_turn_research():
    """Multi-turn conversation for deep research."""
    print("\n=== Multi-turn Research Example ===")
    
    messages = [
        {"role": "user", "content": "What are the current leading companies in autonomous vehicles?"},
        {"role": "assistant", "content": "Based on recent data, the leading companies in autonomous vehicles include Waymo (Google), Cruise (GM), Tesla, Zoox (Amazon), and Aurora. These companies are at various stages of testing and deployment."},
        {"role": "user", "content": "What recent milestones have Waymo achieved in 2024?"}
    ]
    
    response = client.chat.completions.create(
        model="perplexity/llama-3.1-sonar-large-128k-online",
        messages=messages,
        max_tokens=300
    )
    
    print("Latest Waymo developments:")
    print(response.choices[0].message['content'])


def main():
    """Run all examples."""
    print("Perplexity AI Provider Examples\n")
    print("Search-augmented AI with real-time web access!\n")
    
    # Check if API key is set
    if not os.environ.get("PERPLEXITY_API_KEY"):
        print("Please set PERPLEXITY_API_KEY environment variable")
        print("Get your API key from: https://www.perplexity.ai/settings/api")
        return
    
    try:
        basic_chat_completion()
        online_search_example()
        streaming_example()
        different_perplexity_models()
        fact_checking_example()
        research_assistant_example()
        comparison_example()
        code_search_example()
        multi_turn_research()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
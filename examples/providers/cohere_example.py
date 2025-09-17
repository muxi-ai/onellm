#!/usr/bin/env python3
"""
Cohere Provider Example

This example demonstrates how to use the Cohere provider with OneLLM.
Cohere specializes in enterprise NLP with RAG capabilities.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Cohere from the model name)
# Uses COHERE_API_KEY environment variable by default
client = OpenAI()

def basic_chat_completion():
    """Basic chat completion with Cohere."""
    print("=== Basic Chat Completion ===")

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What makes Cohere's models special for enterprise use?"}
        ],
        temperature=0.7,
        max_tokens=200
    )

    print(f"Response: {response.choices[0].message['content']}")

def streaming_example():
    """Streaming response with Cohere."""
    print("\n=== Streaming Example ===")

    stream = client.chat.completions.create(
        model="cohere/command-r",
        messages=[
            {"role": "user", "content": "Explain the concept of retrieval-augmented generation (RAG)."}
        ],
        stream=True,
        max_tokens=150
    )

    print("Streaming explanation: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")

def multi_turn_conversation():
    """Multi-turn conversation with context."""
    print("\n=== Multi-turn Conversation ===")

    messages = [
        {"role": "user", "content": "I'm building a customer support chatbot."},
        {"role": "assistant", "content": "That's great! Customer support chatbots can significantly improve response times and customer satisfaction. What specific features are you planning to implement?"},
        {"role": "user", "content": "I want it to understand customer sentiment and route to human agents when needed."}
    ]

    response = client.chat.completions.create(
        model="cohere/command-r",
        messages=messages,
        max_tokens=200
    )

    print(f"Cohere's response: {response.choices[0].message['content']}")

def structured_generation_example():
    """Structured generation for data extraction."""
    print("\n=== Structured Generation Example ===")

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction assistant. Extract structured information and format it clearly."
            },
            {
                "role": "user",
                "content": """Extract the key information from this text:

                John Smith, aged 32, is a software engineer at TechCorp Inc.
                He has 8 years of experience in Python and JavaScript.
                His email is john.smith@techcorp.com and he's based in San Francisco.
                """
            }
        ],
        max_tokens=200
    )

    print(f"Extracted information:\n{response.choices[0].message['content']}")

def different_cohere_models():
    """Compare different Cohere models."""
    print("\n=== Different Cohere Models ===")

    models = [
        ("cohere/command-r", "Command R (efficient, good for chat)"),
        ("cohere/command-r-plus", "Command R+ (most capable)"),
        ("cohere/command", "Command (legacy)"),
        ("cohere/command-light", "Command Light (fastest)")
    ]

    prompt = "What is natural language processing?"

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
            print(f"Note: {model_id} might not be available: {str(e)[:100]}")

def rag_style_example():
    """Example showing RAG-style question answering."""
    print("\n=== RAG-Style Question Answering ===")

    # Simulate a document/context that would come from a retrieval system
    context = """
    Company Policy on Remote Work:
    - Employees may work remotely up to 3 days per week
    - Remote work requires manager approval
    - Core hours are 10 AM to 3 PM in the employee's time zone
    - All remote workers must be available for video calls during core hours
    - VPN connection is required when accessing company resources
    """

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=[
            {
                "role": "system",
                "content": "Answer questions based on the provided context. If the answer isn't in the context, say so."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: How many days can I work from home?"
            }
        ],
        max_tokens=100
    )

    print(f"Answer: {response.choices[0].message['content']}")

def summarization_example():
    """Text summarization with Cohere."""
    print("\n=== Summarization Example ===")

    long_text = """
    Artificial intelligence has transformed from a futuristic concept to a practical reality
    that impacts our daily lives. From voice assistants that help us manage our schedules
    to recommendation systems that suggest what we should watch or buy, AI is everywhere.

    In healthcare, AI assists doctors in diagnosing diseases earlier and more accurately.
    In transportation, self-driving cars are becoming a reality. In finance, AI helps detect
    fraud and make investment decisions.

    However, with these advances come important considerations about privacy, bias, and the
    future of work. As AI continues to evolve, society must grapple with these challenges
    while harnessing the technology's benefits.
    """

    response = client.chat.completions.create(
        model="cohere/command-r",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this text in 2-3 sentences:\n\n{long_text}"
            }
        ],
        max_tokens=100,
        temperature=0.3
    )

    print(f"Summary: {response.choices[0].message['content']}")

def code_explanation_example():
    """Code explanation with Cohere."""
    print("\n=== Code Explanation Example ===")

    code = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """

    response = client.chat.completions.create(
        model="cohere/command-r-plus",
        messages=[
            {
                "role": "user",
                "content": f"Explain this Python code in simple terms:\n\n```python\n{code}\n```"
            }
        ],
        max_tokens=200
    )

    print("Explanation:")
    print(response.choices[0].message['content'])

def main():
    """Run all examples."""
    print("Cohere Provider Examples\n")

    # Check if API key is set
    if not os.environ.get("COHERE_API_KEY"):
        print("Please set COHERE_API_KEY environment variable")
        print("Get your API key from: https://dashboard.cohere.com/api-keys")
        return

    try:
        basic_chat_completion()
        streaming_example()
        multi_turn_conversation()
        structured_generation_example()
        different_cohere_models()
        rag_style_example()
        summarization_example()
        code_explanation_example()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

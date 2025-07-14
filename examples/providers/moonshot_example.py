#!/usr/bin/env python3
"""
Moonshot Provider Example

This example demonstrates how to use the Moonshot provider with OneLLM.
Moonshot is a Chinese AI company offering Kimi models with strong long-context capabilities.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Moonshot from the model name)
# Uses KIMI_API_KEY environment variable by default
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Moonshot."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "You are Kimi, a helpful multilingual assistant."},
            {"role": "user", "content": "What makes Moonshot's Kimi models unique in the AI landscape?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Moonshot."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=[
            {"role": "user", "content": "Write a short story about AI helping humanity, in exactly 3 sentences."}
        ],
        stream=True,
        max_tokens=150
    )
    
    print("Streaming story: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def long_context_example():
    """Demonstrate Moonshot's long context capabilities."""
    print("\n=== Long Context Example ===")
    
    # Create a longer document to test context handling
    long_document = """
    This is a comprehensive report on artificial intelligence developments in 2024.
    The field has seen significant advances in large language models, with improvements in reasoning,
    code generation, and multilingual capabilities. Companies like OpenAI, Anthropic, Google,
    and Chinese companies like Moonshot AI have made substantial contributions.
    
    Key developments include:
    1. Enhanced reasoning capabilities in models like GPT-4 and Claude
    2. Improved code generation and debugging assistance
    3. Better multilingual support, especially for Chinese and other Asian languages
    4. Long context windows allowing for processing of entire documents
    5. Multi-modal capabilities combining text, images, and audio
    
    The Chinese AI landscape has been particularly active, with companies like Moonshot AI
    developing models with exceptional long-context capabilities. Their Kimi models can handle
    contexts of up to 200,000 tokens, making them suitable for document analysis and
    long-form content generation.
    
    Looking ahead, the focus is on making AI more accessible, efficient, and capable of
    handling complex real-world tasks across various domains.
    """ * 3  # Repeat to make it longer
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-32k",  # Using 32k context model
        messages=[
            {"role": "user", "content": f"Please summarize this document in 2-3 sentences: {long_document}"}
        ],
        max_tokens=100
    )
    
    print("Document summary:")
    print(response.choices[0].message['content'])


def multilingual_example():
    """Demonstrate Moonshot's multilingual capabilities."""
    print("\n=== Multilingual Example ===")
    
    languages = [
        ("English", "How does artificial intelligence work?"),
        ("Chinese", "人工智能是如何工作的？"),
        ("Spanish", "¿Cómo funciona la inteligencia artificial?"),
        ("Japanese", "人工知能はどのように機能しますか？"),
        ("Korean", "인공지능은 어떻게 작동하나요?")
    ]
    
    for lang, question in languages:
        response = client.chat.completions.create(
            model="moonshot/moonshot-v1-8k",
            messages=[
                {"role": "user", "content": f"{question} (Please answer in {lang})"}
            ],
            max_tokens=100,
            temperature=0.5
        )
        
        print(f"\n{lang}: {question}")
        print(f"Response: {response.choices[0].message['content']}")


def code_generation_example():
    """Code generation with Moonshot."""
    print("\n=== Code Generation Example ===")
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to convert temperature between Celsius, Fahrenheit, and Kelvin."
            }
        ],
        temperature=0.2,
        max_tokens=400
    )
    
    print("Generated code:")
    print(response.choices[0].message['content'])


def chinese_specific_example():
    """Example focusing on Chinese language tasks."""
    print("\n=== Chinese Language Tasks ===")
    
    tasks = [
        {
            "task": "Translation",
            "prompt": "Translate to Chinese: 'Artificial intelligence is transforming the world.'"
        },
        {
            "task": "Poetry",
            "prompt": "写一首关于春天的五言绝句。(Write a 5-character quatrain about spring)"
        },
        {
            "task": "Explanation",
            "prompt": "用简单的中文解释什么是机器学习。(Explain machine learning in simple Chinese)"
        }
    ]
    
    for task_info in tasks:
        response = client.chat.completions.create(
            model="moonshot/moonshot-v1-8k",
            messages=[
                {"role": "user", "content": task_info["prompt"]}
            ],
            max_tokens=200
        )
        
        print(f"\n{task_info['task']}:")
        print(f"Prompt: {task_info['prompt']}")
        print(f"Response: {response.choices[0].message['content']}")


def mathematical_reasoning_example():
    """Mathematical problem solving with Moonshot."""
    print("\n=== Mathematical Reasoning ===")
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=[
            {
                "role": "user",
                "content": """Solve this problem step by step:
                
                A train travels from City A to City B at 80 km/h and returns at 120 km/h.
                What is the average speed for the entire round trip?"""
            }
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    print("Solution:")
    print(response.choices[0].message['content'])


def json_output_example():
    """JSON structured output with Moonshot."""
    print("\n=== JSON Output Example ===")
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs valid JSON."
            },
            {
                "role": "user",
                "content": "Create a JSON object with information about 3 major Chinese tech companies."
            }
        ],
        max_tokens=300
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def different_moonshot_models():
    """Try different Moonshot model variants."""
    print("\n=== Different Moonshot Models ===")
    
    models = [
        ("moonshot/moonshot-v1-8k", "Moonshot 8K (8,000 tokens context)"),
        ("moonshot/moonshot-v1-32k", "Moonshot 32K (32,000 tokens context)"),
        ("moonshot/moonshot-v1-128k", "Moonshot 128K (128,000 tokens context)"),
        ("moonshot/kimi-k2-0711-preview", "Kimi K2 (latest preview model)")
    ]
    
    prompt = "What is the difference between machine learning and deep learning?"
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            print(f"Response: {response.choices[0].message['content'][:150]}...")
        except Exception as e:
            print(f"Note: {model_id} might not be available: {str(e)[:100]}")


def conversation_example():
    """Multi-turn conversation with context."""
    print("\n=== Multi-turn Conversation ===")
    
    messages = [
        {"role": "user", "content": "I want to learn about Chinese culture."},
        {"role": "assistant", "content": "That's wonderful! Chinese culture is rich and diverse, spanning over 5,000 years. What aspect interests you most - history, philosophy, art, cuisine, or modern culture?"},
        {"role": "user", "content": "Tell me about Chinese philosophy, especially Confucianism."}
    ]
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-8k",
        messages=messages,
        max_tokens=300
    )
    
    print("Conversation about Chinese philosophy:")
    print(response.choices[0].message['content'])


def document_analysis_example():
    """Document analysis with long context."""
    print("\n=== Document Analysis Example ===")
    
    # Simulate a long document
    document = """
    Executive Summary: Q3 2024 AI Market Report
    
    The artificial intelligence market has experienced unprecedented growth in Q3 2024,
    with significant developments in large language models, computer vision, and
    autonomous systems. Key findings include:
    
    1. Market Growth: The global AI market reached $150 billion, representing 35% YoY growth
    2. LLM Adoption: Enterprise adoption of LLMs increased by 200% compared to Q2 2024
    3. Regional Analysis:
       - North America: $60B market share (40%)
       - Asia-Pacific: $45B market share (30%)
       - Europe: $30B market share (20%)
       - Rest of World: $15B market share (10%)
    
    Chinese AI companies have been particularly active, with Moonshot AI, Baidu, and
    Alibaba launching new models with enhanced capabilities. The focus on long-context
    processing and multilingual support has driven significant user adoption.
    
    Technical Developments:
    - Context windows expanded to 200K+ tokens
    - Improved reasoning capabilities
    - Enhanced code generation
    - Better multilingual support
    
    Market Challenges:
    - Compute costs remain high
    - Regulatory uncertainty
    - Competition for AI talent
    - Data privacy concerns
    
    Outlook: The market is expected to continue growing at 30%+ CAGR through 2025,
    driven by enterprise adoption and new use cases in healthcare, finance, and education.
    """
    
    response = client.chat.completions.create(
        model="moonshot/moonshot-v1-32k",
        messages=[
            {
                "role": "user",
                "content": f"Analyze this market report and provide: 1) Key insights, 2) Market opportunities, 3) Risks to consider:\n\n{document}"
            }
        ],
        max_tokens=400
    )
    
    print("Document Analysis:")
    print(response.choices[0].message['content'])


def main():
    """Run all examples."""
    print("Moonshot Provider Examples\n")
    print("Kimi AI with exceptional long-context capabilities!\n")
    
    # Check if API key is set
    if not os.environ.get("KIMI_API_KEY"):
        print("Please set KIMI_API_KEY environment variable")
        print("Get your API key from: https://platform.moonshot.ai/")
        return
    
    try:
        basic_chat_completion()
        streaming_example()
        long_context_example()
        multilingual_example()
        code_generation_example()
        chinese_specific_example()
        mathematical_reasoning_example()
        json_output_example()
        different_moonshot_models()
        conversation_example()
        document_analysis_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

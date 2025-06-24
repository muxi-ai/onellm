#!/usr/bin/env python3
"""
Vertex AI Provider Example

This example demonstrates how to use the Vertex AI provider with OneLLM.
Vertex AI is Google Cloud's enterprise AI platform.
"""

import os
from onellm import OpenAI

# Initialize the client (it automatically detects Vertex AI from the model name)
# Uses service account credentials from GOOGLE_APPLICATION_CREDENTIALS
client = OpenAI()


def basic_chat_completion():
    """Basic chat completion with Vertex AI."""
    print("=== Basic Chat Completion ===")
    
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of using Vertex AI for enterprise ML?"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Response: {response.choices[0].message['content']}")


def streaming_example():
    """Streaming response with Vertex AI."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="vertexai/gemini-1.5-flash",
        messages=[
            {"role": "user", "content": "List 5 key features of Google Cloud's AI services."}
        ],
        stream=True,
        max_tokens=200
    )
    
    print("Streaming features: ")
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta["content"], end="", flush=True)
    print("\n")


def multimodal_example():
    """Multimodal capabilities with Vertex AI Gemini."""
    print("\n=== Multimodal Example ===")
    
    # Example with image URL (GCS URL recommended for Vertex AI)
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-pro",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://storage.googleapis.com/your-bucket/sample-image.jpg"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    print(f"Image analysis: {response.choices[0].message['content']}")


def json_mode_example():
    """JSON mode for structured output."""
    print("\n=== JSON Mode Example ===")
    
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-flash",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Create a JSON object describing Google Cloud's AI/ML services hierarchy."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    
    print(f"JSON Response: {response.choices[0].message['content']}")


def different_gemini_models():
    """Compare different Gemini models on Vertex AI."""
    print("\n=== Different Gemini Models ===")
    
    models = [
        ("vertexai/gemini-1.5-flash", "Gemini 1.5 Flash (fast, efficient)"),
        ("vertexai/gemini-1.5-pro", "Gemini 1.5 Pro (most capable)"),
        ("vertexai/gemini-pro", "Gemini Pro (previous generation)")
    ]
    
    prompt = "Explain the concept of federated learning in one paragraph."
    
    for model_id, description in models:
        try:
            print(f"\n{description}:")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.5
            )
            print(f"Response: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error with {model_id}: {str(e)[:100]}")


def enterprise_features_example():
    """Demonstrate enterprise features of Vertex AI."""
    print("\n=== Enterprise Features Example ===")
    
    # Example showing how Vertex AI can be used for enterprise scenarios
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-pro",
        messages=[
            {
                "role": "system",
                "content": "You are an enterprise AI assistant with knowledge of compliance and security."
            },
            {
                "role": "user",
                "content": "What are the key compliance and security features of Vertex AI for regulated industries?"
            }
        ],
        max_tokens=300
    )
    
    print("Enterprise features:")
    print(response.choices[0].message['content'])


def embedding_example():
    """Generate embeddings with Vertex AI."""
    print("\n=== Embedding Example ===")
    
    try:
        # Generate embeddings
        response = client.embeddings.create(
            model="vertexai/text-embedding-004",
            input=[
                "Vertex AI is Google Cloud's ML platform",
                "Machine learning in the cloud",
                "Enterprise AI solutions"
            ]
        )
        
        for i, embedding in enumerate(response.data):
            print(f"Embedding {i+1}: {len(embedding['embedding'])} dimensions")
            print(f"First 5 values: {embedding['embedding'][:5]}")
            
    except Exception as e:
        print(f"Embedding generation error: {str(e)[:150]}")


def code_generation_example():
    """Code generation with Gemini on Vertex AI."""
    print("\n=== Code Generation Example ===")
    
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-pro",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to interact with Google Cloud Storage using the google-cloud-storage library."
            }
        ],
        temperature=0.2,
        max_tokens=400
    )
    
    print("Generated code:")
    print(response.choices[0].message['content'])


def safety_settings_example():
    """Example with safety settings (Vertex AI specific)."""
    print("\n=== Safety Settings Example ===")
    
    response = client.chat.completions.create(
        model="vertexai/gemini-1.5-flash",
        messages=[
            {
                "role": "user",
                "content": "Write a professional email declining a job offer politely."
            }
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print("Safe professional content:")
    print(response.choices[0].message['content'])


def batch_prediction_scenario():
    """Simulate batch prediction scenario."""
    print("\n=== Batch Prediction Scenario ===")
    
    # Simulate processing multiple items
    items = [
        "Classify this sentiment: 'This product exceeded my expectations!'",
        "Classify this sentiment: 'The service was disappointing.'",
        "Classify this sentiment: 'It's okay, nothing special.'"
    ]
    
    print("Processing batch predictions...")
    for i, item in enumerate(items):
        response = client.chat.completions.create(
            model="vertexai/gemini-1.5-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment classifier. Respond with only: POSITIVE, NEGATIVE, or NEUTRAL"
                },
                {"role": "user", "content": item}
            ],
            max_tokens=10,
            temperature=0
        )
        
        print(f"Item {i+1}: {item}")
        print(f"Classification: {response.choices[0].message['content']}\n")


def main():
    """Run all examples."""
    print("Vertex AI Provider Examples\n")
    print("Enterprise Google Cloud AI Platform\n")
    
    # Check if credentials are set
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("This should point to your service account JSON file")
        print("Example: export GOOGLE_APPLICATION_CREDENTIALS='path/to/vertexai.json'")
        print("\nTo set up Vertex AI:")
        print("1. Create a Google Cloud project")
        print("2. Enable Vertex AI API")
        print("3. Create a service account with Vertex AI permissions")
        print("4. Download the service account JSON key")
        return
    
    # Note about configuration
    print("Note: Vertex AI requires proper Google Cloud setup:")
    print("- Project ID (from service account or config)")
    print("- Region/location (default: us-central1)")
    print("- Proper IAM permissions\n")
    
    try:
        basic_chat_completion()
        streaming_example()
        # multimodal_example()  # Uncomment with valid GCS image
        json_mode_example()
        different_gemini_models()
        enterprise_features_example()
        embedding_example()
        code_generation_example()
        safety_settings_example()
        batch_prediction_scenario()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your service account has Vertex AI permissions")
        print("2. Check that Vertex AI API is enabled in your project")
        print("3. Verify the project has billing enabled")


if __name__ == "__main__":
    main()
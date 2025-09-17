#!/usr/bin/env python3
"""
Test various Vertex AI model naming conventions.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Set the credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexai.json"


def test_vertex_models():
    """Test various Vertex AI model naming conventions."""
    print("🔍 Testing various Vertex AI model naming conventions...")

    from onellm import OpenAI

    # Test many different model name patterns based on Vertex AI documentation
    models = [
        # Standard Gemini models
        "vertexai/gemini-1.5-flash",
        "vertexai/gemini-1.5-flash-001",
        "vertexai/gemini-1.5-flash-002",
        "vertexai/gemini-1.5-pro",
        "vertexai/gemini-1.5-pro-001",
        "vertexai/gemini-1.5-pro-002",
        "vertexai/gemini-pro",
        "vertexai/gemini-pro-001",
        "vertexai/gemini-pro-vision",
        # Gemini 1.0 models
        "vertexai/gemini-1.0-pro",
        "vertexai/gemini-1.0-pro-001",
        "vertexai/gemini-1.0-pro-002",
        "vertexai/gemini-1.0-pro-vision",
        "vertexai/gemini-1.0-pro-vision-001",
        # Text models
        "vertexai/text-bison",
        "vertexai/text-bison@001",
        "vertexai/text-bison@002",
        # Chat models
        "vertexai/chat-bison",
        "vertexai/chat-bison@001",
        "vertexai/chat-bison@002",
        # Code models
        "vertexai/code-bison",
        "vertexai/code-bison@001",
        "vertexai/code-bison@002",
        # Embedding models
        "vertexai/textembedding-gecko",
        "vertexai/textembedding-gecko@001",
        "vertexai/textembedding-gecko@002",
        "vertexai/textembedding-gecko@003",
        # Try without version suffixes
        "vertexai/gemini",
        "vertexai/bison",
        # Try latest/stable tags
        "vertexai/gemini-1.5-flash-latest",
        "vertexai/gemini-1.5-pro-latest",
        "vertexai/gemini-pro-latest",
    ]

    client = OpenAI()

    success_count = 0
    for model in models:
        try:
            print(f"\nTesting {model}...", end=" ")
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Say hi"}], max_tokens=5
            )

            content = response.choices[0].message.get("content", "")
            print(f"✅ SUCCESS: {content}")
            print(f"  Model ID in response: {response.model}")
            success_count += 1

            if success_count == 1:
                print(f"\n🎉 FIRST WORKING MODEL FOUND: {model}")

        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                # Extract the actual path tried
                if "projects/" in error_msg:
                    path_start = error_msg.find("projects/")
                    path_end = error_msg.find("`", path_start + 1)
                    if path_end > path_start:
                        path = error_msg[path_start:path_end]
                        print(f"❌ Not found - tried: {path}")
                else:
                    print("❌ Not found")
            elif "not supported" in error_msg.lower():
                print("❌ Not supported")
            elif "invalid" in error_msg.lower():
                print("❌ Invalid model")
            else:
                print(f"❌ Error: {error_msg[:50]}...")

    print(f"\n📊 Summary: {success_count}/{len(models)} models worked")
    return success_count > 0


if __name__ == "__main__":
    result = test_vertex_models()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")

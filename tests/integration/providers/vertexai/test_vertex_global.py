#!/usr/bin/env python3
"""
Test Vertex AI with global location.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Set the credential path and location
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexai.json"
os.environ["VERTEX_AI_LOCATION"] = "global"


def test_vertex_global():
    """Test Vertex AI with global location."""
    print("🔍 Testing Vertex AI with global location...")

    from onellm import OpenAI

    models = [
        "vertexai/gemini-1.5-flash",
        "vertexai/gemini-1.5-flash-001",
        "vertexai/gemini-1.5-pro",
        "vertexai/gemini-1.5-pro-001",
        "vertexai/gemini-pro",
        "vertexai/gemini-1.0-pro",
        "vertexai/gemini-1.0-pro-001",
    ]

    client = OpenAI()

    for model in models:
        try:
            print(f"\nTesting {model}...", end=" ")
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
            )

            content = response.choices[0].message.get("content", "")
            print(f"✅ SUCCESS: {content}")
            print(f"Model: {model}")
            print("\n🎉 VERTEX AI IS NOW WORKING WITH GLOBAL LOCATION!")
            return True

        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                print("❌ Not found")
            else:
                print(f"❌ {error_msg[:100]}...")

    return False


if __name__ == "__main__":
    # First test with global location
    result = test_vertex_global()

    if not result:
        print("\n🔄 Testing with different location formats...")
        # Try other location formats
        for loc in ["us", "europe", "asia"]:
            os.environ["VERTEX_AI_LOCATION"] = loc
            print(f"\nTrying location: {loc}")

            from onellm import OpenAI

            client = OpenAI()

            try:
                response = client.chat.completions.create(
                    model="vertexai/gemini-1.5-flash",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                print(f"✅ SUCCESS with location: {loc}")
                result = True
                break
            except Exception as e:
                print(f"❌ Failed: {str(e)[:50]}...")

    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")

#!/usr/bin/env python3
"""
Test AWS Bedrock with Amazon Nova models.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()


def test_bedrock():
    """Test AWS Bedrock provider with Nova models."""
    print("🔍 Testing AWS Bedrock provider with Amazon Nova models...")

    from onellm import OpenAI

    try:
        client = OpenAI()

        # Test with Amazon Nova models (newer, might be enabled by default)
        models = [
            "bedrock/nova-micro",
            "bedrock/nova-lite",
            "bedrock/nova-pro",
            "bedrock/amazon.nova-micro-v1:0",
            "bedrock/amazon.nova-lite-v1:0",
        ]

        for model in models:
            try:
                print(f"\nTesting {model}...")
                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )

                content = response.choices[0].message.content
                print(f"✅ SUCCESS: {content}")
                print(f"Model: {model}")
                print("\n🎉 AWS BEDROCK IS NOW WORKING!")
                return True

            except Exception as e:
                error_msg = str(e)
                if "access denied" in error_msg.lower():
                    print(f"❌ {model}: Not enabled in account")
                elif "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                    print(f"❌ {model}: Invalid model ID")
                else:
                    print(f"❌ {model}: {error_msg[:100]}...")

    except Exception as e:
        print(f"❌ Client initialization error: {e}")

    return False


if __name__ == "__main__":
    result = test_bedrock()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")

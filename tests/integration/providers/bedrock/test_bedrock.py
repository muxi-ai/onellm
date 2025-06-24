#!/usr/bin/env python3
"""
Test AWS Bedrock provider implementation.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_bedrock():
    """Test AWS Bedrock provider."""
    print("🔍 Testing AWS Bedrock provider...")
    print(f"Bedrock config file exists: {os.path.exists('bedrock.json')}")
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Test with common Bedrock models
        models = [
            "bedrock/claude-3-5-sonnet",
            "bedrock/claude-3-haiku",
            "bedrock/claude-instant",
            "bedrock/llama3-8b",
            "bedrock/titan-text-lite"
        ]
        
        for model in models:
            try:
                print(f"\nTesting {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=10
                )
                
                content = response.choices[0].message.content
                print(f"✅ SUCCESS: {content}")
                print(f"Model: {model}")
                print("\n🎉 AWS BEDROCK IS NOW WORKING!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                if "throttling" in error_msg.lower():
                    print(f"⚠️  {model}: Rate limited (normal for new accounts)")
                elif "access" in error_msg.lower() or "not authorized" in error_msg.lower():
                    print(f"❌ {model}: Access denied - model not enabled")
                else:
                    print(f"❌ {model}: {error_msg[:200]}")
    
    except Exception as e:
        print(f"❌ Client initialization error: {e}")
    
    return False

if __name__ == "__main__":
    result = test_bedrock()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
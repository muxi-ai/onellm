#!/usr/bin/env python3
"""
Test Anyscale provider implementation.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load environment variables
load_dotenv()

def test_anyscale():
    """Test Anyscale provider."""
    print("🔍 Testing Anyscale provider...")
    
    # Check if API key is set
    api_key = os.environ.get('ANYSCALE_API_KEY')
    if not api_key:
        print("❌ ANYSCALE_API_KEY not found in environment")
        print("Please set it with: export ANYSCALE_API_KEY='esecret_your_key_here'")
        return False
    
    print(f"API Key prefix: {api_key[:10]}...")
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Test with common Anyscale models
        models = [
            "anyscale/llama3-8b",  # Using alias
            "anyscale/meta-llama/Meta-Llama-3-8B-Instruct",  # Full name
            "anyscale/mistral-7b",  # Using alias
            "anyscale/codellama-7b"  # Code model
        ]
        
        for model in models:
            try:
                print(f"\nTesting {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=10
                )
                
                content = response.choices[0].message["content"]
                print(f"✅ SUCCESS: {content}")
                print(f"Model: {model}")
                print("\n🎉 ANYSCALE IS WORKING!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower():
                    print(f"⚠️  {model}: Rate limited (30 concurrent requests max)")
                elif "not found" in error_msg.lower():
                    print(f"❌ {model}: Model not found")
                elif "authentication" in error_msg.lower():
                    print(f"❌ {model}: Authentication failed - check API key")
                else:
                    print(f"❌ {model}: {error_msg[:150]}...")
    
    except Exception as e:
        print(f"❌ Client initialization error: {e}")
    
    return False

if __name__ == "__main__":
    result = test_anyscale()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
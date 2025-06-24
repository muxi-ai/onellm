#!/usr/bin/env python3
"""
Test AWS Bedrock with models that have access granted.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_bedrock():
    """Test AWS Bedrock provider with enabled models."""
    print("🔍 Testing AWS Bedrock provider with enabled models...")
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Test with models that show "Access granted" in your screenshot
        models = [
            "bedrock/titan-text-lite",
            "bedrock/titan-text-express", 
            "bedrock/claude-3-sonnet",
            "bedrock/claude-3-haiku",
            "bedrock/llama3-8b",
            "bedrock/mistral-7b"
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
                print("\n🎉 AWS BEDROCK IS NOW WORKING!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ {model}: {error_msg[:150]}...")
    
    except Exception as e:
        print(f"❌ Client initialization error: {e}")
    
    return False

if __name__ == "__main__":
    result = test_bedrock()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
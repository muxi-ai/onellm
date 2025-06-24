#!/usr/bin/env python3
"""
Test Vertex AI provider implementation.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Set the credential path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vertexai.json'

def test_vertex():
    """Test Vertex AI provider."""
    print("🔍 Testing Vertex AI provider...")
    print(f"Credentials file exists: {os.path.exists('vertexai.json')}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}")
    
    # First check if google-auth is installed
    try:
        import google.auth
        print("✅ google-auth is installed")
    except ImportError:
        print("❌ google-auth is NOT installed")
        print("Please install it with: pip install google-auth")
        return False
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Test with common Vertex AI models
        models = [
            "vertexai/gemini-1.5-flash",
            "vertexai/gemini-1.5-flash-001",
            "vertexai/gemini-1.5-flash-002",
            "vertexai/gemini-1.5-pro",
            "vertexai/gemini-1.5-pro-001",
            "vertexai/gemini-1.5-pro-002",
            "vertexai/gemini-pro",
            "vertexai/gemini-pro-001",
            "vertexai/gemini-pro-vision"
        ]
        
        for model in models:
            try:
                print(f"\nTesting {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=10
                )
                
                content = response.choices[0].message.get('content', '')
                print(f"✅ SUCCESS: {content}")
                print(f"Model: {model}")
                print("\n🎉 VERTEX AI IS NOW WORKING!")
                return True
                
            except Exception as e:
                print(f"❌ {model}: {str(e)[:200]}")
    
    except Exception as e:
        print(f"❌ Client initialization error: {e}")
    
    return False

if __name__ == "__main__":
    result = test_vertex()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
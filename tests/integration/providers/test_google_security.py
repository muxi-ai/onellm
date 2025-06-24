#!/usr/bin/env python3
"""
Test Google provider after security fix (API key in header instead of URL).
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_google():
    """Test Google provider with API key in header."""
    print("🔍 Testing Google provider after security fix...")
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Test with Gemini model
        print("\nTesting google/gemini-1.5-flash...")
        response = client.chat.completions.create(
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        content = response.choices[0].message["content"]
        print(f"✅ SUCCESS: {content}")
        print("Google provider working with API key in headers!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}...")
        return False

if __name__ == "__main__":
    result = test_google()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
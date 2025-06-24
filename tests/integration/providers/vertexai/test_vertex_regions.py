#!/usr/bin/env python3
"""
Test Vertex AI provider with different regions.
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

def test_vertex_regions():
    """Test Vertex AI provider with different regions."""
    print("🔍 Testing Vertex AI provider with different regions...")
    
    from onellm import OpenAI
    
    # Test different regions
    regions = [
        "us-central1",
        "us-east1", 
        "us-west1",
        "europe-west1",
        "europe-west4",
        "asia-northeast1"
    ]
    
    # Test a few model variations
    models = [
        "vertexai/gemini-1.5-flash",
        "vertexai/gemini-1.5-flash-001",
        "vertexai/gemini-pro"
    ]
    
    for region in regions:
        print(f"\n🌍 Testing region: {region}")
        
        # Set region via environment variable
        os.environ['VERTEX_AI_LOCATION'] = region
        
        # Create new client for each region
        try:
            client = OpenAI()
            
            for model in models:
                try:
                    print(f"  Testing {model}...", end=" ")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Say hello"}],
                        max_tokens=10
                    )
                    
                    content = response.choices[0].message.get('content', '')
                    print(f"✅ SUCCESS: {content}")
                    print(f"\n🎉 VERTEX AI WORKS IN REGION: {region}")
                    print(f"Working model: {model}")
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    if "not found" in error_msg.lower():
                        print("❌ Not found")
                    else:
                        print(f"❌ {error_msg[:50]}...")
        
        except Exception as e:
            print(f"  ❌ Client error: {str(e)[:100]}...")
    
    return False

if __name__ == "__main__":
    result = test_vertex_regions()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
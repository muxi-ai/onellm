#!/usr/bin/env python3
"""
Simple test runner for provider integration tests.
Tests only basic functionality with working models.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Source the API keys file if it exists
api_keys_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts', 'api-keys.sh')
if os.path.exists(api_keys_file):
    print(f"Loading API keys from {api_keys_file}")
    # Parse the shell script and set environment variables
    with open(api_keys_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                # Extract variable name and value
                var_assignment = line[7:]  # Remove 'export '
                var_name, var_value = var_assignment.split('=', 1)
                # Remove quotes if present
                var_value = var_value.strip('"').strip("'")
                # Only set if value is not empty
                if var_value:
                    os.environ[var_name] = var_value
else:
    print(f"Warning: API keys file not found at {api_keys_file}")

async def test_groq():
    """Test Groq provider with real API."""
    print("Testing Groq provider...")
    
    if not os.getenv("GROQ_API_KEY"):
        print("  SKIPPED: GROQ_API_KEY not set")
        return None
    
    try:
        from onellm.providers.groq import GroqProvider
        provider = GroqProvider()
        response = await provider.create_chat_completion(
            model="llama3-8b-8192",  # Working model name
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_deepseek():
    """Test DeepSeek provider with real API."""
    print("Testing DeepSeek provider...")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  SKIPPED: DEEPSEEK_API_KEY not set")
        return None
    
    try:
        from onellm.providers.deepseek import DeepSeekProvider
        provider = DeepSeekProvider()
        response = await provider.create_chat_completion(
            model="deepseek-chat",  # Try without prefix
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_google():
    """Test Google provider with real API."""
    print("Testing Google provider...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("  SKIPPED: GOOGLE_API_KEY not set")
        return None
    
    try:
        from onellm.providers.google import GoogleProvider
        provider = GoogleProvider()
        response = await provider.create_chat_completion(
            model="gemini-1.5-flash",  # Try without prefix
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_together():
    """Test Together provider with real API."""
    print("Testing Together provider...")
    
    if not os.getenv("TOGETHER_API_KEY"):
        print("  SKIPPED: TOGETHER_API_KEY not set")
        return None
    
    try:
        from onellm.providers.together import TogetherProvider
        provider = TogetherProvider()
        response = await provider.create_chat_completion(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Try without prefix
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_ollama():
    """Test Ollama provider with local model."""
    print("Testing Ollama provider...")
    
    try:
        from onellm.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        response = await provider.create_chat_completion(
            model="llama3:8b",  # Available model
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def main():
    """Run basic provider tests."""
    print("🚀 Running Basic OneLLM Provider Tests")
    print("=" * 50)
    
    tests = [
        test_groq,
        test_deepseek, 
        test_google,
        test_together,
        test_ollama
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR: {e}")
            results.append(False)
        print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")  
    print(f"⏭️  SKIPPED: {skipped}")
    print(f"📋 TOTAL: {len(results)}")
    
    if passed > 0:
        print(f"\n🎉 {passed} providers working successfully!")
    if failed > 0:
        print(f"\n⚠️  {failed} providers need model name/configuration fixes.")

if __name__ == "__main__":
    asyncio.run(main())
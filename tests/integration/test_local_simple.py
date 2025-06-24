#!/usr/bin/env python3
"""
Simple test for local OneLLM providers focusing on core functionality.
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

async def test_ollama_basic():
    """Test basic Ollama functionality."""
    print("Testing Ollama provider...")
    
    try:
        from onellm.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        
        # Basic test
        response = await provider.create_chat_completion(
            model="llama3:8b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ BASIC: {response.choices[0].message['content']}")
        
        # Dynamic endpoint test
        response2 = await provider.create_chat_completion(
            model="llama3:8b@localhost:11434",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=15
        )
        
        print(f"  ✅ ENDPOINT ROUTING: {response2.choices[0].message['content']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_llama_cpp_basic():
    """Test basic llama.cpp functionality."""
    print("Testing llama.cpp provider...")
    
    model_path = "/Users/ran/Projects/muxi/code/onellm/meta-llama-3-8b-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        print(f"  SKIPPED: Model file not found")
        return None
    
    try:
        # First, let me check if llama-cpp-python is installed
        try:
            import llama_cpp
            print(f"  llama-cpp-python version: {llama_cpp.__version__}")
        except ImportError:
            print(f"  ❌ llama-cpp-python not installed")
            return False
        
        from onellm.providers.llama_cpp import LlamaCppProvider
        provider = LlamaCppProvider()
        
        # Test with absolute path
        response = await provider.create_chat_completion(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def main():
    """Run simple local provider tests."""
    print("🏠 Testing Local OneLLM Providers (Simple)")
    print("=" * 50)
    
    results = {}
    
    # Test Ollama
    results["Ollama"] = await test_ollama_basic()
    print()
    
    # Test llama.cpp
    results["llama.cpp"] = await test_llama_cpp_basic()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 LOCAL PROVIDER SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        if result is True:
            print(f"✅ {name}: WORKING")
        elif result is False:
            print(f"❌ {name}: FAILED")
        else:
            print(f"⏭️  {name}: SKIPPED")
    
    working = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    if total > 0:
        print(f"\n🎯 LOCAL SUCCESS RATE: {working}/{total} ({working/total*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
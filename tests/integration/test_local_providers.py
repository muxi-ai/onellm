#!/usr/bin/env python3
"""
Test runner for local OneLLM providers (Ollama and llama.cpp).
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_ollama():
    """Test Ollama provider with local models."""
    print("Testing Ollama provider...")

    try:
        from onellm.providers.ollama import OllamaProvider

        provider = OllamaProvider()

        print(f"  API Base: {provider.api_base}")
        print(f"  Requires API Key: {provider.requires_api_key}")

        # Test with available model
        response = await provider.create_chat_completion(
            model="llama3:8b",  # We know this model is available
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")

        # Test dynamic endpoint routing
        print("  Testing dynamic endpoint routing...")
        response2 = await provider.create_chat_completion(
            model="llama3:8b@localhost:11434",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=15,
        )

        print(f"  ✅ ENDPOINT ROUTING SUCCESS: {response2.choices[0].message['content']}")

        # Test streaming
        print("  Testing streaming...")
        chunks = []
        stream = await provider.create_chat_completion(
            model="llama3:8b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            stream=True,
        )
        async for chunk in stream:
            chunks.append(chunk)

        print(f"  ✅ STREAMING SUCCESS: {len(chunks)} chunks received")

        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_llama_cpp():
    """Test llama.cpp provider with local model."""
    print("Testing llama.cpp provider...")

    model_path = "/Users/ran/Projects/muxi/code/onellm/meta-llama-3-8b-instruct-q4_k_m.gguf"

    if not os.path.exists(model_path):
        print(f"  SKIPPED: Model file not found at {model_path}")
        return None

    try:
        from onellm.providers.llama_cpp import LlamaCppProvider

        provider = LlamaCppProvider()

        print(f"  Model path exists: {os.path.exists(model_path)}")
        print(f"  Model size: {os.path.getsize(model_path) / 1024 / 1024 / 1024:.1f} GB")

        # Test with full path (llama.cpp provider expects just the path, not prefixed)
        response = await provider.create_chat_completion(
            model=model_path, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")

        # Test model caching (second call should be faster)
        print("  Testing model caching...")
        response2 = await provider.create_chat_completion(
            model=model_path, messages=[{"role": "user", "content": "Count to 3"}], max_tokens=15
        )

        print(f"  ✅ CACHING SUCCESS: {response2.choices[0].message['content']}")

        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_ollama_models():
    """Test what models are available in Ollama."""
    print("Checking available Ollama models...")

    try:
        from onellm.providers.ollama import OllamaProvider

        provider = OllamaProvider()

        models = await provider.list_models()
        print(f"  Available models: {models}")

        return models

    except Exception as e:
        print(f"  ❌ ERROR listing models: {e}")
        return []


async def main():
    """Run local provider tests."""
    print("🏠 Testing Local OneLLM Providers")
    print("=" * 50)

    # Check available Ollama models first
    await test_ollama_models()
    print()

    # Test providers
    tests = [
        ("Ollama", test_ollama),
        ("llama.cpp", test_llama_cpp),
    ]

    results = {}

    for provider_name, test_func in tests:
        try:
            result = await test_func()
            results[provider_name] = result
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR in {provider_name}: {e}")
            results[provider_name] = False
        print()

    # Summary
    print("=" * 50)
    print("📊 LOCAL PROVIDER TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print("\n🟢 WORKING PROVIDERS:")
    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")

    print("\n🔴 FAILED PROVIDERS:")
    for name, result in results.items():
        if result is False:
            print(f"  ❌ {name}")

    print("\n⚪ SKIPPED PROVIDERS:")
    for name, result in results.items():
        if result is None:
            print(f"  ⏭️  {name}")

    print("\n📈 LOCAL PROVIDER STATS:")
    print(f"✅ WORKING: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"⏭️  SKIPPED: {skipped}")
    print(f"📋 TOTAL: {len(results)}")

    if passed + failed > 0:
        success_rate = (passed / (passed + failed)) * 100
        print(f"🎯 SUCCESS RATE: {success_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())

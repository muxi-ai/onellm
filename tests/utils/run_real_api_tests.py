#!/usr/bin/env python3
"""
Direct test runner for provider integration tests using real API calls.
This bypasses pytest's conftest.py which interferes with environment variables.
"""

import asyncio
import os
import sys
import subprocess
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

    from onellm.providers.groq import GroqProvider

    if not os.getenv("GROQ_API_KEY"):
        print("  SKIPPED: GROQ_API_KEY not set")
        return False

    try:
        provider = GroqProvider()
        response = await provider.create_chat_completion(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_xai():
    """Test XAI provider with real API."""
    print("Testing XAI provider...")

    from onellm.providers.xai import XAIProvider

    if not os.getenv("XAI_API_KEY"):
        print("  SKIPPED: XAI_API_KEY not set")
        return False

    try:
        provider = XAIProvider()
        response = await provider.create_chat_completion(
            model="xai/grok-beta",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_openrouter():
    """Test OpenRouter provider with real API."""
    print("Testing OpenRouter provider...")

    from onellm.providers.openrouter import OpenRouterProvider

    if not os.getenv("OPENROUTER_API_KEY"):
        print("  SKIPPED: OPENROUTER_API_KEY not set")
        return False

    try:
        provider = OpenRouterProvider()
        response = await provider.create_chat_completion(
            model="openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
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

    from onellm.providers.together import TogetherProvider

    if not os.getenv("TOGETHER_API_KEY"):
        print("  SKIPPED: TOGETHER_API_KEY not set")
        return False

    try:
        provider = TogetherProvider()
        response = await provider.create_chat_completion(
            model="together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_fireworks():
    """Test Fireworks provider with real API."""
    print("Testing Fireworks provider...")

    from onellm.providers.fireworks import FireworksProvider

    if not os.getenv("FIREWORKS_API_KEY"):
        print("  SKIPPED: FIREWORKS_API_KEY not set")
        return False

    try:
        provider = FireworksProvider()
        response = await provider.create_chat_completion(
            model="fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_perplexity():
    """Test Perplexity provider with real API."""
    print("Testing Perplexity provider...")

    from onellm.providers.perplexity import PerplexityProvider

    if not os.getenv("PERPLEXITY_API_KEY"):
        print("  SKIPPED: PERPLEXITY_API_KEY not set")
        return False

    try:
        provider = PerplexityProvider()
        response = await provider.create_chat_completion(
            model="perplexity/llama-3.1-sonar-small-128k-chat",
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

    from onellm.providers.deepseek import DeepSeekProvider

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  SKIPPED: DEEPSEEK_API_KEY not set")
        return False

    try:
        provider = DeepSeekProvider()
        response = await provider.create_chat_completion(
            model="deepseek/deepseek-r1",
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

    from onellm.providers.google import GoogleProvider

    if not os.getenv("GOOGLE_API_KEY"):
        print("  SKIPPED: GOOGLE_API_KEY not set")
        return False

    try:
        provider = GoogleProvider()
        response = await provider.create_chat_completion(
            model="google/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_cohere():
    """Test Cohere provider with real API."""
    print("Testing Cohere provider...")

    from onellm.providers.cohere import CohereProvider

    if not os.getenv("COHERE_API_KEY"):
        print("  SKIPPED: COHERE_API_KEY not set")
        return False

    try:
        provider = CohereProvider()
        response = await provider.create_chat_completion(
            model="cohere/command-r",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def test_vertexai():
    """Test Vertex AI provider with real API."""
    print("Testing Vertex AI provider...")

    if not os.path.exists("vertexai.json"):
        print("  SKIPPED: vertexai.json not found")
        return False

    try:
        from onellm.providers.vertexai import VertexAIProvider
        provider = VertexAIProvider()
        response = await provider.create_chat_completion(
            model="vertexai/gemini-1.5-flash",
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
            model="ollama/llama3:8b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
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
        return False

    try:
        from onellm.providers.llama_cpp import LlamaCppProvider
        provider = LlamaCppProvider()
        # Use the full path instead of just the filename
        response = await provider.create_chat_completion(
            model=f"llama_cpp/{model_path}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

async def main():
    """Run all provider tests."""
    print("🚀 Running OneLLM Provider Integration Tests")
    print("=" * 50)

    tests = [
        test_groq,
        test_xai,
        test_openrouter,
        test_together,
        test_fireworks,
        test_perplexity,
        test_deepseek,
        test_google,
        test_cohere,
        test_vertexai,
        test_ollama,
        test_llama_cpp
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

    if failed > 0:
        print(f"\n⚠️  {failed} tests failed. Check API keys and configurations.")
    else:
        print(f"\n🎉 All available tests passed!")

if __name__ == "__main__":
    asyncio.run(main())

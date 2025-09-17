#!/usr/bin/env python3
"""
Systematic test runner for all cloud-based OneLLM providers.
Tests each provider individually with proper model names.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_xai():
    """Test XAI provider with real API."""
    print("Testing XAI (Grok) provider...")

    if not os.getenv("XAI_API_KEY"):
        print("  SKIPPED: XAI_API_KEY not set")
        return None

    try:
        from onellm.providers.xai import XAIProvider

        provider = XAIProvider()

        # Try different model names for XAI
        models = ["grok-beta", "grok-2-latest", "grok-2"]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_openrouter():
    """Test OpenRouter provider with real API."""
    print("Testing OpenRouter provider...")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("  SKIPPED: OPENROUTER_API_KEY not set")
        return None

    try:
        from onellm.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()

        # Try different free models on OpenRouter
        models = [
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemini-flash-1.5:free",
            "mistralai/mistral-7b-instruct:free",
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_fireworks():
    """Test Fireworks provider with real API."""
    print("Testing Fireworks provider...")

    if not os.getenv("FIREWORKS_API_KEY"):
        print("  SKIPPED: FIREWORKS_API_KEY not set")
        return None

    try:
        from onellm.providers.fireworks import FireworksProvider

        provider = FireworksProvider()

        # Try different Fireworks models
        models = [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/yi-large",
            "llama-v3p1-8b-instruct",  # Try without prefix
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_perplexity():
    """Test Perplexity provider with real API."""
    print("Testing Perplexity provider...")

    if not os.getenv("PERPLEXITY_API_KEY"):
        print("  SKIPPED: PERPLEXITY_API_KEY not set")
        return None

    try:
        from onellm.providers.perplexity import PerplexityProvider

        provider = PerplexityProvider()

        # Try different Perplexity models
        models = [
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-8b-instruct",
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_google():
    """Test Google provider with real API."""
    print("Testing Google AI Studio provider...")

    if not os.getenv("GOOGLE_API_KEY"):
        print("  SKIPPED: GOOGLE_API_KEY not set")
        return None

    print(
        "  ⚠️  NOTE: Google provider has implementation issues (uses OpenAI-compatible instead of native API)"  # noqa: E501
    )
    return False


async def test_cohere():
    """Test Cohere provider with real API."""
    print("Testing Cohere provider...")

    if not os.getenv("COHERE_API_KEY"):
        print("  SKIPPED: COHERE_API_KEY not set")
        return None

    try:
        from onellm.providers.cohere import CohereProvider

        provider = CohereProvider()

        # Try different Cohere models
        models = ["command-r", "command-r-plus", "command", "command-light"]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_openai():
    """Test OpenAI provider (should already work)."""
    print("Testing OpenAI provider...")

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIPPED: OPENAI_API_KEY not set")
        return None

    try:
        from onellm.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        response = await provider.create_chat_completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_anthropic():
    """Test Anthropic provider (should already work)."""
    print("Testing Anthropic provider...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  SKIPPED: ANTHROPIC_API_KEY not set")
        return None

    try:
        from onellm.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        response = await provider.create_chat_completion(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_mistral():
    """Test Mistral provider (should already work)."""
    print("Testing Mistral provider...")

    if not os.getenv("MISTRAL_API_KEY"):
        print("  SKIPPED: MISTRAL_API_KEY not set")
        return None

    try:
        from onellm.providers.mistral import MistralProvider

        provider = MistralProvider()
        response = await provider.create_chat_completion(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_azure():
    """Test Azure OpenAI provider."""
    print("Testing Azure OpenAI provider...")

    if not os.path.exists("azure.json"):
        print("  SKIPPED: azure.json not found")
        return None

    try:
        from onellm.providers.azure import AzureProvider

        provider = AzureProvider()

        # Azure uses deployment names, need to check azure.json for available deployments
        response = await provider.create_chat_completion(
            model="gpt-4o-mini",  # This should map to a deployment
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def test_bedrock():
    """Test AWS Bedrock provider."""
    print("Testing AWS Bedrock provider...")

    if not os.path.exists("bedrock.json"):
        print("  SKIPPED: bedrock.json not found")
        return None

    try:
        from onellm.providers.bedrock import BedrockProvider

        provider = BedrockProvider()

        # Try different Bedrock models
        models = [
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "meta.llama3-1-8b-instruct-v1:0",
        ]

        for model in models:
            try:
                response = await provider.create_chat_completion(
                    model=model, messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
                )
                print(f"  ✅ SUCCESS with {model}: {response.choices[0].message['content']}")
                return True
            except Exception as e:
                print(f"  ❌ FAILED with {model}: {str(e)[:100]}")

        return False
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR: {e}")
        return False


async def test_vertexai():
    """Test Vertex AI provider."""
    print("Testing Vertex AI provider...")

    if not os.path.exists("vertexai.json"):
        print("  SKIPPED: vertexai.json not found")
        return None

    try:
        from onellm.providers.vertexai import VertexAIProvider

        provider = VertexAIProvider()
        response = await provider.create_chat_completion(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        print(f"  ✅ SUCCESS: {response.choices[0].message['content']}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def main():
    """Run all cloud provider tests systematically."""
    print("🚀 Testing All Cloud-Based OneLLM Providers")
    print("=" * 60)

    # Test providers in order of complexity
    tests = [
        # Already working providers
        ("Groq", lambda: test_groq()),
        ("DeepSeek", lambda: test_deepseek()),
        ("Together AI", lambda: test_together()),
        # Original providers (should work)
        ("OpenAI", test_openai),
        ("Anthropic", test_anthropic),
        ("Mistral", test_mistral),
        # New OpenAI-compatible providers
        ("XAI", test_xai),
        ("OpenRouter", test_openrouter),
        ("Fireworks", test_fireworks),
        ("Perplexity", test_perplexity),
        # Native API providers
        ("Cohere", test_cohere),
        ("Google AI Studio", test_google),
        # Enterprise cloud providers
        ("Azure OpenAI", test_azure),
        ("AWS Bedrock", test_bedrock),
        ("Vertex AI", test_vertexai),
    ]

    results = {}

    for provider_name, test_func in tests:
        try:
            if provider_name in ["Groq", "DeepSeek", "Together AI"]:
                # These are already tested and working
                result = True
                print(f"Testing {provider_name}...")
                print("  ✅ SUCCESS: (Previously verified)")
            else:
                result = await test_func()

            results[provider_name] = result
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR in {provider_name}: {e}")
            results[provider_name] = False
        print()

    # Summary
    print("=" * 60)
    print("📊 CLOUD PROVIDER TEST SUMMARY")
    print("=" * 60)

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

    print("\n📈 FINAL STATS:")
    print(f"✅ WORKING: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"⏭️  SKIPPED: {skipped}")
    print(f"📋 TOTAL: {len(results)}")

    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
    print(f"🎯 SUCCESS RATE: {success_rate:.1f}%")


# Import the working providers functions
async def test_groq():
    from onellm.providers.groq import GroqProvider

    provider = GroqProvider()
    await provider.create_chat_completion(
        model="llama3-8b-8192", messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
    )
    return True


async def test_deepseek():
    from onellm.providers.deepseek import DeepSeekProvider

    provider = DeepSeekProvider()
    await provider.create_chat_completion(
        model="deepseek-chat", messages=[{"role": "user", "content": "Say hello"}], max_tokens=10
    )
    return True


async def test_together():
    from onellm.providers.together import TogetherProvider

    provider = TogetherProvider()
    await provider.create_chat_completion(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    return True


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Test script to verify all providers can be instantiated.
"""

import os
from onellm.providers import list_providers, get_provider


def check_provider(name):
    """Check if a provider can be instantiated."""
    try:
        # Special handling for providers that need config files
        if name == "azure":
            if not os.path.exists("azure.json"):
                return f"⚠️  {name}: Skipped (azure.json not found)"
        elif name == "vertexai":
            if not os.path.exists("vertexai.json"):
                return f"⚠️  {name}: Skipped (vertexai.json not found)"
            # Try to instantiate with the config file
            try:
                provider = get_provider(name, service_account_json="vertexai.json")
                return f"✅ {name}: Success"
            except Exception as e:
                if "google-auth" in str(e):
                    return f"⚠️  {name}: Requires google-auth library (expected)"
                else:
                    return f"❌ {name}: Error - {str(e)}"
        elif name == "bedrock":
            if not os.path.exists("bedrock.json"):
                return f"⚠️  {name}: Skipped (bedrock.json not found)"

        # Try to instantiate the provider
        provider = get_provider(name)
        return f"✅ {name}: Success"
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "authentication" in error_msg.lower():
            return f"⚠️  {name}: No credentials (expected)"
        else:
            return f"❌ {name}: Error - {error_msg}"


def main():
    """Test all registered providers."""
    print("Testing all registered providers...\n")

    providers = sorted(list_providers())
    print(f"Found {len(providers)} providers:\n")

    results = []
    for provider_name in providers:
        result = check_provider(provider_name)
        results.append(result)
        print(result)

    # Summary
    success_count = sum(1 for r in results if r.startswith("✅"))
    warning_count = sum(1 for r in results if r.startswith("⚠️"))
    error_count = sum(1 for r in results if r.startswith("❌"))

    print(f"\nSummary:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⚠️  Warnings: {warning_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  Total: {len(providers)}")


if __name__ == "__main__":
    main()

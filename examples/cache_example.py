#!/usr/bin/env python3
"""
Example demonstrating OneLLM's semantic caching feature.

This example shows how to:
1. Enable semantic caching to reduce API costs
2. See cache hits for identical and similar queries
3. View cache statistics
"""

import onellm
from onellm import ChatCompletion
import time

def main():
    print("=" * 60)
    print("OneLLM Semantic Cache Example")
    print("=" * 60)
    print()

    # Initialize cache (loads embedding model, ~13s one-time cost)
    print("Initializing cache...")
    onellm.init_cache()
    print("✅ Cache initialized with multilingual support\n")

    model = "openai/gpt-4"

    # First query - cache miss, makes API call
    print("1. First query (cache miss, API call):")
    print("-" * 60)
    start = time.time()
    response = ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    elapsed = time.time() - start
    print(f"Response: {response.choices[0].message['content'][:100]}...")
    print(f"Time: {elapsed:.2f}s")
    print(f"Cache stats: {onellm.cache_stats()}\n")

    # Second query - exact match, cache hit (~1ms)
    print("2. Exact same query (cache hit, instant):")
    print("-" * 60)
    start = time.time()
    response = ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    elapsed = time.time() - start
    print(f"Response: {response.choices[0].message['content'][:100]}...")
    print(f"Time: {elapsed:.2f}s (from hash cache)")
    print(f"Cache stats: {onellm.cache_stats()}\n")

    # Third query - semantically similar, cache hit (~18ms)
    print("3. Similar query (semantic cache hit, ~18ms):")
    print("-" * 60)
    start = time.time()
    response = ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Tell me about the Python programming language"}]
    )
    elapsed = time.time() - start
    print(f"Response: {response.choices[0].message['content'][:100]}...")
    print(f"Time: {elapsed:.2f}s (from semantic cache)")
    print(f"Cache stats: {onellm.cache_stats()}\n")

    # Fourth query - different topic, cache miss
    print("4. Different query (cache miss, API call):")
    print("-" * 60)
    start = time.time()
    response = ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    elapsed = time.time() - start
    print(f"Response: {response.choices[0].message['content'][:100]}...")
    print(f"Time: {elapsed:.2f}s")
    print(f"Cache stats: {onellm.cache_stats()}\n")

    # Show final statistics
    stats = onellm.cache_stats()
    hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) if stats["misses"] > 0 else 0
    print("=" * 60)
    print("Final Cache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {hit_rate:.1%}")
    print(f"  Entries: {stats['entries']}")
    print("=" * 60)

    # Advanced: Adjust similarity threshold
    print("\nAdvanced: Custom similarity threshold")
    print("-" * 60)
    onellm.clear_cache()
    onellm.init_cache(p=0.85)  # More aggressive matching
    print("Cache re-initialized with p=0.85 (more aggressive matching)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example demonstrating OneLLM's auto routing feature.

This example shows how to:
1. Register a developer-authored routing map
2. Let model="auto" pick the right model per request
3. Pin a subtree with model="auto/<label>"
4. Inspect decisions with response.routing, explain_route() and routing_stats()

Requires: pip install "onellm[routing]"
"""

import onellm
from onellm import ChatCompletion


def main():
    print("=" * 60)
    print("OneLLM Auto Routing Example")
    print("=" * 60)
    print()

    # Register the routing map once at startup. The map is 100%
    # developer-authored: labels, examples and model choices are yours.
    # (First call loads the local embedding model, one-time cost.)
    print("Initializing routing...")
    onellm.init_routing(
        {
            "default": "openai/gpt-4o-mini",
            "code": {
                "description": "Programming, debugging, code review",
                "examples": [
                    "fix this function",
                    "why does this test fail?",
                    "review my pull request",
                ],
                "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-5"],
            },
            "research": {
                "description": "Deep analysis, comparisons, long documents",
                "examples": [
                    "compare these two papers",
                    "summarize this report in depth",
                ],
                "models": "openai/gpt-4o",
            },
        }
    )
    print("✅ Routing initialized\n")

    # 1. Dry-run decisions without making any provider calls
    print("1. Dry-run routing decisions (no API calls):")
    print("-" * 60)
    for content in [
        "Why does this unit test fail intermittently?",
        "Compare the approaches in these two papers.",
        "What's the weather like?",
    ]:
        decision = onellm.explain_route(messages=[{"role": "user", "content": content}])
        print(f"  {content!r}")
        print(
            f"    -> {decision['resolved_path'] or '(root default)'} "
            f"= {decision['model']} (source: {decision['source']})"
        )
    print()

    # 2. A real request: the classifier picks the label, the label
    #    resolves to a model plus fallback chain
    print("2. Routed chat completion:")
    print("-" * 60)
    response = ChatCompletion.create(
        model="auto",
        messages=[{"role": "user", "content": "Why does this unit test fail intermittently?"}],
    )
    print(f"  Answered by: {response.model}")
    print(f"  Routing: {response.routing}")
    print()

    # 3. Pin a subtree: skip classification of the top level entirely
    print("3. Explicit entry point (auto/code):")
    print("-" * 60)
    response = ChatCompletion.create(
        model="auto/code",
        messages=[{"role": "user", "content": "Refactor this loop into a comprehension."}],
    )
    print(f"  Answered by: {response.model}")
    print(f"  Resolved path: {response.routing['resolved_path']}")
    print()

    # 4. Aggregate statistics
    print("4. Routing statistics:")
    print("-" * 60)
    stats = onellm.routing_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 60)
    print("Tips:")
    print("- Keep the map in version control: onellm.load_routing_map('routing.yaml')")
    print("- Assert explain_route() decisions in CI to catch silent rerouting")
    print("- Set ONELLM_ROUTING_DEBUG=1 to log what the classifier sees")
    print("=" * 60)


if __name__ == "__main__":
    main()

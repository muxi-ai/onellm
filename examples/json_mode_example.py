#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the JSON mode feature of muxi-llm.

This example shows how to use the response_format parameter to request
JSON responses from models that support it, and demonstrates the fallback
to text-based instructions for models that don't support native JSON mode.
"""

import os
import json
import muxi_llm

# Set API key from environment
muxi_llm.openai_api_key = os.environ.get("OPENAI_API_KEY")

# Example with OpenAI model that supports JSON mode
print("Example 1: Using OpenAI model with JSON mode")
print("-" * 50)

response = muxi_llm.ChatCompletion.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a list of 3 todo items as JSON"}
    ],
    response_format={"type": "json_object"}
)

print("JSON response from OpenAI:")
print(response.choices[0].message["content"])
print()

# Parse the JSON to validate it's properly formatted
try:
    parsed = json.loads(response.choices[0].message["content"])
    print("Successfully parsed JSON response:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")

print("\n" + "-" * 50 + "\n")

# Example showing fallback behavior for non-OpenAI model
# This requires setting up other providers' API keys
# Uncomment to test with other providers

"""
print("Example 2: Using a provider that doesn't support JSON mode natively")
print("-" * 50)

# Set up your API key for other providers
# muxi_llm.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

try:
    response = muxi_llm.ChatCompletion.create(
        model="anthropic/claude-3-opus",  # Use the appropriate model for your provider
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a list of 3 todo items as JSON"}
        ],
        response_format={"type": "json_object"}
    )

    print("Response from non-OpenAI provider:")
    print(response.choices[0].message["content"])
    print()

    # Parse the JSON to validate it's properly formatted
    try:
        parsed = json.loads(response.choices[0].message["content"])
        print("Successfully parsed JSON response:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
except Exception as e:
    print(f"Error: {e}")
"""

print("\n" + "-" * 50 + "\n")

# Example showing using parameters with fallback models
print("Example 3: Using JSON mode with fallback models")
print("-" * 50)

response = muxi_llm.ChatCompletion.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a list of 3 todo items as JSON"}
    ],
    response_format={"type": "json_object"},
    fallback_models=["openai/gpt-3.5-turbo"]  # Both support JSON mode
)

print("JSON response with fallback models:")
print(response.choices[0].message["content"])

print("\n" + "Done!" + "\n")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# ============================================================================ #
# MUXI-LLM EXAMPLE: JSON Mode for Structured Responses
# ============================================================================ #
#
# This example demonstrates how to use OneLLM's JSON mode feature to get
# structured, parseable responses from LLMs.
# Key features demonstrated:
#
# - Requesting JSON-formatted responses
# - Working with structured data from LLMs
# - Handling models with and without native JSON support
# - Using fallbacks with JSON mode requirements
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages OneLLM's support for:
# - ChatCompletion API with response_format parameter
# - Automatic JSON mode handling across providers
# - Fallback mechanisms with format requirements
# - JSON validation and parsing
#
# RELATED EXAMPLES:
# ----------------
# - chat_completion_example.py: Basic text interactions without JSON mode
# - fallback_example.py: Using fallback models for reliability
# - vision_example.py: Multi-modal capabilities with structured responses
#
# REQUIREMENTS:
# ------------
# - OneLLM
# - OpenAI API key (for models with native JSON support)
# - Other provider API keys (optional, for testing fallback behavior)
#
# EXPECTED OUTPUT:
# ---------------
# 1. JSON response from an OpenAI model with native JSON support
# 2. Optional example with non-OpenAI provider (commented out)
# 3. JSON response with fallback model configuration
# ============================================================================ #
"""

import os
import json
import onellm

# Set API key from environment
onellm.openai_api_key = os.environ.get("OPENAI_API_KEY")


def main():
    """
    Main function to demonstrate JSON mode capabilities in OneLLM.

    This function runs three examples:
    1. Using an OpenAI model with native JSON mode support
    2. (Commented out) Using a provider without native JSON mode support
    3. Using JSON mode with fallback models

    Each example demonstrates different aspects of the JSON mode functionality.
    """
    # Example 1: Demonstrate JSON mode with a supporting model
    demonstrate_json_mode_with_openai()

    print("\n" + "-" * 50 + "\n")

    # Example 2: Demonstrate fallback behavior (commented out)
    # This requires setting up other providers' API keys
    demonstrate_json_mode_with_other_provider_commented()

    print("\n" + "-" * 50 + "\n")

    # Example 3: Demonstrate JSON mode with fallback models
    demonstrate_json_mode_with_fallbacks()

    print("\n" + "Done!" + "\n")


def demonstrate_json_mode_with_openai():
    """
    Demonstrates using JSON mode with an OpenAI model that natively supports it.

    This function:
    1. Makes a request to GPT-4o with JSON mode enabled
    2. Prints the raw response
    3. Validates the JSON by parsing it
    4. Prints the parsed JSON with formatting
    """
    print("Example 1: Using OpenAI model with JSON mode")
    print("-" * 50)

    # Make the API call with JSON mode enabled
    response = onellm.ChatCompletion.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a list of 3 todo items as JSON"},
        ],
        response_format={"type": "json_object"},  # This parameter enables JSON mode
    )

    # Print the raw response content
    print("JSON response from OpenAI:")
    print(response.choices[0].message["content"])
    print()

    # Validate the JSON by attempting to parse it
    try:
        parsed = json.loads(response.choices[0].message["content"])
        print("Successfully parsed JSON response:")
        print(json.dumps(parsed, indent=2))  # Pretty-print the JSON
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")


def demonstrate_json_mode_with_other_provider_commented():
    """
    Contains commented-out code that demonstrates JSON mode with non-OpenAI providers.

    This function is not executed but serves as a template for users who want to
    test JSON mode with other providers like Anthropic. The code shows:
    1. How to set up API keys for other providers
    2. How to make a request with JSON mode
    3. How to handle and validate the response
    """
    # The entire function is commented out as it requires additional API keys
    """
    print("Example 2: Using a provider that doesn't support JSON mode natively")
    print("-" * 50)

    # Set up your API key for other providers
    # onellm.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        response = onellm.ChatCompletion.create(
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


def demonstrate_json_mode_with_fallbacks():
    """
    Demonstrates using JSON mode with fallback models.

    This function shows how to:
    1. Configure a primary model with fallback options
    2. Request JSON responses with this configuration
    3. Handle the response from whichever model was used

    This is useful for ensuring reliability in production systems.
    """
    print("Example 3: Using JSON mode with fallback models")
    print("-" * 50)

    # Make API call with primary model and fallback options
    response = onellm.ChatCompletion.create(
        model="openai/gpt-4o",  # Primary model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a list of 3 todo items as JSON"},
        ],
        response_format={"type": "json_object"},
        fallback_models=["openai/gpt-3.5-turbo"],  # Model to use if primary fails
    )

    # Print the response (could be from primary or fallback model)
    print("JSON response with fallback models:")
    print(response.choices[0].message["content"])


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()

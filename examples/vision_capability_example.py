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
# OneLLM EXAMPLE: Vision Capability Testing
# ============================================================================ #
#
# This example demonstrates multi-modal capabilities of OneLLM and how it handles
# different models' support for image inputs.
# Key features demonstrated:
#
# - Using models with built-in vision support
# - Graceful degradation for text-only models
# - Fallback behavior between vision and non-vision models
# - Structured message format for multi-modal inputs
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages OneLLM's support for:
# - Multi-modal inputs in the ChatCompletion API
# - Automatic content filtering based on model capabilities
# - Fallback mechanisms across different model types
# - Provider-specific model feature detection
#
# RELATED EXAMPLES:
# ----------------
# - vision_example.py: In-depth vision model usage
# - fallback_example.py: General fallback mechanisms
# - chat_completion_example.py: Text-only model interactions
# - image_generation_example.py: Creating images (inverse operation)
#
# REQUIREMENTS:
# ------------
# - OneLLM
# - OpenAI API key with access to vision-capable models
#
# EXPECTED OUTPUT:
# ---------------
# Three different responses demonstrating:
# 1. Vision-capable model describing an image
# 2. Text-only model response with image content removed
# 3. Fallback from vision to non-vision model with appropriate handling
# ============================================================================ #
"""

import os
import onellm

# Set API keys from environment
onellm.openai_api_key = os.environ.get("OPENAI_API_KEY")


def main():
    """
    Main function to demonstrate vision capabilities in OneLLM.

    This function runs three examples:
    1. Using a model with native vision support
    2. Using a text-only model with automatic image content removal
    3. Demonstrating fallback from vision to non-vision models

    Each example shows different aspects of handling multi-modal inputs.
    """
    # Example 1: Using a model with vision support
    demonstrate_vision_model()

    # Example 2: Using a non-vision model with fallback
    demonstrate_text_only_model()

    # Example 3: Fallback handling
    demonstrate_vision_fallback()


def demonstrate_vision_model():
    """
    Demonstrates using a model with native vision capabilities.

    This function:
    1. Creates a message with both text and image content
    2. Sends the message to GPT-4o which supports vision
    3. Prints the model's response describing the image
    """
    print("Example 1: Using a model with vision support")
    print("-" * 50)

    # Create a message with text and image content
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Sample image URL of a nature boardwalk
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
    ]

    # Call a model with vision support (GPT-4o)
    response = onellm.ChatCompletion.create(
        model="openai/gpt-4o",  # Using GPT-4o which has vision capabilities
        messages=messages
    )

    print("Vision-capable model response:")
    print(response.choices[0].message["content"])
    print("\n")


def demonstrate_text_only_model():
    """
    Demonstrates graceful degradation when using a text-only model with image content.

    This function:
    1. Uses the same multi-modal message with text and image
    2. Sends it to GPT-3.5-turbo which doesn't support vision
    3. Shows how OneLLM automatically removes image content and warns the user
    """
    print("Example 2: Using a non-vision model with fallback")
    print("-" * 50)

    # Create a message with text and image content (same as in Example 1)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
    ]

    # Using a regular text model (without vision), we'll get a warning
    # and the image content will be removed automatically
    response = onellm.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",  # This doesn't support vision
        messages=messages
    )

    print("Text-only model response (image content removed):")
    print(response.choices[0].message["content"])
    print("\n")


def demonstrate_vision_fallback():
    """
    Demonstrates fallback from a vision model to a text-only model.

    This function:
    1. Attempts to use a vision-capable model first
    2. Configures a text-only model as fallback
    3. Shows how OneLLM handles the transition if the vision model fails

    This is useful for ensuring reliability in production systems.
    """
    print("Example 3: Fallback from vision to non-vision")
    print("-" * 50)

    # Create a message with text and image content (same as in previous examples)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
                               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
    ]

    # Try a vision model with fallback to a non-vision model
    response = onellm.ChatCompletion.create(
        model="openai/gpt-4-vision-preview",  # Vision model as primary choice
        fallback_models=["openai/gpt-3.5-turbo"],  # Non-vision fallback if primary fails
        messages=messages
    )

    print("Response (will use vision if available, fallback if needed):")
    print(response.choices[0].message["content"])


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()

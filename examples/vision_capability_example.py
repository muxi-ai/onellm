#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the multi-modal capabilities of muxi-llm.

This example shows how to use image inputs with OpenAI's vision models and
demonstrates graceful degradation when using providers without vision support.
"""

import os
import muxi_llm

# Set API keys from environment
muxi_llm.openai_api_key = os.environ.get("OPENAI_API_KEY")

# Example 1: Using a model with vision support
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
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                           "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
                           "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]
    }
]

# Call a model with vision support (GPT-4o)
response = muxi_llm.ChatCompletion.create(
    model="openai/gpt-4o",
    messages=messages
)

print("Vision-capable model response:")
print(response.choices[0].message["content"])
print("\n")

# Example 2: Using a non-vision model with fallback
print("Example 2: Using a non-vision model with fallback")
print("-" * 50)

# Using a regular text model (without vision), we'll get a warning
# and the image content will be removed automatically
response = muxi_llm.ChatCompletion.create(
    model="openai/gpt-3.5-turbo",  # This doesn't support vision
    messages=messages
)

print("Text-only model response (image content removed):")
print(response.choices[0].message["content"])
print("\n")

# Example 3: Fallback handling
print("Example 3: Fallback from vision to non-vision")
print("-" * 50)

# Try a vision model with fallback to a non-vision model
response = muxi_llm.ChatCompletion.create(
    model="openai/gpt-4-vision-preview",  # Vision model
    fallback_models=["openai/gpt-3.5-turbo"],  # Non-vision fallback
    messages=messages
)

print("Response (will use vision if available, fallback if needed):")
print(response.choices[0].message["content"])

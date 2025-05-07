#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
# ============================================================================ #
# MUXI-LLM EXAMPLE: Audio Translation
# ============================================================================ #
#
# This example demonstrates how to use muxi-llm to translate speech in foreign
# language audio files to English text.
# Key features demonstrated:
#
# - Translating foreign language audio to English text
# - Working with audio file inputs
# - Configuring translation options with prompts
# - Handling different response formats
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages muxi-llm's support for:
# - AudioTranslation API
# - Synchronous API interface
# - Provider-specific models (Whisper)
# - Response format configuration
#
# RELATED EXAMPLES:
# ----------------
# - audio_transcription_example.py: Transcribing audio to text in its original language
# - text_to_speech_example.py: Converting text to speech audio
# - chat_completion_example.py: Basic text interactions with LLMs
#
# REQUIREMENTS:
# ------------
# - muxi-llm
# - Audio file in a foreign language (examples/audio/foreign_language.mp3)
# - OpenAI API key with access to Whisper models
#
# EXPECTED OUTPUT:
# ---------------
# The translated English text from the audio file, displayed in two formats:
# 1. Basic translation with minimal parameters
# 2. Translation with additional options like context prompts
# ============================================================================ #
"""

import os
import sys
from pathlib import Path

from muxi_llm import AudioTranslation
from muxi_llm.config import set_api_key


def main():
    """
    Run the audio translation example.

    This function demonstrates how to use the AudioTranslation API to convert
    speech in a foreign language audio file to English text. It shows both
    basic usage and advanced options with additional parameters.

    The function performs the following steps:
    1. Sets up the OpenAI API key from environment variables
    2. Locates and validates the audio file
    3. Performs a basic translation to English
    4. Performs a translation with additional options like prompt and response format

    Returns:
        None
    """
    # Set up API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Exit if the API key is not available in environment variables
        print("Error: OPENAI_API_KEY environment variable is required for this example")
        sys.exit(1)

    # Configure the API key for the OpenAI provider
    set_api_key(api_key, "openai")

    # Path to the audio file
    # This example expects a file at "examples/audio/foreign_language.mp3"
    # You can replace this with your own audio file path
    examples_dir = Path(__file__).parent  # Get the directory containing this script
    audio_file = examples_dir / "audio" / "foreign_language.mp3"  # Construct path to audio file

    # Validate that the audio file exists before proceeding
    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file}")
        print("Please provide a valid audio file path or create the examples/audio directory")
        print("with a foreign_language.mp3 file for this example.")
        sys.exit(1)

    print(f"Translating audio file to English: {audio_file}")

    # Basic translation - simplest form with minimal parameters
    # Just specify the file and model to use
    result = AudioTranslation.create_sync(
        file=str(audio_file),
        model="openai/whisper-1"
    )

    print("\n--- Basic Translation to English ---")
    print(result["text"])

    # Translation with more options - demonstrating additional parameters
    # Adding a prompt can help guide the translation for better context understanding
    # Specifying response_format as "text" returns a simple text string instead of a dictionary
    result_with_options = AudioTranslation.create_sync(
        file=str(audio_file),
        model="openai/whisper-1",
        prompt="This is a conversation about technology",  # Guide the translation
        response_format="text"  # Get simple text response
    )

    print("\n--- Translation with Options ---")
    print(result_with_options)

    # Note: Unlike transcription, translation always outputs English text
    # regardless of the source language in the audio file.


if __name__ == "__main__":
    main()

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
# OneLLM EXAMPLE: Audio Transcription
# ============================================================================ #
#
# This example demonstrates how to use OneLLM to transcribe speech in audio files to text.
# Key features demonstrated:
#
# - Converting audio files to text using AI transcription
# - Working with audio file inputs
# - Configuring transcription options like language and prompts
# - Handling different response formats
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages OneLLM's support for:
# - AudioTranscription API
# - Synchronous API interface
# - Provider-specific models (Whisper)
# - Response format configuration
#
# RELATED EXAMPLES:
# ----------------
# - audio_translation_example.py: Translating foreign language audio to English
# - text_to_speech_example.py: Converting text to speech audio
# - chat_completion_example.py: Basic text interactions with LLMs
#
# REQUIREMENTS:
# ------------
# - OneLLM
# - Audio file to transcribe (examples/audio/sample.mp3)
# - OpenAI API key with access to Whisper models
#
# EXPECTED OUTPUT:
# ---------------
# The transcribed text from the audio file, displayed in two formats:
# 1. Basic transcription with minimal parameters
# 2. Transcription with additional options like language specification
# ============================================================================ #
"""

import os
import sys
from pathlib import Path

from onellm import AudioTranscription
from onellm.config import set_api_key


def main():
    """
    Run the audio transcription example.

    This function demonstrates how to use the AudioTranscription API to convert
    speech in an audio file to text. It shows both basic usage and advanced options.

    The function performs the following steps:
    1. Sets up the OpenAI API key from environment variables
    2. Locates and validates the audio file
    3. Performs a basic transcription
    4. Performs a transcription with additional options

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
    # This example expects a file at "examples/audio/sample.mp3"
    # You can replace this with your own audio file path
    examples_dir = Path(__file__).parent  # Get the directory containing this script
    audio_file = examples_dir / "audio" / "sample.mp3"  # Construct path to audio file

    # Validate that the audio file exists before proceeding
    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file}")
        print("Please provide a valid audio file path or create the examples/audio directory")
        print("with a sample.mp3 file for this example.")
        sys.exit(1)

    print(f"Transcribing audio file: {audio_file}")

    # Basic transcription - minimal parameters
    # Using the synchronous API to wait for results
    result = AudioTranscription.create_sync(
        file=str(audio_file),  # Convert Path to string as required by the API
        model="openai/whisper-1"  # Specify the model in provider/model format
    )

    print("\n--- Basic Transcription ---")
    print(result["text"])  # Access the transcribed text from the result dictionary

    # Transcription with more options - demonstrating additional parameters
    result_with_options = AudioTranscription.create_sync(
        file=str(audio_file),
        model="openai/whisper-1",
        language="en",  # Optionally specify language for better accuracy
        prompt="This is a technical discussion",  # Guide the transcription with context
        response_format="text"  # Get simple text response instead of JSON
    )

    print("\n--- Transcription with Options ---")
    print(result_with_options)  # With response_format="text", this is directly the transcribed text


if __name__ == "__main__":
    main()

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
# MUXI-LLM EXAMPLE: Text-to-Speech (TTS)
# ============================================================================ #
#
# This example demonstrates how to use OneLLM to convert text to speech using
# AI voice synthesis models.
# Key features demonstrated:
#
# - Converting text to spoken audio
# - Selecting different voice options
# - Configuring audio format and quality
# - Saving and managing audio output files
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages OneLLM's support for:
# - Speech API for text-to-speech conversion
# - Provider-specific TTS models
# - Audio file format handling
# - Command-line argument processing
#
# RELATED EXAMPLES:
# ----------------
# - audio_transcription_example.py: Converting audio to text
# - audio_translation_example.py: Translating foreign language audio
# - chat_completion_example.py: Text-based interactions with LLMs
#
# REQUIREMENTS:
# ------------
# - OneLLM
# - OpenAI API key with access to TTS models
# - Writeable file system for saving audio files
#
# EXPECTED OUTPUT:
# ---------------
# 1. A success message confirming the TTS generation
# 2. Information about the output audio file location
# 3. The size of the generated audio in bytes
# 4. The actual audio file saved to disk that can be played
# ============================================================================ #
"""

import os
import asyncio
import argparse
from datetime import datetime

# Import the Speech class from OneLLM
from onellm import Speech


async def text_to_speech_example(text, voice, output_file=None, model="tts-1"):
    """
    Generate speech from text and save it to a file.

    This function uses the OneLLM Speech API to convert the provided text into
    speech audio using the specified voice and model. The generated audio is saved
    to the specified output file or an auto-generated file name.

    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (e.g., "alloy", "echo", "fable", etc.)
        output_file (str, optional): Path to save the audio file. If None, an auto-generated
                                    filename will be used. Defaults to None.
        model (str, optional): The TTS model to use. Defaults to "tts-1".

    Returns:
        str or None: The path to the saved audio file if successful, None if an error occurs.
    """
    try:
        # Generate a default output file name if not provided
        if not output_file:
            # Create a timestamp-based filename to avoid overwriting previous outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"speech_output_{voice}_{timestamp}.mp3"

        # Create the speech audio
        print(f"Converting text to speech using voice '{voice}'...")
        print(f"Text: '{text}'")

        # Call the Speech.create method from OneLLM to generate the audio
        # The model name is prefixed with "openai/" to specify the provider
        audio_data = await Speech.create(
            input=text,
            model=f"openai/{model}",
            voice=voice,
            output_file=output_file,
            response_format="mp3",  # Available formats: mp3, opus, aac, or flac
            speed=1.0  # Speed factor ranges from 0.25 to 4.0
        )

        # Print success message and file information
        print("Speech generated successfully!")
        print(f"Audio saved to: {output_file}")
        print(f"Audio size: {len(audio_data)} bytes")

        return output_file

    except Exception as e:
        # Handle any errors that occur during speech generation
        print(f"Error generating speech: {e}")
        return None


def main():
    """
    Parse command line arguments and run the text-to-speech example.

    This function sets up the argument parser to allow customization of the
    text-to-speech conversion parameters, checks for the required API key,
    and runs the example with the provided or default parameters.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Text-to-Speech Example")
    parser.add_argument("--text", default="Hello, this is a test of the OpenAI text-to-speech API.",
                        help="Text to convert to speech")
    parser.add_argument("--voice", default="alloy",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="Voice to use")
    parser.add_argument("--model", default="tts-1",
                        choices=["tts-1", "tts-1-hd"],
                        help="TTS model to use")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: auto-generated)")

    # Parse the command line arguments
    args = parser.parse_args()

    # Check for API key in environment variables
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your-api-key")
        return

    # Run the example with the provided or default arguments
    asyncio.run(text_to_speech_example(
        args.text,
        args.voice,
        args.output,
        args.model
    ))


if __name__ == "__main__":
    main()

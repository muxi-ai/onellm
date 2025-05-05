"""
Text-to-Speech (TTS) Example

This example demonstrates how to use the muxi-llm library to convert text to speech
using OpenAI's text-to-speech models.

Requirements:
- muxi-llm
- An OpenAI API key with access to TTS models

Usage:
- Set your OpenAI API key as an environment variable: export OPENAI_API_KEY=your-api-key
- Run this script: python text_to_speech_example.py
"""

import os
import asyncio
import argparse
from datetime import datetime

# Import the Speech class from muxi-llm
from muxi.llm import Speech


async def text_to_speech_example(text, voice, output_file=None, model="tts-1"):
    """Generate speech from text and save it to a file."""
    try:
        # Generate a default output file name if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"speech_output_{voice}_{timestamp}.mp3"

        # Create the speech audio
        print(f"Converting text to speech using voice '{voice}'...")
        print(f"Text: '{text}'")

        audio_data = await Speech.create(
            input=text,
            model=f"openai/{model}",
            voice=voice,
            output_file=output_file,
            response_format="mp3",  # mp3, opus, aac, or flac
            speed=1.0  # 0.25 to 4.0
        )

        print("Speech generated successfully!")
        print(f"Audio saved to: {output_file}")
        print(f"Audio size: {len(audio_data)} bytes")

        return output_file

    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


def main():
    """Parse arguments and run the example."""
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

    args = parser.parse_args()

    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your-api-key")
        return

    # Run the example
    asyncio.run(text_to_speech_example(
        args.text,
        args.voice,
        args.output,
        args.model
    ))


if __name__ == "__main__":
    main()

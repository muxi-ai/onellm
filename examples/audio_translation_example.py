"""
Example demonstrating the use of the AudioTranslation API.

This example shows how to translate an audio file in a foreign language
to English text using OpenAI's Whisper model.
"""

import os
import sys
from pathlib import Path

from muxi_llm import AudioTranslation
from muxi_llm.config import set_api_key


def main():
    """Run the audio translation example."""
    # Set up API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required for this example")
        sys.exit(1)

    set_api_key(api_key, "openai")

    # Path to the audio file
    # This example expects a file at "examples/audio/foreign_language.mp3"
    # You can replace this with your own audio file path
    examples_dir = Path(__file__).parent
    audio_file = examples_dir / "audio" / "foreign_language.mp3"

    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file}")
        print("Please provide a valid audio file path or create the examples/audio directory")
        print("with a foreign_language.mp3 file for this example.")
        sys.exit(1)

    print(f"Translating audio file to English: {audio_file}")

    # Basic translation
    result = AudioTranslation.create_sync(
        file=str(audio_file),
        model="openai/whisper-1"
    )

    print("\n--- Basic Translation to English ---")
    print(result["text"])

    # Translation with more options
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

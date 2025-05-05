"""
Example demonstrating the use of the AudioTranscription API.

This example shows how to transcribe an audio file to text using
OpenAI's Whisper model.
"""

import os
import sys
from pathlib import Path

from muxi_llm import AudioTranscription
from muxi_llm.config import set_api_key


def main():
    """Run the audio transcription example."""
    # Set up API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required for this example")
        sys.exit(1)

    set_api_key(api_key, "openai")

    # Path to the audio file
    # This example expects a file at "examples/audio/sample.mp3"
    # You can replace this with your own audio file path
    examples_dir = Path(__file__).parent
    audio_file = examples_dir / "audio" / "sample.mp3"

    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file}")
        print("Please provide a valid audio file path or create the examples/audio directory")
        print("with a sample.mp3 file for this example.")
        sys.exit(1)

    print(f"Transcribing audio file: {audio_file}")

    # Basic transcription
    result = AudioTranscription.create_sync(
        file=str(audio_file),
        model="openai/whisper-1"
    )

    print("\n--- Basic Transcription ---")
    print(result["text"])

    # Transcription with more options
    result_with_options = AudioTranscription.create_sync(
        file=str(audio_file),
        model="openai/whisper-1",
        language="en",  # Optionally specify language
        prompt="This is a technical discussion",  # Guide the transcription
        response_format="text"  # Get simple text response
    )

    print("\n--- Transcription with Options ---")
    print(result_with_options)


if __name__ == "__main__":
    main()

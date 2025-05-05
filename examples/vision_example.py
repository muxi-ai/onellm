"""
Example demonstrating the use of vision capabilities in ChatCompletion.

This example shows how to use the ChatCompletion class to process images
along with text using OpenAI's vision models.
"""

import os
from typing import List, Dict, Any

from muxi.llm import ChatCompletion
from muxi.llm.config import set_api_key


def create_message_with_image(image_url: str) -> List[Dict[str, Any]]:
    """
    Create a message that includes both text and an image.

    Args:
        image_url: URL of the image to process

    Returns:
        List of messages for the conversation
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that can see and understand images."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Please describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"  # Options: auto, low, high
                    }
                }
            ]
        }
    ]


def main():
    """Run the vision example."""
    # Set up API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for this example"
        )

    set_api_key(api_key, "openai")

    # Sample image URL - replace with your own if needed
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )

    # Create messages with image
    messages = create_message_with_image(image_url)

    print("Sending request with an image to OpenAI's vision model...\n")

    # Call OpenAI with vision capabilities
    response = ChatCompletion.create(
        model="openai/gpt-4-vision-preview",  # Use a vision-capable model
        messages=messages,
        max_tokens=300
    )

    # Print the response
    print("--- Vision Model Response ---")
    print(response.choices[0].message["content"])

    # Example of asking a follow-up question about the same image
    follow_up_messages = messages.copy()
    follow_up_messages.append({
        "role": "assistant",
        "content": response.choices[0].message["content"]
    })
    follow_up_messages.append({
        "role": "user",
        "content": "What season does this image appear to be from?"
    })

    print("\nSending follow-up question...\n")

    follow_up_response = ChatCompletion.create(
        model="openai/gpt-4-vision-preview",
        messages=follow_up_messages,
        max_tokens=100
    )

    print("--- Follow-up Response ---")
    print(follow_up_response.choices[0].message["content"])

    if response.usage:
        print(f"\nToken usage: {response.usage}")


if __name__ == "__main__":
    main()

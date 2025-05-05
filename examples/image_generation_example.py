"""
Image Generation Example

This example demonstrates how to use the muxi-llm library to generate images
using OpenAI's DALL-E models.

Requirements:
- muxi-llm
- An OpenAI API key with access to DALL-E models

Usage:
- Set your OpenAI API key as an environment variable: export OPENAI_API_KEY=your-api-key
- Run this script: python image_generation_example.py
"""

import os
import asyncio
import argparse

# Import the Image class from muxi-llm
from muxi_llm import Image


async def generate_image_example(prompt, model="dall-e-3", size="1024x1024", output_dir=None):
    """Generate an image from a text prompt and save it to a file."""
    try:
        # Generate a default output directory if not provided
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "generated_images")
            os.makedirs(output_dir, exist_ok=True)

        print(f"Generating image with prompt: '{prompt}'")
        print(f"Using model: {model}, size: {size}")

        # Generate the image
        result = await Image.create(
            prompt=prompt,
            model=f"openai/{model}",
            size=size,
            output_dir=output_dir
        )

        # Print information about the generated image
        print("\nImage generated successfully!")

        # If the model is DALL-E 3, it provides a revised prompt
        if "revised_prompt" in result["data"][0]:
            print(f"\nRevised prompt: {result['data'][0]['revised_prompt']}")

        # Print the file path
        print(f"\nImage saved to: {result['data'][0]['filepath']}")

        return result
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with DALL-E using muxi-llm")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="A photorealistic image of a cute robot exploring a futuristic city",
        help="Text prompt describing the image to generate"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="dall-e-3",
        choices=["dall-e-3", "dall-e-2"],
        help="DALL-E model to use"
    )
    parser.add_argument(
        "--size", "-s",
        type=str,
        default="1024x1024",
        help="Size of the generated image (model-dependent)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Directory to save the generated image"
    )
    return parser.parse_args()


def main():
    """Run the image generation example."""
    args = parse_arguments()

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your-api-key")
        return

    # Run the example
    asyncio.run(generate_image_example(
        prompt=args.prompt,
        model=args.model,
        size=args.size,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()

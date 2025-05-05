"""
OpenAI image generation capabilities.

This module provides a high-level API for OpenAI's image generation capabilities.
"""

import asyncio
import os
from typing import Dict

from .providers import get_provider
from .utils.model import parse_model_name


class Image:
    """API class for image generation."""

    @classmethod
    async def create(
        cls,
        prompt: str,
        model: str = "openai/dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        **kwargs
    ) -> Dict:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image
            model: Model ID in format "provider/model" (default: "openai/dall-e-3")
            n: Number of images to generate (default: 1)
            size: Size of the generated images (default: "1024x1024")
            **kwargs: Additional parameters:
                - quality: Quality of the image ("standard" or "hd"), for DALL-E 3
                - style: Style of image ("natural" or "vivid"), for DALL-E 3
                - response_format: Format of the response ("url" or "b64_json")
                - user: End-user ID for tracking
                - output_dir: Optional path to save the generated images
                - output_format: Optional format for output files ("png", "jpg", etc.)

        Returns:
            Dict with generated images data
        """
        # Extract kwargs that are for our logic, not the API
        output_dir = kwargs.pop("output_dir", None)
        output_format = kwargs.pop("output_format", "png")

        # Get provider and model name
        provider_name, model_name = parse_model_name(model)
        provider = get_provider(provider_name)

        # Generate image
        result = await provider.create_image(
            prompt, model_name, n=n, size=size, **kwargs
        )

        # Save images if output_dir is provided
        if output_dir and result.get("data"):
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(result.get("created", asyncio.get_event_loop().time()))

            for i, img_data in enumerate(result.get("data", [])):
                # Get image data (url or base64)
                if "url" in img_data:
                    # For URL responses, we'll need to download the image
                    image_url = img_data["url"]
                    image_bytes = await cls._download_image(image_url)
                elif "b64_json" in img_data:
                    # For base64 responses, decode the data
                    import base64
                    image_bytes = base64.b64decode(img_data["b64_json"])
                else:
                    continue  # Skip if no image data

                # Create filename
                filename = f"image_{timestamp}_{i}.{output_format}"
                filepath = os.path.join(output_dir, filename)

                # Save the image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                # Add the file path to the result
                img_data["filepath"] = filepath

        return result

    @classmethod
    def create_sync(cls, *args, **kwargs) -> Dict:
        """
        Synchronous version of create().

        Args:
            Same arguments as create()

        Returns:
            Dict with generated images data
        """
        return asyncio.run(cls.create(*args, **kwargs))

    @classmethod
    async def _download_image(cls, url: str) -> bytes:
        """
        Download an image from a URL.

        Args:
            url: URL of the image

        Returns:
            Image data as bytes
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download image: {response.status}")
                return await response.read()

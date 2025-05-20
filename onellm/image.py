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
Image generation capabilities.

This module provides a high-level API for image generation across multiple providers.
It supports various models and offers both synchronous and asynchronous interfaces
with fallback options.
"""

import asyncio
import os
from typing import Dict, List, Optional

from .providers.base import get_provider_with_fallbacks
from .utils.fallback import FallbackConfig


class Image:
    """API class for image generation."""

    @classmethod
    async def create(
        cls,
        prompt: str,
        model: str = "openai/dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate images from a text prompt.

        This method handles the asynchronous generation of images based on a text prompt.
        It supports fallback models if the primary model fails and can save generated
        images to disk if an output directory is specified.

        Args:
            prompt: Text description of the desired image
            model: Model ID in format "provider/model" (default: "openai/dall-e-3")
            n: Number of images to generate (default: 1)
            size: Size of the generated images (default: "1024x1024")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
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
        # These parameters are used locally and should not be passed to the provider API
        output_dir = kwargs.pop("output_dir", None)
        output_format = kwargs.pop("output_format", "png")

        # Process fallback configuration
        # Convert the dictionary to a FallbackConfig object if provided
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        # This splits the model string into provider and model name parts
        # and handles fallback logic if specified
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config,
        )

        # Generate image using the selected provider
        result = await provider.create_image(
            prompt, model_name, n=n, size=size, **kwargs
        )

        # Save images if output_dir is provided
        if output_dir and result.get("data"):
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Use the creation timestamp from the result or current time as fallback
            timestamp = int(result.get("created", asyncio.get_event_loop().time()))

            # Process each generated image
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

                # Create filename with timestamp and index to ensure uniqueness
                filename = f"image_{timestamp}_{i}.{output_format}"
                filepath = os.path.join(output_dir, filename)

                # Save the image to disk
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                # Add the file path to the result for reference
                img_data["filepath"] = filepath

        return result

    @classmethod
    def create_sync(
        cls,
        prompt: str,
        model: str = "openai/dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Synchronous version of create().

        This method provides a synchronous interface to the asynchronous create() method
        by running it in an event loop. It has the same functionality but can be used
        in synchronous contexts.

        Args:
            prompt: Text description of the desired image
            model: Model ID in format "provider/model" (default: "openai/dall-e-3")
            n: Number of images to generate (default: 1)
            size: Size of the generated images (default: "1024x1024")
            fallback_models: Optional list of models to try if the primary model fails
            fallback_config: Optional configuration for fallback behavior
            **kwargs: Additional parameters as in create()

        Returns:
            Dict with generated images data
        """
        # Use asyncio.run to execute the async create method in a synchronous context
        return asyncio.run(
            cls.create(
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                fallback_models=fallback_models,
                fallback_config=fallback_config,
                **kwargs,
            )
        )

    @classmethod
    async def _download_image(cls, url: str) -> bytes:
        """
        Download an image from a URL.

        This helper method asynchronously downloads image data from a given URL.
        It uses aiohttp for efficient async HTTP requests.

        Args:
            url: URL of the image to download

        Returns:
            Image data as bytes

        Raises:
            ValueError: If the download fails (non-200 status code)
        """
        import aiohttp

        # Create a session for HTTP requests
        async with aiohttp.ClientSession() as session:
            # Make the GET request to download the image
            async with session.get(url) as response:
                # Check if the request was successful
                if response.status != 200:
                    raise ValueError(f"Failed to download image: {response.status}")
                # Return the binary image data
                return await response.read()

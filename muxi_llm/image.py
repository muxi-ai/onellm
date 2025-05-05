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
OpenAI image generation capabilities.

This module provides a high-level API for OpenAI's image generation capabilities.
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
        **kwargs
    ) -> Dict:
        """
        Generate images from a text prompt.

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
        output_dir = kwargs.pop("output_dir", None)
        output_format = kwargs.pop("output_format", "png")

        # Process fallback configuration
        fb_config = None
        if fallback_config:
            fb_config = FallbackConfig(**fallback_config)

        # Get provider with fallbacks or a regular provider
        provider, model_name = get_provider_with_fallbacks(
            primary_model=model,
            fallback_models=fallback_models,
            fallback_config=fb_config
        )

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
    def create_sync(
        cls,
        prompt: str,
        model: str = "openai/dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        fallback_models: Optional[List[str]] = None,
        fallback_config: Optional[dict] = None,
        **kwargs
    ) -> Dict:
        """
        Synchronous version of create().

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
        return asyncio.run(cls.create(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            fallback_models=fallback_models,
            fallback_config=fallback_config,
            **kwargs
        ))

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

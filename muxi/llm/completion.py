"""
Text completion functionality for muxi-llm.

This module provides a Completion class that can be used to create text
completions from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import Any, AsyncGenerator, Union

from .providers.base import parse_model_name, get_provider
from .models import CompletionResponse


class Completion:
    """Class for creating text completions with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-davinci-003')
            prompt: Text prompt to complete
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or a streaming generator

        Example:
            >>> response = Completion.create(
            ...     model="openai/text-davinci-003",
            ...     prompt="Once upon a time",
            ...     max_tokens=50
            ... )
            >>> print(response.choices[0].text)
        """
        # Parse model name to get provider and model
        provider_name, model_name = parse_model_name(model)

        # Get provider instance
        provider = get_provider(provider_name)

        # Call the provider's method synchronously
        if stream:
            # For streaming, we need to use async properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                provider.create_completion(
                    prompt=prompt,
                    model=model_name,
                    stream=stream,
                    **kwargs
                )
            )
        else:
            # For non-streaming, we can just run and get the result
            return asyncio.run(
                provider.create_completion(
                    prompt=prompt,
                    model=model_name,
                    stream=stream,
                    **kwargs
                )
            )

    @classmethod
    async def acreate(
        cls,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion asynchronously.

        Args:
            model: Model name with provider prefix (e.g., 'openai/text-davinci-003')
            prompt: Text prompt to complete
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            CompletionResponse or a streaming generator

        Example:
            >>> response = await Completion.acreate(
            ...     model="openai/text-davinci-003",
            ...     prompt="Once upon a time",
            ...     max_tokens=50
            ... )
            >>> print(response.choices[0].text)
        """
        # Parse model name to get provider and model
        provider_name, model_name = parse_model_name(model)

        # Get provider instance
        provider = get_provider(provider_name)

        # Call the provider's method asynchronously
        return await provider.create_completion(
            prompt=prompt,
            model=model_name,
            stream=stream,
            **kwargs
        )

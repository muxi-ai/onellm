"""
Chat completion functionality for muxi-llm.

This module provides a ChatCompletion class that can be used to create chat
completions from various providers in a manner compatible with OpenAI's API.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Union

from .providers.base import parse_model_name, get_provider
from .models import ChatCompletionResponse, ChatCompletionChunk


class ChatCompletion:
    """Class for creating chat completions with various providers."""

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion.

        Args:
            model: Model name with provider prefix (e.g., 'openai/gpt-4')
            messages: List of messages in the conversation
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects

        Example:
            >>> response = ChatCompletion.create(
            ...     model="openai/gpt-4",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Hello, how are you?"}
            ...     ]
            ... )
            >>> print(response.choices[0].message["content"])
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
                provider.create_chat_completion(
                    messages=messages,
                    model=model_name,
                    stream=stream,
                    **kwargs
                )
            )
        else:
            # For non-streaming, we can just run and get the result
            return asyncio.run(
                provider.create_chat_completion(
                    messages=messages,
                    model=model_name,
                    stream=stream,
                    **kwargs
                )
            )

    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion asynchronously.

        Args:
            model: Model name with provider prefix (e.g., 'openai/gpt-4')
            messages: List of messages in the conversation
            stream: Whether to stream the response
            **kwargs: Additional model parameters

        Returns:
            ChatCompletionResponse or a generator yielding ChatCompletionChunk objects

        Example:
            >>> response = await ChatCompletion.acreate(
            ...     model="openai/gpt-4",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Hello, how are you?"}
            ...     ]
            ... )
            >>> print(response.choices[0].message["content"])
        """
        # Parse model name to get provider and model
        provider_name, model_name = parse_model_name(model)

        # Get provider instance
        provider = get_provider(provider_name)

        # Call the provider's method asynchronously
        return await provider.create_chat_completion(
            messages=messages,
            model=model_name,
            stream=stream,
            **kwargs
        )

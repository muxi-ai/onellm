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
Fallback provider proxy implementation.

This module implements a provider proxy that supports fallbacks to alternative models
when the primary model fails.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..errors import APIError, FallbackExhaustionError
from ..models import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    CompletionResponse,
    EmbeddingResponse,
    FileObject,
)
from ..types import Message
from ..utils.fallback import FallbackConfig, maybe_await
from .base import Provider, get_provider, parse_model_name


class FallbackProviderProxy(Provider):
    """Provider implementation that supports fallbacks to alternative models."""

    def __init__(
        self, models: List[str], fallback_config: Optional[FallbackConfig] = None
    ):
        """
        Initialize with a list of models to try.

        Args:
            models: List of models to try in order (including primary model)
            fallback_config: Optional configuration for fallback behavior
        """
        self.models = models
        self.providers: Dict[str, Provider] = {}  # Lazy-loaded providers
        self.fallback_config = fallback_config or FallbackConfig()
        self.logger = logging.getLogger("onellm.fallback")

        # Lazy initialize capability flags - will be set on first access
        self._json_mode_support = None
        self._vision_support = None
        self._audio_input_support = None
        self._video_input_support = None
        self._streaming_support = None
        self._token_by_token_support = None
        self._realtime_support = None

    def _check_provider_capability(self, capability_name: str) -> bool:
        """
        Check if the primary provider supports a specific capability.

        This method lazily loads the primary provider and checks if it supports
        the requested capability.

        Args:
            capability_name: Name of the capability flag to check

        Returns:
            True if the primary provider supports the capability, False otherwise
        """
        if not self.models:
            return False

        # Get the provider for the primary model
        primary_model = self.models[0]
        provider_name, _ = parse_model_name(primary_model)

        # Lazy load the provider if needed
        if provider_name not in self.providers:
            self.providers[provider_name] = get_provider(provider_name)

        # Get the provider instance
        provider = self.providers[provider_name]

        # Check if the capability is supported
        return getattr(provider, capability_name, False)

    @property
    def json_mode_support(self) -> bool:
        """Check if JSON mode is supported by the primary provider."""
        if self._json_mode_support is None:
            # Lazy initialization of the capability flag
            self._json_mode_support = self._check_provider_capability("json_mode_support")
        return self._json_mode_support

    @property
    def vision_support(self) -> bool:
        """Check if vision is supported by the primary provider."""
        if self._vision_support is None:
            # Lazy initialization of the capability flag
            self._vision_support = self._check_provider_capability("vision_support")
        return self._vision_support

    @property
    def audio_input_support(self) -> bool:
        """Check if audio input is supported by the primary provider."""
        if self._audio_input_support is None:
            # Lazy initialization of the capability flag
            self._audio_input_support = self._check_provider_capability("audio_input_support")
        return self._audio_input_support

    @property
    def video_input_support(self) -> bool:
        """Check if video input is supported by the primary provider."""
        if self._video_input_support is None:
            # Lazy initialization of the capability flag
            self._video_input_support = self._check_provider_capability("video_input_support")
        return self._video_input_support

    @property
    def streaming_support(self) -> bool:
        """Check if streaming is supported by the primary provider."""
        if self._streaming_support is None:
            # Lazy initialization of the capability flag
            self._streaming_support = self._check_provider_capability("streaming_support")
        return self._streaming_support

    @property
    def token_by_token_support(self) -> bool:
        """Check if token-by-token streaming is supported by the primary provider."""
        if self._token_by_token_support is None:
            # Lazy initialization of the capability flag
            self._token_by_token_support = self._check_provider_capability("token_by_token_support")
        return self._token_by_token_support

    @property
    def realtime_support(self) -> bool:
        """Check if realtime API is supported by the primary provider."""
        if self._realtime_support is None:
            # Lazy initialization of the capability flag
            self._realtime_support = self._check_provider_capability("realtime_support")
        return self._realtime_support

    async def _try_with_fallbacks(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Try a provider method with fallbacks.

        This method attempts to call the specified method on each provider in sequence,
        starting with the primary model and falling back to alternatives if errors occur.

        Args:
            method_name: Name of the provider method to call
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result of the successful method call

        Raises:
            FallbackExhaustionError: If all models fail
            AttributeError: If method is missing on all providers
        """
        last_error = None
        models_tried = []
        attribute_errors = 0

        # Limit the number of fallbacks if max_fallbacks is set
        models_to_try = self.models
        if self.fallback_config.max_fallbacks is not None:
            # Only try up to max_fallbacks+1 models (primary + fallbacks)
            models_to_try = self.models[: self.fallback_config.max_fallbacks + 1]

        # Try each model in sequence
        for model_string in models_to_try:
            provider_name, model_name = parse_model_name(model_string)
            models_tried.append(model_string)

            # Get or create provider instance
            if provider_name not in self.providers:
                # Lazy load the provider
                self.providers[provider_name] = get_provider(provider_name)

            provider = self.providers[provider_name]

            try:
                # Get the provider method
                method = getattr(provider, method_name)

                # Call the method with the appropriate model
                kwargs_with_model = {**kwargs, "model": model_name}
                result = await method(*args, **kwargs_with_model)

                # Log fallback usage if not the primary model
                if (
                    model_string != self.models[0]
                    and self.fallback_config.log_fallbacks
                ):
                    self.logger.info(
                        f"Fallback succeeded: Using {model_string} instead of {self.models[0]}"
                    )

                # Call the callback if provided and we're using a fallback model
                if (
                    model_string != self.models[0]
                    and self.fallback_config.fallback_callback
                ):
                    await maybe_await(
                        self.fallback_config.fallback_callback(
                            primary_model=self.models[0],
                            fallback_model=model_string,
                            error=last_error,
                        )
                    )

                return result

            except AttributeError as e:
                # Method not implemented on this provider
                attribute_errors += 1
                last_error = e
                # Continue to next provider without logging as retriable
                continue

            except Exception as e:
                last_error = e

                # Log the failure
                if self.fallback_config.log_fallbacks:
                    self.logger.warning(f"Model {model_string} failed: {str(e)}")

                # Determine if this error should trigger a fallback
                retriable = any(
                    isinstance(e, err_type)
                    for err_type in self.fallback_config.retriable_errors
                )
                if not retriable:
                    # Non-retriable error - raise immediately
                    raise

                # Continue to next fallback

        # If attribute errors on all providers, that means the method wasn't supported
        if attribute_errors == len(models_tried) and last_error:
            raise last_error  # Re-raise the AttributeError

        # If we get here, all models failed
        if last_error:
            # Use the correct fallback_models list based on max_fallbacks
            fallback_models = self.models[1:]
            if self.fallback_config.max_fallbacks is not None:
                fallback_models = self.models[1:self.fallback_config.max_fallbacks + 1]

            raise FallbackExhaustionError(
                message=f"All models failed: {str(last_error)}",
                primary_model=self.models[0],
                fallback_models=fallback_models,
                models_tried=models_tried,
                original_error=last_error,
            )

        # Should never reach here, but just in case
        raise APIError(
            f"All models failed but no error was recorded. Models tried: {models_tried}"
        )

    async def _try_streaming_with_fallbacks(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Try a streaming method with fallbacks.

        This method attempts to call the specified streaming method on each provider in sequence,
        starting with the primary model and falling back to alternatives if errors occur.

        Args:
            method_name: Name of the provider method to call
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            AsyncGenerator from the first successful provider

        Raises:
            FallbackExhaustionError: If all models fail
        """
        last_error = None
        models_tried = []

        # Limit the number of fallbacks if max_fallbacks is set
        models_to_try = self.models
        if self.fallback_config.max_fallbacks is not None:
            # Only try up to max_fallbacks+1 models (primary + fallbacks)
            models_to_try = self.models[: self.fallback_config.max_fallbacks + 1]

        # Try each model in sequence
        for model_string in models_to_try:
            provider_name, model_name = parse_model_name(model_string)
            models_tried.append(model_string)

            # Get or create provider instance
            if provider_name not in self.providers:
                # Lazy load the provider
                self.providers[provider_name] = get_provider(provider_name)

            provider = self.providers[provider_name]

            try:
                # Get the provider method
                method = getattr(provider, method_name)

                # Call the method with the appropriate model to get the generator
                kwargs_with_model = {**kwargs, "model": model_name}
                generator = await method(*args, **kwargs_with_model)

                # Log fallback usage if not the primary model
                if (
                    model_string != self.models[0]
                    and self.fallback_config.log_fallbacks
                ):
                    self.logger.info(
                        f"Fallback succeeded: Using {model_string} instead of {self.models[0]}"
                    )

                # Call the callback if provided and we're using a fallback model
                if (
                    model_string != self.models[0]
                    and self.fallback_config.fallback_callback
                ):
                    await maybe_await(
                        self.fallback_config.fallback_callback(
                            primary_model=self.models[0],
                            fallback_model=model_string,
                            error=last_error,
                        )
                    )

                # Create a wrapper generator that forwards chunks but handles errors
                async def safe_generator():
                    """
                    Wrapper generator that handles errors during streaming.

                    This ensures proper error propagation if the generator fails
                    after yielding some chunks.
                    """
                    try:
                        async for chunk in generator:
                            yield chunk
                    except Exception as e:
                        # If the generator fails after yielding some chunks,
                        # propagate the error with proper error type
                        if any(
                            isinstance(e, err_type)
                            for err_type in self.fallback_config.retriable_errors
                        ):
                            # This is a retriable error, let the outer loop handle it
                            raise
                        else:
                            # Non-retriable error, propagate directly
                            raise

                # Return the wrapped generator
                return safe_generator()

            except Exception as e:
                last_error = e

                # Log the failure
                if self.fallback_config.log_fallbacks:
                    self.logger.warning(f"Model {model_string} failed: {str(e)}")

                # Determine if this error should trigger a fallback
                retriable = any(
                    isinstance(e, err_type)
                    for err_type in self.fallback_config.retriable_errors
                )
                if not retriable:
                    # Non-retriable error - raise immediately
                    raise

                # Continue to next fallback

        # If we get here, all models failed
        if last_error:
            # Use the correct fallback_models list based on max_fallbacks
            fallback_models = self.models[1:]
            if self.fallback_config.max_fallbacks is not None:
                fallback_models = self.models[1:self.fallback_config.max_fallbacks + 1]

            raise FallbackExhaustionError(
                message=f"All models failed: {str(last_error)}",
                primary_model=self.models[0],
                fallback_models=fallback_models,
                models_tried=models_tried,
                original_error=last_error,
            )

        # Should never reach here, but just in case
        raise APIError(
            f"All models failed but no error was recorded. Models tried: {models_tried}"
        )

    async def create_chat_completion(
        self,
        messages: List[Message],
        model: str = None,  # Ignored since we use models from the proxy
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create a chat completion with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            messages: List of messages to send to the model
            model: Ignored parameter (models are defined in the proxy)
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Either a ChatCompletionResponse or an AsyncGenerator of ChatCompletionChunk
            depending on the stream parameter
        """
        # Special handling for streaming
        if stream:
            try:
                # Get streaming generator from _try_streaming_with_fallbacks
                generator = await self._try_streaming_with_fallbacks(
                    "create_chat_completion", messages=messages, stream=stream, **kwargs
                )

                # Create a wrapper generator that just yields from the inner generator
                async def stream_generator():
                    """Simple wrapper to yield chunks from the inner generator."""
                    async for chunk in generator:
                        yield chunk

                return stream_generator()
            except Exception as e:
                # Propagate exceptions correctly
                raise e
        else:
            # For non-streaming requests, use the standard fallback mechanism
            return await self._try_with_fallbacks(
                "create_chat_completion", messages=messages, stream=stream, **kwargs
            )

    async def create_completion(
        self,
        prompt: str,
        model: str = None,  # Ignored since we use models from the proxy
        stream: bool = False,
        **kwargs,
    ) -> Union[CompletionResponse, AsyncGenerator[Any, None]]:
        """
        Create a text completion with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            prompt: Text prompt to send to the model
            model: Ignored parameter (models are defined in the proxy)
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Either a CompletionResponse or an AsyncGenerator depending on the stream parameter
        """
        # Special handling for streaming
        if stream:
            try:
                # Get streaming generator from _try_streaming_with_fallbacks
                generator = await self._try_streaming_with_fallbacks(
                    "create_completion", prompt=prompt, stream=stream, **kwargs
                )

                # Create a wrapper generator that just yields from the inner generator
                async def stream_generator():
                    """Simple wrapper to yield chunks from the inner generator."""
                    async for chunk in generator:
                        yield chunk

                return stream_generator()
            except Exception as e:
                # Propagate exceptions correctly
                raise e
        else:
            # For non-streaming requests, use the standard fallback mechanism
            return await self._try_with_fallbacks(
                "create_completion", prompt=prompt, stream=stream, **kwargs
            )

    async def create_embedding(
        self,
        input: Union[str, List[str]],
        model: str = None,  # Ignored since we use models from the proxy
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Create embeddings with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            input: Text or list of texts to create embeddings for
            model: Ignored parameter (models are defined in the proxy)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            EmbeddingResponse containing the embeddings
        """
        return await self._try_with_fallbacks("create_embedding", input=input, **kwargs)

    async def upload_file(self, file: Any, purpose: str, **kwargs) -> FileObject:
        """
        Upload a file with fallback support.

        This method will try the primary provider first and fall back to alternative providers
        if the primary provider fails.

        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters to pass to the provider

        Returns:
            FileObject containing information about the uploaded file
        """
        return await self._try_with_fallbacks(
            "upload_file", file=file, purpose=purpose, **kwargs
        )

    async def download_file(self, file_id: str, **kwargs) -> bytes:
        """
        Download a file with fallback support.

        This method will try the primary provider first and fall back to alternative providers
        if the primary provider fails.

        Args:
            file_id: ID of the file to download
            **kwargs: Additional parameters to pass to the provider

        Returns:
            File content as bytes
        """
        return await self._try_with_fallbacks(
            "download_file", file_id=file_id, **kwargs
        )

    async def create_speech(
        self,
        input: str,
        model: str = None,  # Ignored since we use models from the proxy
        **kwargs,
    ) -> bytes:
        """
        Create speech with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            input: Text to convert to speech
            model: Ignored parameter (models are defined in the proxy)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Audio data as bytes
        """
        # This method is not required by the Provider interface, so check provider support
        return await self._try_with_fallbacks("create_speech", input=input, **kwargs)

    async def create_image(
        self,
        prompt: str,
        model: str = None,  # Ignored since we use models from the proxy
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Create images with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            prompt: Text prompt to generate images from
            model: Ignored parameter (models are defined in the proxy)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            List of dictionaries containing image data
        """
        # This method is not required by the Provider interface, so check provider support
        return await self._try_with_fallbacks("create_image", prompt=prompt, **kwargs)

    async def create_transcription(
        self,
        file: Any,
        model: str = None,  # Ignored since we use models from the proxy
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a transcription with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            file: Audio file to transcribe
            model: Ignored parameter (models are defined in the proxy)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dictionary containing the transcription
        """
        return await self._try_with_fallbacks("create_transcription", file=file, **kwargs)

    async def create_translation(
        self,
        file: Any,
        model: str = None,  # Ignored since we use models from the proxy
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a translation with fallback support.

        This method will try the primary model first and fall back to alternative models
        if the primary model fails.

        Args:
            file: Audio file to translate
            model: Ignored parameter (models are defined in the proxy)
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dictionary containing the translation
        """
        return await self._try_with_fallbacks("create_translation", file=file, **kwargs)

    async def list_files(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List files with fallback support.

        This method will try the primary provider first and fall back to alternative providers
        if the primary provider fails.

        Args:
            **kwargs: Additional parameters to pass to the provider

        Returns:
            List of dictionaries containing file information
        """
        return await self._try_with_fallbacks("list_files", **kwargs)

    async def delete_file(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file with fallback support.

        This method will try the primary provider first and fall back to alternative providers
        if the primary provider fails.

        Args:
            file_id: ID of the file to delete
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dictionary containing the deletion result
        """
        return await self._try_with_fallbacks("delete_file", file_id=file_id, **kwargs)

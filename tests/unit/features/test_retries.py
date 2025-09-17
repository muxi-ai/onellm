#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the retry functionality.
"""

import unittest.mock as mock
import asyncio

from onellm.chat_completion import ChatCompletion
from onellm.completion import Completion
from onellm.providers.fallback import FallbackProviderProxy

class TestRetries:
    """Test the retry functionality."""

    @mock.patch("onellm.chat_completion.get_provider_with_fallbacks")
    @mock.patch("onellm.chat_completion.asyncio.run")
    def test_chat_completion_retries_without_fallbacks(self, mock_asyncio_run, mock_get_provider):
        """Test retry parameter adds the same model multiple times to fallbacks."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_chat_completion.return_value = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, "gpt-4")
        mock_asyncio_run.return_value = mock.MagicMock()

        # Call with retries=3
        ChatCompletion.create(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            retries=3
        )

        # Check that get_provider_with_fallbacks was called with the expected arguments
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["primary_model"] == "openai/gpt-4"
        assert kwargs["fallback_models"] == ["openai/gpt-4"] * 3

    @mock.patch("onellm.chat_completion.get_provider_with_fallbacks")
    @mock.patch("onellm.chat_completion.asyncio.run")
    def test_chat_completion_retries_with_fallbacks(self, mock_asyncio_run, mock_get_provider):
        """Test retry parameter adds models before existing fallbacks."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_chat_completion.return_value = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, "gpt-4")
        mock_asyncio_run.return_value = mock.MagicMock()

        # Call with retries=2 and fallback_models
        ChatCompletion.create(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            retries=2,
            fallback_models=["anthropic/claude-3", "openai/gpt-3.5-turbo"]
        )

        # Check that get_provider_with_fallbacks was called with the expected arguments
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["primary_model"] == "openai/gpt-4"
        expected_fallbacks = [
            "openai/gpt-4", "openai/gpt-4", "anthropic/claude-3", "openai/gpt-3.5-turbo"
        ]
        assert kwargs["fallback_models"] == expected_fallbacks

    @mock.patch("onellm.completion.get_provider_with_fallbacks")
    @mock.patch("onellm.completion.asyncio.run")
    def test_completion_retries_without_fallbacks(self, mock_asyncio_run, mock_get_provider):
        """Test retry parameter for Completion adds the same model multiple times."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_completion.return_value = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, "gpt-3.5-turbo-instruct")
        mock_asyncio_run.return_value = mock.MagicMock()

        # Call with retries=3
        Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt="Hello",
            retries=3
        )

        # Check that get_provider_with_fallbacks was called with the expected arguments
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["primary_model"] == "openai/gpt-3.5-turbo-instruct"
        assert kwargs["fallback_models"] == ["openai/gpt-3.5-turbo-instruct"] * 3

    @mock.patch("onellm.completion.get_provider_with_fallbacks")
    @mock.patch("onellm.completion.asyncio.run")
    def test_completion_retries_with_fallbacks(self, mock_asyncio_run, mock_get_provider):
        """Test retry parameter for Completion adds models before existing fallbacks."""
        # Setup mocks
        mock_provider = mock.MagicMock()
        mock_provider.create_completion.return_value = mock.MagicMock()
        mock_get_provider.return_value = (mock_provider, "gpt-3.5-turbo-instruct")
        mock_asyncio_run.return_value = mock.MagicMock()

        # Call with retries=2 and fallback_models
        Completion.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt="Hello",
            retries=2,
            fallback_models=["anthropic/claude-instant-1", "openai/text-davinci-003"]
        )

        # Check that get_provider_with_fallbacks was called with the expected arguments
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["primary_model"] == "openai/gpt-3.5-turbo-instruct"
        expected_fallbacks = [
            "openai/gpt-3.5-turbo-instruct",
            "openai/gpt-3.5-turbo-instruct",
            "anthropic/claude-instant-1",
            "openai/text-davinci-003"
        ]
        assert kwargs["fallback_models"] == expected_fallbacks

    @mock.patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_chat_completion_async_retries(self, mock_get_provider):
        """Test retry parameter works with async methods."""
        # Setup mock
        mock_provider = mock.MagicMock()
        mock_provider.create_chat_completion = mock.AsyncMock()
        mock_get_provider.return_value = (mock_provider, "gpt-4")

        # Call async with retries=2
        async def test_async():
            await ChatCompletion.acreate(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                retries=2
            )

        asyncio.run(test_async())

        # Check that get_provider_with_fallbacks was called with the expected arguments
        mock_get_provider.assert_called_once()
        args, kwargs = mock_get_provider.call_args
        assert kwargs["primary_model"] == "openai/gpt-4"
        assert kwargs["fallback_models"] == ["openai/gpt-4"] * 2

    def test_fallback_provider_integration(self):
        """Test integration with the FallbackProviderProxy class."""
        # Create a fallback provider directly
        primary_model = "openai/gpt-4"
        retries = 3
        fallback_models = ["anthropic/claude-3", "openai/gpt-3.5-turbo"]

        # Prepare the model list as our implementation would
        all_models = [primary_model]
        if retries > 0:
            all_models.extend([primary_model] * retries)
        all_models.extend(fallback_models)

        # Create a fallback provider with these models
        provider = FallbackProviderProxy(all_models)

        # Verify the models are correctly set up
        assert provider.models[0] == primary_model  # Primary model
        assert provider.models[1:4] == [primary_model] * 3  # Retries
        assert provider.models[4:] == fallback_models  # Other fallbacks

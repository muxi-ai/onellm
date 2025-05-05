#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final coverage tests for the completion module.

These tests are designed to achieve 100% coverage for the completion.py module.
"""

import asyncio
from unittest import mock
import pytest

from muxi_llm.completion import Completion


class TestCompletionFinalCoverage:
    """Tests designed to achieve complete coverage of the Completion class."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Save original run function to restore later
        self.original_run = asyncio.run
        # Create a mock for the get_provider_with_fallbacks function
        self.mock_get_provider = mock.patch(
            "muxi_llm.completion.get_provider_with_fallbacks"
        ).start()

        # Create a mock provider
        self.mock_provider = mock.Mock()
        self.mock_provider.create_completion = mock.AsyncMock()

        # Configure get_provider_with_fallbacks to return our mock provider
        self.mock_get_provider.return_value = (self.mock_provider, "model-name")

    def teardown_method(self):
        """Clean up after each test."""
        # Stop all patches
        mock.patch.stopall()
        # Restore original asyncio.run
        asyncio.run = self.original_run

    def test_create_streaming_with_new_event_loop(self):
        """Test create method with streaming=True to cover line 82."""
        # Mock asyncio.new_event_loop and set_event_loop
        mock_loop = mock.Mock()
        mock_loop.run_until_complete = mock.Mock()

        with mock.patch("asyncio.new_event_loop", return_value=mock_loop) as mock_new_loop, \
             mock.patch("asyncio.set_event_loop") as mock_set_loop:

            # Call the create method with streaming=True
            Completion.create(
                model="openai/model",
                prompt="Test prompt",
                stream=True
            )

            # Verify that new_event_loop and set_event_loop were called
            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)

            # Verify that run_until_complete was called
            mock_loop.run_until_complete.assert_called_once()

            # Verify create_completion was called with correct arguments
            self.mock_provider.create_completion.assert_called_once_with(
                prompt="Test prompt",
                model="model-name",
                stream=True
            )

    def test_acreate_with_stream_false(self):
        """Test the acreate method with stream=False to cover line 153."""
        # Configure mock to return a response
        self.mock_provider.create_completion.return_value = "mock_response"

        # Call the acreate method
        result = asyncio.run(Completion.acreate(
            model="openai/model",
            prompt="Test prompt",
            stream=False
        ))

        # Verify result is the provider's response
        assert result == "mock_response"

        # Verify create_completion was called with correct arguments
        self.mock_provider.create_completion.assert_called_once_with(
            prompt="Test prompt",
            model="model-name",
            stream=False
        )

    def test_create_streaming_with_fallback_config(self):
        """Test create method with fallback_config to cover line 82."""
        # Mock the FallbackConfig constructor
        with mock.patch("muxi_llm.completion.FallbackConfig") as mock_fb_config:
            mock_fb_config.return_value = "mock_fb_config"

            # Call the create method with fallback_config
            Completion.create(
                model="openai/model",
                prompt="Test prompt",
                stream=False,
                fallback_config={"max_fallbacks": 3, "log_fallbacks": True}
            )

            # Verify FallbackConfig was initialized with our dictionary
            mock_fb_config.assert_called_once_with(
                max_fallbacks=3,
                log_fallbacks=True
            )

            # Verify get_provider_with_fallbacks was called with the correct fallback_config
            self.mock_get_provider.assert_called_once_with(
                primary_model="openai/model",
                fallback_models=None,
                fallback_config="mock_fb_config"
            )

    def test_acreate_with_fallback_config(self):
        """Test acreate method with fallback_config to cover line 153."""
        # Mock the FallbackConfig constructor
        with mock.patch("muxi_llm.completion.FallbackConfig") as mock_fb_config:
            mock_fb_config.return_value = "mock_fb_config"

            # Configure mock to return a response
            self.mock_provider.create_completion.return_value = "mock_response"

            # Call the acreate method with fallback_config
            result = asyncio.run(Completion.acreate(
                model="openai/model",
                prompt="Test prompt",
                stream=False,
                fallback_config={"max_fallbacks": 2, "log_fallbacks": False}
            ))

            # Verify FallbackConfig was initialized with our dictionary
            mock_fb_config.assert_called_once_with(
                max_fallbacks=2,
                log_fallbacks=False
            )

            # Verify get_provider_with_fallbacks was called with the correct fallback_config
            self.mock_get_provider.assert_called_once_with(
                primary_model="openai/model",
                fallback_models=None,
                fallback_config="mock_fb_config"
            )

            # Verify result is the provider's response
            assert result == "mock_response"

    def test_create_empty_prompt(self):
        """Test that create raises ValueError with empty prompt (line 77)."""
        # Test with empty string
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            Completion.create(
                model="openai/model",
                prompt=""
            )

        # Test with whitespace-only string
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            Completion.create(
                model="openai/model",
                prompt="   "
            )

    @pytest.mark.asyncio
    async def test_acreate_empty_prompt(self):
        """Test that acreate raises ValueError with empty prompt (line 153)."""
        # Test with empty string
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await Completion.acreate(
                model="openai/model",
                prompt=""
            )

        # Test with whitespace-only string
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await Completion.acreate(
                model="openai/model",
                prompt="   "
            )

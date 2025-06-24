#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the provider capability flags in OneLLM.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import warnings

from onellm.chat_completion import ChatCompletion
from onellm.providers.base import Provider
from onellm.providers.fallback import FallbackProviderProxy


class TestProviderCapabilities(unittest.TestCase):
    """Test cases for provider capability flags."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock providers with different capabilities
        self.fully_capable_provider = MagicMock(spec=Provider)
        self.fully_capable_provider.json_mode_support = True
        self.fully_capable_provider.vision_support = True
        self.fully_capable_provider.audio_input_support = True
        self.fully_capable_provider.streaming_support = True
        self.fully_capable_provider.token_by_token_support = True
        self.fully_capable_provider.create_chat_completion = AsyncMock()

        # Provider with no special capabilities
        self.basic_provider = MagicMock(spec=Provider)
        self.basic_provider.json_mode_support = False
        self.basic_provider.vision_support = False
        self.basic_provider.audio_input_support = False
        self.basic_provider.streaming_support = False
        self.basic_provider.token_by_token_support = False
        self.basic_provider.create_chat_completion = AsyncMock()

    @patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_json_mode_capability(self, mock_get_provider):
        """Test that JSON mode is handled correctly based on provider capability."""
        # Mock the provider that doesn't support JSON mode
        mock_get_provider.return_value = (self.basic_provider, "model-name")

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Make request with JSON mode
            ChatCompletion.create(
                model="test/model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format={"type": "json_object"}
            )

            # Verify warning was issued
            self.assertTrue(
                any("does not support JSON mode" in str(warning.message) for warning in w)
            )

            # Verify JSON mode parameter was removed
            args, kwargs = self.basic_provider.create_chat_completion.call_args
            self.assertNotIn("response_format", kwargs)

            # Verify system message was added to request JSON format
            messages_arg = kwargs.get("messages", [])
            self.assertEqual(len(messages_arg), 2)  # Original message plus system message
            self.assertEqual(messages_arg[0]["role"], "system")
            self.assertIn("json", messages_arg[0]["content"].lower())

    @patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_streaming_capability(self, mock_get_provider):
        """Test that streaming is handled correctly based on provider capability."""
        # Mock the provider that doesn't support streaming
        mock_get_provider.return_value = (self.basic_provider, "model-name")

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Make request with streaming
            ChatCompletion.create(
                model="test/model",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

            # Verify warning was issued
            self.assertTrue(
                any("does not support streaming" in str(warning.message) for warning in w)
            )

            # Verify streaming was disabled
            args, kwargs = self.basic_provider.create_chat_completion.call_args
            self.assertFalse(kwargs["stream"])

    @patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_vision_capability(self, mock_get_provider):
        """Test that vision content is handled correctly based on provider capability."""
        # Mock the provider that doesn't support vision
        mock_get_provider.return_value = (self.basic_provider, "model-name")

        # Message with image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            }
        ]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Make request with image content
            ChatCompletion.create(
                model="test/model",
                messages=messages
            )

            # Verify warning was issued
            self.assertTrue(
                any("does not support vision/image inputs" in str(warning.message)
                    for warning in w)
            )

            # Verify image content was removed
            args, kwargs = self.basic_provider.create_chat_completion.call_args
            processed_messages = kwargs.get("messages", [])
            self.assertEqual(len(processed_messages), 1)
            self.assertEqual(processed_messages[0]["content"], "What's in this image?")

    @patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_audio_capability(self, mock_get_provider):
        """Test that audio content is handled correctly based on provider capability."""
        # Mock the provider that doesn't support audio
        mock_get_provider.return_value = (self.basic_provider, "model-name")

        # Message with audio content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this audio?"},
                    {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
                ]
            }
        ]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Make request with audio content
            ChatCompletion.create(
                model="test/model",
                messages=messages
            )

            # Verify warning was issued
            self.assertTrue(
                any("does not support audio inputs" in str(warning.message)
                    for warning in w)
            )

            # Verify audio content was removed
            args, kwargs = self.basic_provider.create_chat_completion.call_args
            processed_messages = kwargs.get("messages", [])
            self.assertEqual(len(processed_messages), 1)

            # The content might be a list with a single text item
            content = processed_messages[0]["content"]
            if isinstance(content, list):
                self.assertEqual(len(content), 1)
                self.assertEqual(content[0]["type"], "text")
                self.assertEqual(content[0]["text"], "What's in this audio?")
            else:
                self.assertEqual(content, "What's in this audio?")

    @patch("onellm.chat_completion.get_provider_with_fallbacks")
    def test_all_capabilities_supported(self, mock_get_provider):
        """Test that no warnings or modifications occur when provider supports all capabilities."""
        # Mock the provider that supports all capabilities
        mock_get_provider.return_value = (self.fully_capable_provider, "model-name")

        # Message with mixed content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image and audio:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                    {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
                ]
            }
        ]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Make request with all features
            ChatCompletion.create(
                model="test/model",
                messages=messages,
                stream=True,
                response_format={"type": "json_object"}
            )

            # Verify no warnings were issued
            self.assertEqual(len(w), 0)

            # Verify content was not modified
            args, kwargs = self.fully_capable_provider.create_chat_completion.call_args
            processed_messages = kwargs.get("messages", [])
            self.assertEqual(processed_messages, messages)

            # Verify parameters were passed through
            self.assertTrue(kwargs["stream"])
            self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_fallback_provider_capability_inheritance(self):
        """Test that FallbackProviderProxy correctly inherits capabilities from primary provider."""
        # Mock Provider._get_provider and FallbackProviderProxy._check_provider_capability methods
        with patch("onellm.providers.fallback.get_provider") as mock_get_provider:
            # Set up the mock to return our capable provider
            mock_get_provider.return_value = self.fully_capable_provider

            # Create a FallbackProviderProxy with our mock provider
            fallback_provider = FallbackProviderProxy(
                models=["test/model", "test/fallback"]
            )

            # Verify capabilities are correctly inherited
            self.assertTrue(fallback_provider.json_mode_support)
            self.assertTrue(fallback_provider.vision_support)
            self.assertTrue(fallback_provider.audio_input_support)
            self.assertTrue(fallback_provider.streaming_support)
            self.assertTrue(fallback_provider.token_by_token_support)

            # Now test with a basic provider
            mock_get_provider.return_value = self.basic_provider

            # Create a new FallbackProviderProxy with our basic provider
            basic_fallback_provider = FallbackProviderProxy(
                models=["basic/model", "basic/fallback"]
            )

            # Reset the cache to force re-evaluation
            basic_fallback_provider._json_mode_support = None
            basic_fallback_provider._vision_support = None
            basic_fallback_provider._streaming_support = None

            # Verify capabilities are correctly inherited
            self.assertFalse(basic_fallback_provider.json_mode_support)
            self.assertFalse(basic_fallback_provider.vision_support)
            self.assertFalse(basic_fallback_provider.audio_input_support)
            self.assertFalse(basic_fallback_provider.streaming_support)
            self.assertFalse(basic_fallback_provider.token_by_token_support)


if __name__ == "__main__":
    unittest.main()

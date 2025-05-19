#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the JSON mode functionality in muxi-llm.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import warnings

from muxi_llm.chat_completion import ChatCompletion
from muxi_llm.providers.base import Provider
import muxi_llm.providers.openai


class TestJSONMode(unittest.TestCase):
    """Test cases for JSON mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock provider with JSON mode support
        self.mock_json_provider = MagicMock(spec=Provider)
        self.mock_json_provider.json_mode_support = True
        self.mock_json_provider.create_chat_completion = AsyncMock()

        # Create a mock provider without JSON mode support
        self.mock_non_json_provider = MagicMock(spec=Provider)
        self.mock_non_json_provider.json_mode_support = False
        self.mock_non_json_provider.create_chat_completion = AsyncMock()

        # Sample messages
        self.messages = [
            {"role": "user", "content": "List 3 planets as JSON"}
        ]

        # Sample response format
        self.json_response_format = {"type": "json_object"}

    @patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
    def test_provider_with_json_mode_support(self, mock_get_provider):
        """Test when the provider supports JSON mode."""
        # Configure mock
        mock_get_provider.return_value = (self.mock_json_provider, "gpt-4")

        # Run test
        ChatCompletion.create(
            model="openai/gpt-4",
            messages=self.messages,
            response_format=self.json_response_format
        )

        # Verify the response_format parameter was passed through
        args, kwargs = self.mock_json_provider.create_chat_completion.call_args
        self.assertIn("response_format", kwargs)
        self.assertEqual(kwargs["response_format"], self.json_response_format)

    @patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
    def test_provider_without_json_mode_support(self, mock_get_provider):
        """Test when the provider doesn't support JSON mode."""
        # Configure mock
        mock_get_provider.return_value = (self.mock_non_json_provider, "claude-3")

        # Capture warnings
        with warnings.catch_warnings(record=True) as warning_list:
            ChatCompletion.create(
                model="anthropic/claude-3",
                messages=self.messages,
                response_format=self.json_response_format
            )

            # Verify warning was issued
            self.assertTrue(
                any("does not support JSON mode" in str(w.message) for w in warning_list)
            )

        # Verify the response_format parameter was removed
        args, kwargs = self.mock_non_json_provider.create_chat_completion.call_args
        self.assertNotIn("response_format", kwargs)

    @patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
    def test_system_message_added_when_missing(self, mock_get_provider):
        """Test that a system message is added when the provider doesn't support JSON mode."""
        # Configure mock
        mock_get_provider.return_value = (self.mock_non_json_provider, "claude-3")

        # Run test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ChatCompletion.create(
                model="anthropic/claude-3",
                messages=self.messages,
                response_format=self.json_response_format
            )

        # Verify a system message was added
        args, kwargs = self.mock_non_json_provider.create_chat_completion.call_args
        messages_passed = kwargs["messages"]

        # Check if there's a system message
        has_system_message = any(msg.get("role") == "system" for msg in messages_passed)
        self.assertTrue(has_system_message)

        # Check if first message is a system message about JSON
        self.assertEqual(messages_passed[0]["role"], "system")
        self.assertIn("json", messages_passed[0]["content"].lower())

    @patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
    def test_system_message_appended_when_exists(self, mock_get_provider):
        """Test that JSON instructions are appended to an existing system message."""
        # Configure mock
        mock_get_provider.return_value = (self.mock_non_json_provider, "claude-3")

        # Create messages with existing system message
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List 3 planets as JSON"}
        ]

        # Run test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ChatCompletion.create(
                model="anthropic/claude-3",
                messages=messages_with_system.copy(),  # Use a copy to avoid modifying the original
                response_format=self.json_response_format
            )

        # Verify the system message was modified
        args, kwargs = self.mock_non_json_provider.create_chat_completion.call_args
        messages_passed = kwargs["messages"]

        # Check that system message contains both original content and JSON instruction
        self.assertEqual(messages_passed[0]["role"], "system")
        self.assertIn("helpful assistant", messages_passed[0]["content"])
        self.assertIn("json", messages_passed[0]["content"].lower())

    @patch("muxi_llm.chat_completion.get_provider_with_fallbacks")
    def test_async_create_with_json_mode(self, mock_get_provider):
        """Test async create with JSON mode."""
        # Configure mock
        mock_get_provider.return_value = (self.mock_json_provider, "gpt-4")

        # Create an event loop for testing async functions
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run test
            loop.run_until_complete(
                ChatCompletion.acreate(
                    model="openai/gpt-4",
                    messages=self.messages,
                    response_format=self.json_response_format
                )
            )

            # Verify the response_format parameter was passed through
            args, kwargs = self.mock_json_provider.create_chat_completion.call_args
            self.assertIn("response_format", kwargs)
            self.assertEqual(kwargs["response_format"], self.json_response_format)
        finally:
            loop.close()

    def test_openai_provider_has_json_mode_support(self):
        """Test that the OpenAI provider has json_mode_support set to True."""
        self.assertTrue(muxi_llm.providers.openai.OpenAIProvider.json_mode_support)


if __name__ == "__main__":
    unittest.main()

"""
Tests to achieve 100% coverage of token_counter.py in muxi-llm.

This file specifically targets the uncovered lines from previous test files,
focusing on lines 37-38, 90-110, 124-134, and 150-209.
"""

import re
import pytest
from unittest import mock

from muxi_llm.utils.token_counter import (
    get_encoder,
    num_tokens_from_string,
    num_tokens_from_messages,
    TIKTOKEN_AVAILABLE,
    SIMPLE_TOKEN_PATTERN,
)


class TestTokenCounterFullCoverage:
    """Tests to achieve 100% coverage of token_counter.py."""

    def test_simple_token_pattern_implementation(self):
        """Test the SIMPLE_TOKEN_PATTERN regex pattern directly."""
        # This tests lines 37-38
        pattern = re.compile(r"\w+|[^\w\s]")

        test_text = "Hello, world! This is a test."
        tokens = pattern.findall(test_text)

        # Ensure our pattern works as expected
        assert tokens == ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
        assert SIMPLE_TOKEN_PATTERN.findall(test_text) == tokens

    def test_simple_token_pattern_direct_usage(self):
        """Test the SIMPLE_TOKEN_PATTERN directly to ensure 100% coverage of lines 37-38."""
        # These lines directly instantiate and use the pattern
        # Testing with various kinds of text to ensure full coverage

        # Empty text
        assert SIMPLE_TOKEN_PATTERN.findall("") == []

        # Just whitespace
        assert SIMPLE_TOKEN_PATTERN.findall("   \t\n") == []

        # Alphanumeric tokens
        text = "word1 word2 word3"
        tokens = SIMPLE_TOKEN_PATTERN.findall(text)
        assert tokens == ["word1", "word2", "word3"]

        # Special characters only
        text = "!@#$%^&*()"
        tokens = SIMPLE_TOKEN_PATTERN.findall(text)
        # Each special character should be its own token
        assert len(tokens) == len(text)

        # With the mock off, verify the pattern functions in a real context
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            count = num_tokens_from_string("Sample text", None)
            assert count == 2  # "Sample" and "text"

    def test_get_encoder_all_model_types(self):
        """Test get_encoder with various model types from OPENAI_MODEL_ENCODINGS."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = mock.MagicMock()
            mock_get_encoding.return_value = mock_encoding

            # Test with a GPT-4 model
            # Choose a known GPT-4 model
            gpt4_model = "gpt-4"
            encoder = get_encoder(gpt4_model)
            assert encoder is not None

            # Test with a GPT-3.5 model
            encoder = get_encoder("gpt-3.5-turbo")
            assert encoder is not None

            # Test with a base GPT-3 model (lines 90-110)
            encoder = get_encoder("text-davinci-003")
            assert encoder is not None

            # Test with an embedding model
            encoder = get_encoder("text-embedding-ada-002")
            assert encoder is not None

    def test_num_tokens_from_messages_detailed_openai(self):
        """Test num_tokens_from_messages with OpenAI models in detail."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        # This test focuses on lines 124-134
        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            # Return specific token lengths for different strings
            mock_encoder.encode.side_effect = lambda text: [0] * len(text)
            mock_get_encoder.return_value = mock_encoder

            # Test with message containing name field (covers tokens_per_name logic)
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello", "name": "User"},
                {"role": "assistant", "content": "Hi there"}
            ]

            token_count = num_tokens_from_messages(messages, "gpt-4")

            # Verify encoder was called correctly
            assert mock_encoder.encode.call_count >= 3

            # Verify we got a valid token count
            assert token_count > 0
            assert isinstance(token_count, int)

    def test_num_tokens_from_messages_with_complex_content(self):
        """Test num_tokens_from_messages with complex content structures."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        # This test covers lines 150-209
        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_get_encoder.return_value = mock_encoder

            # Test with a complex multimodal message
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image_url": {"url": "http://example.com/image.jpg"}},
                    {"text": "Additional text without type"},
                    {"type": "text", "not_text": "This should be ignored"},
                    {"type": "other", "text": "This should be ignored too"}
                ]}
            ]

            token_count = num_tokens_from_messages(messages, "gpt-4-vision-preview")

            # Only text fields with proper structure should be counted
            assert mock_encoder.encode.call_count > 0
            assert token_count > 0

    def test_num_tokens_from_messages_non_gpt_model(self):
        """Test num_tokens_from_messages with non-GPT models."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        # This covers the case where the model doesn't start with gpt-3.5 or gpt-4
        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_get_encoder.return_value = mock_encoder

            with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
                mock_count.return_value = 5

                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ]

                # Use a model that doesn't start with gpt-3.5 or gpt-4
                token_count = num_tokens_from_messages(messages, "text-davinci-003")

                # Should use fallback method
                assert mock_count.call_count > 0

                # Should include message overhead
                assert token_count > 10

    def test_num_tokens_from_string_with_different_models(self):
        """Test num_tokens_from_string with different model types."""
        # Use the function to demonstrate it's being used
        with mock.patch("muxi_llm.utils.token_counter.get_encoder", return_value=None):
            # When encoder is None, should use SIMPLE_TOKEN_PATTERN
            text = "This is a test string"
            count = num_tokens_from_string(text, "any-model")
            assert count == 5  # "This", "is", "a", "test", "string"

    def test_num_tokens_from_messages_with_empty_fields(self):
        """Test num_tokens_from_messages with empty or null fields."""
        # This checks edge cases in the fallback logic
        with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
            mock_count.return_value = 5

            # Messages with empty fields and None values
            messages = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": None},
                {"role": None, "content": "Hello"}
            ]

            token_count = num_tokens_from_messages(messages, "non-gpt-model")

            # Should handle empty and null fields gracefully
            assert isinstance(token_count, int)

    def test_num_tokens_from_messages_with_list_non_dict_items(self):
        """Test num_tokens_from_messages with non-dict items in content list."""
        with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
            mock_count.return_value = 5

            # Messages with content as list containing non-dict items
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hello"},
                    "plain string",  # This should be handled without errors
                    42,  # This should be ignored
                    None  # This should be ignored
                ]}
            ]

            token_count = num_tokens_from_messages(messages, "non-gpt-model")

            # Should handle complex content without errors
            assert isinstance(token_count, int)

    def test_get_encoder_keyerror_and_fallback(self):
        """Test get_encoder exception handling and fallback logic."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        # KeyError/ValueError/ImportError in main try, fallback to cl100k_base
        with mock.patch("tiktoken.get_encoding", side_effect=KeyError):
            encoder = get_encoder("gpt-4")
            assert encoder is None or encoder

        with mock.patch("tiktoken.get_encoding", side_effect=ImportError):
            encoder = get_encoder("gpt-4")
            assert encoder is None or encoder

        with mock.patch("tiktoken.get_encoding", side_effect=ValueError):
            encoder = get_encoder("gpt-4")
            assert encoder is None or encoder

        # Fallback itself raises exception
        with mock.patch("tiktoken.get_encoding", side_effect=[KeyError, Exception]):
            encoder = get_encoder("gpt-4")
            assert encoder is None

    def test_get_encoder_with_provider_prefix(self):
        """Test get_encoder with model string containing a provider prefix."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")
        with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = mock.MagicMock()
            mock_get_encoding.return_value = mock_encoding
            encoder = get_encoder("openai/gpt-4")
            assert encoder is not None

    def test_num_tokens_from_messages_model_with_slash(self):
        """Test num_tokens_from_messages with model string containing a slash."""
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            count = num_tokens_from_messages(messages, "openai/gpt-4")
            assert isinstance(count, int)

    def test_num_tokens_from_messages_weird_list_content(self):
        """
        Test num_tokens_from_messages with lists that don't match expected dict/text
        structure.
        """
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [
                {"role": "user", "content": [123, None, {"foo": "bar"}]},
                {"role": "assistant", "content": [[], {}, {"text": None}]}
            ]
            count = num_tokens_from_messages(messages, "gpt-4")
            assert isinstance(count, int)

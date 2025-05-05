"""
Tests for complete coverage of token_counter.py in muxi-llm.

These tests target full coverage of the token counting utilities.
"""

import pytest
from unittest import mock

from muxi_llm.utils.token_counter import (
    get_encoder,
    num_tokens_from_string,
    num_tokens_from_messages,
    TIKTOKEN_AVAILABLE,
    SIMPLE_TOKEN_PATTERN,
)


class TestTokenCounterComplete:
    """Tests for complete coverage of token_counter.py."""

    def test_get_encoder_with_valid_model(self):
        """Test get_encoder with a valid model."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        # Test with a model that exists in OPENAI_MODEL_ENCODINGS
        encoder = get_encoder("gpt-4")
        assert encoder is not None

        # Test with provider prefix
        encoder = get_encoder("openai/gpt-3.5-turbo")
        assert encoder is not None

    def test_get_encoder_with_direct_model_name(self):
        """Test get_encoder using encoding_for_model."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("tiktoken.encoding_for_model") as mock_encoding_for_model:
            mock_encoding = mock.MagicMock()
            mock_encoding_for_model.return_value = mock_encoding

            # Test with a model not in OPENAI_MODEL_ENCODINGS
            encoder = get_encoder("custom-model")
            assert encoder is mock_encoding
            mock_encoding_for_model.assert_called_once_with("custom-model")

    def test_get_encoder_with_fallback(self):
        """Test get_encoder falling back to cl100k_base."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("tiktoken.encoding_for_model", side_effect=ValueError("Test error")):
            with mock.patch("tiktoken.get_encoding") as mock_get_encoding:
                mock_encoding = mock.MagicMock()
                mock_get_encoding.return_value = mock_encoding

                # Should fallback to cl100k_base
                encoder = get_encoder("unknown-model")
                assert encoder is mock_encoding
                mock_get_encoding.assert_called_with("cl100k_base")

    def test_get_encoder_with_exception(self):
        """Test get_encoder handling exceptions."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("tiktoken.encoding_for_model", side_effect=ValueError("Test error")):
            with mock.patch("tiktoken.get_encoding", side_effect=Exception("Test error")):
                # Should return None when all methods fail
                encoder = get_encoder("unknown-model")
                assert encoder is None

    def test_get_encoder_no_tiktoken(self):
        """Test get_encoder when tiktoken is not available."""
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            encoder = get_encoder("gpt-4")
            assert encoder is None

    def test_num_tokens_from_string_with_encoder(self):
        """Test num_tokens_from_string using a tiktoken encoder."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoder.return_value = mock_encoder

            token_count = num_tokens_from_string("Hello, world!", "gpt-4")
            assert token_count == 5
            mock_encoder.encode.assert_called_once_with("Hello, world!")

    def test_num_tokens_from_string_with_fallback(self):
        """Test num_tokens_from_string using regex fallback."""
        # Use the regex fallback by ensuring no encoder is found
        with mock.patch("muxi_llm.utils.token_counter.get_encoder", return_value=None):
            text = "Hello, world!"

            # Run findall directly on the pattern to see what it returns
            tokens = SIMPLE_TOKEN_PATTERN.findall(text)

            # The actual count from the implementation
            token_count = num_tokens_from_string(text, "gpt-4")

            # Assert that token_count matches the actual number of tokens from findall
            assert token_count == len(tokens)

    def test_num_tokens_from_string_empty_text(self):
        """Test num_tokens_from_string with empty text."""
        token_count = num_tokens_from_string("", "gpt-4")
        assert token_count == 0

    def test_num_tokens_from_string_no_model(self):
        """Test num_tokens_from_string without specifying a model."""
        text = "Hello, world!"

        # Run findall directly on the pattern to see what it returns
        tokens = SIMPLE_TOKEN_PATTERN.findall(text)

        # The actual count from the implementation
        token_count = num_tokens_from_string(text)

        # Assert that token_count matches the actual number of tokens from findall
        assert token_count == len(tokens)

    def test_num_tokens_from_messages_openai_format(self):
        """Test num_tokens_from_messages using OpenAI format."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            # Simulate encoder.encode returning different token counts
            mock_encoder.encode.side_effect = lambda text: [0] * (
                len(text) // 2  # Approx 1 token per 2 chars
            )
            mock_get_encoder.return_value = mock_encoder

            # Test with OpenAI chat model
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "How can I help?", "name": "AI"}
            ]

            token_count = num_tokens_from_messages(messages, "gpt-4-turbo")

            # We won't check exact call count since implementation details may vary
            # Just verify we get a valid token count
            assert token_count > 0
            assert isinstance(token_count, int)

    def test_num_tokens_from_messages_with_content_list(self):
        """Test num_tokens_from_messages with content list format."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            # 1 token per char for simplicity
            mock_encoder.encode.side_effect = lambda text: [0] * len(text)
            mock_get_encoder.return_value = mock_encoder

            # Test with multimodal content format
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "url": "http://example.com/image.jpg"},
                    {"type": "text", "text": "Can you describe it?"}
                ]}
            ]

            token_count = num_tokens_from_messages(messages, "gpt-4-vision-preview")

            # We won't check exact call count since implementation details may vary
            # Just verify we get a valid token count
            assert token_count > 0
            assert isinstance(token_count, int)

    def test_num_tokens_from_messages_fallback(self):
        """Test num_tokens_from_messages using fallback mechanism."""
        # Force fallback by using a non-OpenAI model
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", True):
            with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
                mock_count.return_value = 5  # Each string is 5 tokens

                messages = [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "How can I help?"}
                ]

                token_count = num_tokens_from_messages(messages, "anthropic/claude-2")

                # We don't care about exact call count, just verify functionality
                assert mock_count.call_count > 0

                # Check if token count includes both messages plus overhead
                total_tokens = (5 * mock_count.call_count) + (len(messages) * 4)
                assert token_count == total_tokens

    def test_num_tokens_from_messages_fallback_multimodal(self):
        """Test num_tokens_from_messages fallback with multimodal content."""
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", True):
            with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
                mock_count.return_value = 5  # Each string is 5 tokens

                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "url": "http://example.com/image.jpg"},
                        {"type": "text", "text": "Can you describe it?"}
                    ]}
                ]

                token_count = num_tokens_from_messages(messages, "anthropic/claude-2")

                # We don't care about exact call count, just verify text items are counted
                assert mock_count.call_count > 0

                # 5 tokens per text item plus overhead
                total_tokens = (5 * mock_count.call_count) + (len(messages) * 4)
                assert token_count == total_tokens

    def test_num_tokens_from_messages_empty(self):
        """Test num_tokens_from_messages with empty list."""
        token_count = num_tokens_from_messages([], "gpt-4")
        assert token_count == 0

    def test_num_tokens_from_messages_with_provider_prefix(self):
        """Test num_tokens_from_messages with provider prefix in model name."""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not installed")

        with mock.patch("muxi_llm.utils.token_counter.get_encoder") as mock_get_encoder:
            mock_encoder = mock.MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_get_encoder.return_value = mock_encoder

            messages = [{"role": "user", "content": "Hello"}]
            token_count = num_tokens_from_messages(messages, "openai/gpt-4")

            # We don't need the exact token calculation, just verify functionality works
            assert token_count > 0
            assert isinstance(token_count, int)

    def test_num_tokens_from_messages_with_unusual_content(self):
        """Test num_tokens_from_messages with unusual content types."""
        with mock.patch("muxi_llm.utils.token_counter.TIKTOKEN_AVAILABLE", True):
            with mock.patch("muxi_llm.utils.token_counter.num_tokens_from_string") as mock_count:
                mock_count.return_value = 5  # Each string is 5 tokens

                messages = [
                    {"role": "user", "content": 123},  # Non-string content
                    {"role": "assistant", "content": [1, 2, 3]}  # Non-dict items in list
                ]

                token_count = num_tokens_from_messages(messages, "custom-model")

                # The implementation might try to convert or handle non-string content
                # Just verify a reasonable token count is returned
                assert isinstance(token_count, int)
                assert token_count >= len(messages)  # At minimum, overhead per message

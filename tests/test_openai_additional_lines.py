import pytest
from unittest import mock
from typing import List

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.types.common import Message
from muxi_llm.errors import AuthenticationError, InvalidRequestError


@pytest.mark.asyncio
class TestOpenAIProviderAdditionalLines:
    """Test targeting specific uncovered lines in the OpenAI provider (lines 123-400)."""

    def setup_method(self):
        """Set up the test environment."""
        # Use a patcher to completely mock get_provider_config
        self.config_patcher = mock.patch('muxi_llm.config.get_provider_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {"api_key": "test-api-key"}

        # Create provider instance with mocked config
        self.provider = OpenAIProvider()

        # Override the api_key directly to ensure consistency in tests
        self.provider.api_key = "test-api-key"

    def teardown_method(self):
        """Clean up patchers."""
        self.config_patcher.stop()

    async def test_process_messages_for_vision_no_images(self):
        """Test message processing with no images (lines 382-420)."""
        # Create messages with no images
        messages: List[Message] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about this concept."},
            {"role": "assistant", "content": "I'd be happy to help."},
            {"role": "user", "content": "Can you elaborate more?"}
        ]

        # Process messages
        processed_messages = self.provider._process_messages_for_vision(messages, "gpt-4")

        # Verify messages are unchanged
        assert processed_messages == messages

    async def test_process_messages_for_vision_with_images(self):
        """Test message processing with images (lines 382-420)."""
        # Create messages with an image
        messages: List[Message] = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]}
        ]

        # Process messages with a vision-capable model
        processed_messages = self.provider._process_messages_for_vision(
            messages, "gpt-4-vision-preview"
        )

        # Verify messages are processed correctly
        assert processed_messages == messages

    async def test_process_messages_for_vision_with_invalid_model(self):
        """Test message processing with images and non-vision model (lines 462-473)."""
        # Create messages with an image
        messages: List[Message] = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]}
        ]

        # Try processing with a non-vision model
        with pytest.raises(InvalidRequestError) as exc_info:
            self.provider._process_messages_for_vision(messages, "gpt-3.5-turbo")

        # Verify error message
        assert "does not support vision inputs" in str(exc_info.value)

    async def test_handle_error_response_with_no_message(self):
        """Test error handling when no error message is present (line 267)."""
        # Test with empty error object
        with pytest.raises(AuthenticationError) as exc_info:
            self.provider._handle_error_response(
                401,
                {"error": {}}  # Empty error object
            )

        # Verify default message is used
        assert "Unknown error" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 401

        # Test with no error object at all
        with pytest.raises(AuthenticationError) as exc_info:
            self.provider._handle_error_response(
                401,
                {}  # No error object
            )

        # Verify default message is used
        assert "Unknown error" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 401

    async def test_create_embedding_with_batched_input(self):
        """Test create_embedding with batched input (lines 415-474)."""
        # Create a list of texts to embed
        texts = ["Hello world", "This is a test", "Embedding example"]

        # Create the mock response object
        mock_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
                {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 2}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 12, "total_tokens": 12}
        }

        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with batched input
            result = await self.provider.create_embedding(
                input=texts,
                model="text-embedding-ada-002"
            )

            # Verify the request parameters
            called_args = mock_request.call_args[1]
            assert called_args["path"] == "/embeddings"
            assert called_args["data"]["model"] == "text-embedding-ada-002"
            assert called_args["data"]["input"] == texts

            # Access result object as data class
            assert len(result.data) == 3
            assert result.model == "text-embedding-ada-002"
            assert result.usage["prompt_tokens"] == 12
            assert result.usage["total_tokens"] == 12

            # Verify individual embeddings
            assert result.data[0].embedding == [0.1, 0.2, 0.3]
            assert result.data[1].embedding == [0.4, 0.5, 0.6]
            assert result.data[2].embedding == [0.7, 0.8, 0.9]

    async def test_create_embedding_with_dimensions(self):
        """Test create_embedding with dimensions parameter (lines 415-474)."""
        # Create input text
        text = "Hello world"

        # Mock the _make_request method
        mock_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }

        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with dimensions parameter
            await self.provider.create_embedding(
                input=text,
                model="text-embedding-ada-002",
                dimensions=2
            )

            # Verify the dimensions parameter was passed
            called_args = mock_request.call_args[1]
            assert called_args["data"]["dimensions"] == 2

    async def test_create_chat_completion_tools(self):
        """Test create_chat_completion with tools parameter (lines 309-390)."""
        # Create messages
        messages: List[Message] = [
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ]

        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        # Mock response from OpenAI
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo, Japan"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }

        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with tools
            result = await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                tools=tools
            )

            # Verify the tools parameter was passed
            called_args = mock_request.call_args[1]
            assert called_args["data"]["tools"] == tools

            # Verify the result contains tool_calls - use dataclass attribute access
            assert hasattr(result, 'choices')
            assert len(result.choices) == 1

            # Access the first choice
            choice = result.choices[0]
            assert hasattr(choice, 'message')

            # Access tool calls
            message = choice.message
            assert 'tool_calls' in message
            tool_calls = message['tool_calls']
            assert len(tool_calls) == 1

            # Verify tool call content
            tool_call = tool_calls[0]
            assert tool_call['function']['name'] == "get_weather"
            assert "Tokyo" in tool_call['function']['arguments']

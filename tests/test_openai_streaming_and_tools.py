import pytest
import json
from unittest import mock

from onellm.providers.openai import OpenAIProvider
from onellm.errors import (
    TimeoutError, AuthenticationError,
    APIError
)


class MockResponse:
    """Mock aiohttp.ClientResponse for testing."""

    def __init__(self, status=200, json_data=None, content=None):
        self.status = status
        self._json_data = json_data
        self._content = content or []

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    @property
    def content(self):
        """Content property that returns self for async iteration."""
        return self

    async def __aiter__(self):
        """Async iterator for content."""
        for chunk in self._content:
            yield chunk


class MockStreamingResponse:
    """Mock streaming response that yields JSON events."""

    def __init__(self, data_chunks, status=200):
        self.status = status
        self._data_chunks = data_chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def content(self):
        return self

    async def __aiter__(self):
        for chunk in self._data_chunks:
            if isinstance(chunk, dict):
                # Convert dict to bytes as if it were a proper data: JSON line
                yield f"data: {json.dumps(chunk)}".encode()
            else:
                # Use string as is (for testing malformed data)
                yield chunk.encode() if isinstance(chunk, str) else chunk


class TestOpenAIStreamingAndTools:
    """Test focusing on streaming implementation and tools handling."""

    def setup_method(self):
        """Set up the test environment."""
        # Use a patcher to completely mock get_provider_config
        self.config_patcher = mock.patch('onellm.config.get_provider_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {"api_key": "test-api-key"}

        # Create provider instance with mocked config
        self.provider = OpenAIProvider()

        # Override the api_key directly to ensure consistency in tests
        self.provider.api_key = "test-api-key"

        # Patch _make_request to bypass the actual HTTP request
        self.request_patcher = mock.patch.object(self.provider, '_make_request')
        self.mock_make_request = self.request_patcher.start()

    def teardown_method(self):
        """Clean up patchers."""
        self.config_patcher.stop()
        self.request_patcher.stop()

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_timeout(self):
        """Test timeout handling in streaming chat completion (lines 321-362)."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create simulated error
        error = TimeoutError("Request timed out", provider="openai", status_code=408)

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the streaming method and expect an error
        with pytest.raises(TimeoutError) as exc_info:
            async for _ in await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-4",
                stream=True
            ):
                pass

        # Verify error details
        assert "Request timed out" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_malformed_json(self):
        """Test handling of malformed JSON in streaming (lines 321-362, 215-220)."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Mock the return of _make_request to be an async generator that yields dicts
        async def mock_generator():
            # Yield valid JSON chunks as dicts (matching the format expected by OpenAI provider)
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None
                    }
                ]
            }

            # The second chunk would be processed by the provider
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": None
                    }
                ]
            }

        # Set the mock to return our generator
        self.mock_make_request.return_value = mock_generator()

        # Call the streaming method
        chunks_received = []
        async for chunk in await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-4",
            stream=True
        ):
            chunks_received.append(chunk)

        # Verify we received the valid chunks
        assert len(chunks_received) == 2
        assert "Hello" in chunks_received[0].choices[0].delta.content
        assert "world" in chunks_received[1].choices[0].delta.content

    @pytest.mark.asyncio
    async def test_chat_completion_with_tool_choice(self):
        """Test create_chat_completion with tool_choice parameter (lines 309-390)."""
        # Create messages
        messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

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

        # Set tool_choice parameter
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

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

        # Set our mock to return the response
        self.mock_make_request.return_value = mock_response

        # Call the method with tools and tool_choice
        await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=tools,
            tool_choice=tool_choice
        )

        # Verify the tool_choice parameter was passed
        called_args = self.mock_make_request.call_args[1]
        assert called_args["data"]["tools"] == tools
        assert called_args["data"]["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_with_tool_calls(self):
        """Test streaming chat completion with tool calls (lines 321-362)."""
        # Create messages
        messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

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

        # Mock the return of _make_request to be an async generator that yields dicts
        async def mock_generator():
            # First chunk with role
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }

            # Tool call name chunk
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {"name": "get_weather"}
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            }

            # Arguments part 1
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"loca'}
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            }

            # Arguments part 2
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'tion": "Tokyo'}
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            }

            # Final chunk
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ', Japan"}'}
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            }

        # Set the mock to return our generator
        self.mock_make_request.return_value = mock_generator()

        # Call the streaming method
        chunks_received = []
        async for chunk in await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=tools,
            stream=True
        ):
            chunks_received.append(chunk)

        # Verify we received the expected number of chunks
        assert len(chunks_received) == 5

        # Check first chunk has role
        assert chunks_received[0].choices[0].delta.role == "assistant"

        # Check tool call name in second chunk
        assert (chunks_received[1].choices[0].delta.tool_calls[0]["function"]["name"] ==
               "get_weather")

        # Check we have tool_calls finish reason in the final chunk
        assert chunks_received[4].choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_api_error(self):
        """Test API error handling in streaming chat completion."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create a simulated API error
        error = APIError(
            "The model 'gpt-5' does not exist",
            provider="openai",
            status_code=404,
            request_id="req_abc123"
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the streaming method and expect an APIError
        with pytest.raises(APIError) as exc_info:
            async for _ in await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-5",  # Using a non-existent model
                stream=True
            ):
                pass

        # Verify error details
        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 404
        assert exc_info.value.request_id == "req_abc123"

    @pytest.mark.asyncio
    async def test_chat_completion_authentication_error(self):
        """Test authentication error handling in regular chat completion."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create a simulated authentication error
        error = AuthenticationError(
            "Invalid API key",
            provider="openai",
            status_code=401
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the non-streaming method and expect an AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-4",
                stream=False
            )

        # Verify error details
        assert "Invalid API key" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_malformed_response(self):
        """Test handling of malformed response data in streaming chat completion."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create a mock response directly representing parsed JSON chunks
        mock_chunks = [
            # Empty dict (should be skipped)
            {},

            # Dict without choices (should be skipped)
            {"id": "chatcmpl-123", "created": 1677858242},

            # Dict with empty choices (should be skipped)
            {"id": "chatcmpl-123", "choices": []},

            # Valid chunk
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None
                    }
                ]
            }
        ]

        # Create a mock generator that will be returned by _make_request
        async def mock_generator():
            for chunk in mock_chunks:
                yield chunk

        # Mock the _make_request method
        self.mock_make_request.return_value = mock_generator()

        # Call the streaming method
        chunks_received = []
        async for chunk in await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-4",
            stream=True
        ):
            chunks_received.append(chunk)

        # We should only get the valid chunk (other chunks should be skipped)
        assert len(chunks_received) == 1
        assert chunks_received[0].choices[0].delta.content == "Hello"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_empty_choices(self):
        """Test handling of empty choices in streaming chat completion."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create a generator that yields a response with no choices
        async def mock_generator():
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-4",
                "choices": []  # Empty choices array
            }

            # Then a valid response
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Valid response"},
                        "finish_reason": None
                    }
                ]
            }

        # Set the mock to return our generator
        self.mock_make_request.return_value = mock_generator()

        # Call the streaming method and collect chunks
        chunks_received = []
        async for chunk in await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-4",
            stream=True
        ):
            chunks_received.append(chunk)

        # We should only get the valid chunk
        assert len(chunks_received) == 1
        assert chunks_received[0].choices[0].delta.content == "Valid response"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_invalid_structure(self):
        """Test handling responses with invalid structure in streaming chat completion."""
        # Create messages
        messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

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

        # Create a mock response with invalid structure
        # Missing delta field in tool call chunk
        mock_chunks = [
            # First chunk with role
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            },
            # Tool call chunk with missing delta information
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        # Missing delta field - should handle gracefully
                        "finish_reason": None
                    }
                ]
            },
            # Valid tool call chunk
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Tokyo, Japan"}'
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            }
        ]

        # Create mock generator
        async def mock_generator():
            for chunk in mock_chunks:
                yield chunk

        # Set the mock
        self.mock_make_request.return_value = mock_generator()

        # Call the streaming method
        chunks_received = []
        async for chunk in await self.provider.create_chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            tools=tools,
            stream=True
        ):
            chunks_received.append(chunk)

        # All three chunks are received, even the one with missing delta
        assert len(chunks_received) == 3

        # First chunk has assistant role
        assert chunks_received[0].choices[0].delta.role == "assistant"

        # Second chunk has empty delta fields
        assert chunks_received[1].choices[0].delta.content is None
        assert chunks_received[1].choices[0].delta.role is None
        assert chunks_received[1].choices[0].delta.function_call is None
        assert chunks_received[1].choices[0].delta.tool_calls is None

        # Third chunk has the tool call
        assert (chunks_received[2].choices[0].delta.tool_calls[0]["function"]["name"] ==
                "get_weather")

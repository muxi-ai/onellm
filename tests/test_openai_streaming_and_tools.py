import pytest
import json
from unittest import mock
import aiohttp
from typing import Dict, Any, List

from muxi_llm.providers.openai import OpenAIProvider
from muxi_llm.errors import TimeoutError
from muxi_llm.models import ChatCompletionChunk, ChoiceDelta, StreamingChoice


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


@pytest.mark.asyncio
class TestOpenAIStreamingAndTools:
    """Test focusing on streaming implementation and tools handling."""

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

    async def test_streaming_chat_completion_timeout(self):
        """Test timeout handling in streaming chat completion (lines 321-362)."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create simulated error
        error = TimeoutError("Request timed out", provider="openai", status_code=408)

        # Create response with error simulation
        mock_response = MockResponse(
            status=200,
            json_data={"choices": [{"delta": {"content": "Hello"}}]}
        )

        # Mock ClientSession.request to return our mock response with error
        mock_session = mock.AsyncMock()
        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Mock aiohttp.ClientSession to return our mock session
        with mock.patch('aiohttp.ClientSession', return_value=mock_session):
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

    async def test_streaming_chat_completion_malformed_json(self):
        """Test handling of malformed JSON in streaming (lines 321-362, 215-220)."""
        # Create messages
        messages = [{"role": "user", "content": "Hello, world!"}]

        # Create response with malformed JSON
        mock_response = MockResponse(
            status=200,
            json_data={"choices": [{"delta": {"content": "Hello"}}]}
        )

        mock_response = MockStreamingResponse(
            [
                {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]},
                "data: {malformed_json",  # This will cause a JSON decode error
                {"id": "chatcmpl-123", "choices": [{"delta": {"content": " world"}}]}
            ],
            status=200
        )

        # Mock ClientSession.request to return our mock response
        mock_session = mock.AsyncMock()
        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Mock aiohttp.ClientSession to return our mock session
        with mock.patch('aiohttp.ClientSession', return_value=mock_session):
            # Call the streaming method
            chunks_received = []
            async for chunk in await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-4",
                stream=True
            ):
                chunks_received.append(chunk)

            # Verify we received the valid chunks and skipped the malformed one
            assert len(chunks_received) == 2
            assert "Hello" in chunks_received[0].choices[0].delta.content
            assert "world" in chunks_received[1].choices[0].delta.content

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

        with mock.patch.object(
            self.provider, '_make_request', return_value=mock_response
        ) as mock_request:
            # Call the method with tools and tool_choice
            await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                tools=tools,
                tool_choice=tool_choice
            )

            # Verify the tool_choice parameter was passed
            called_args = mock_request.call_args[1]
            assert called_args["data"]["tools"] == tools
            assert called_args["data"]["tool_choice"] == tool_choice

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

        # Create streaming chunks that simulate tool calls
        chunks = [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "get_weather"}
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "{\"loca"}
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "tion\": \"Tokyo"}
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": ", Japan\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        ]

        # Create mock response with streaming chunks
        mock_response = MockStreamingResponse(chunks, status=200)

        # Mock ClientSession.request to return our mock response
        mock_session = mock.AsyncMock()
        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Mock aiohttp.ClientSession to return our mock session
        with mock.patch('aiohttp.ClientSession', return_value=mock_session):
            # Call the streaming method
            result = await self.provider.create_chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                tools=tools,
                stream=True
            )

            chunks_received = []
            async for chunk in result:
                chunks_received.append(chunk)

            # Verify we received the correct number of chunks
            assert len(chunks_received) == 5

            # Check the tool calls were correctly parsed
            assert chunks_received[1].choices[0].delta.tool_calls is not None
            assert chunks_received[1].choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"

            # The last chunk should have the finish_reason
            assert chunks_received[-1].choices[0].finish_reason == "tool_calls"

#!/usr/bin/env python3
"""Tests for the AWS Bedrock provider."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from onellm.providers.bedrock import BedrockProvider
from onellm.errors import (
    AuthenticationError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
)


@pytest.fixture
def mock_boto3():
    """Mock boto3 for testing."""
    with patch('onellm.providers.bedrock.boto3') as mock:
        yield mock


@pytest.fixture
def bedrock_provider(mock_boto3):
    """Create a BedrockProvider instance with mocked AWS client."""
    # Mock the session and client creation
    mock_session = Mock()
    mock_client = Mock()
    mock_bedrock_client = Mock()
    
    mock_boto3.Session.return_value = mock_session
    mock_session.client.side_effect = lambda service_name, **kwargs: (
        mock_client if service_name == 'bedrock-runtime' else mock_bedrock_client
    )
    
    # Create provider
    provider = BedrockProvider(region="us-east-1")
    
    # Store mock clients for test access
    provider._mock_client = mock_client
    provider._mock_bedrock_client = mock_bedrock_client
    
    return provider


class TestBedrockProvider:
    """Test cases for BedrockProvider."""
    
    def test_initialization(self, mock_boto3):
        """Test provider initialization."""
        provider = BedrockProvider(region="eu-west-1", profile="test-profile")
        
        assert provider.region == "eu-west-1"
        assert provider.profile == "test-profile"
        assert provider.timeout == 60.0
        assert provider.max_retries == 3
        
        # Verify boto3 session was created with profile
        mock_boto3.Session.assert_called_once_with(profile_name="test-profile")
    
    def test_initialization_with_credentials(self, mock_boto3):
        """Test provider initialization with explicit credentials."""
        provider = BedrockProvider(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
            region="us-west-2"
        )
        
        assert provider.aws_access_key_id == "test-key"
        assert provider.aws_secret_access_key == "test-secret"
        assert provider.aws_session_token == "test-token"
    
    def test_model_name_mapping(self, bedrock_provider):
        """Test model name mapping."""
        # Test predefined mappings
        assert bedrock_provider._map_model_name("claude-3-opus") == "anthropic.claude-3-opus-20240229-v1:0"
        assert bedrock_provider._map_model_name("llama3-2-90b") == "meta.llama3-2-90b-instruct-v1:0"
        assert bedrock_provider._map_model_name("nova-pro") == "amazon.nova-pro-v1:0"
        
        # Test passthrough for full model IDs
        assert bedrock_provider._map_model_name("anthropic.claude-3-opus-20240229-v1:0") == "anthropic.claude-3-opus-20240229-v1:0"
        
        # Test unknown model names
        assert bedrock_provider._map_model_name("unknown-model") == "unknown-model"
    
    def test_message_conversion(self, bedrock_provider):
        """Test OpenAI to Bedrock message conversion."""
        # Test simple text messages
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        bedrock_messages, system_messages = bedrock_provider._convert_openai_to_bedrock_messages(
            messages, "anthropic.claude-3-opus-20240229-v1:0"
        )
        
        assert len(bedrock_messages) == 2  # System message excluded
        assert bedrock_messages[0]["role"] == "user"
        assert bedrock_messages[0]["content"] == [{"text": "Hello"}]
        assert bedrock_messages[1]["role"] == "assistant"
        assert bedrock_messages[1]["content"] == [{"text": "Hi there!"}]
        
        assert system_messages == [{"text": "You are helpful."}]
    
    def test_message_conversion_with_images(self, bedrock_provider):
        """Test message conversion with image content."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
            ]
        }]
        
        bedrock_messages, _ = bedrock_provider._convert_openai_to_bedrock_messages(
            messages, "anthropic.claude-3-opus-20240229-v1:0"
        )
        
        assert len(bedrock_messages) == 1
        assert bedrock_messages[0]["role"] == "user"
        assert len(bedrock_messages[0]["content"]) == 2
        assert bedrock_messages[0]["content"][0] == {"text": "What's in this image?"}
        assert "image" in bedrock_messages[0]["content"][1]
        assert bedrock_messages[0]["content"][1]["image"]["format"] == "jpeg"
    
    @pytest.mark.asyncio
    async def test_chat_completion(self, bedrock_provider):
        """Test chat completion."""
        # Mock the Bedrock response
        mock_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello! How can I help you?"}]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 8,
                "totalTokens": 18
            },
            "stopReason": "stop",
            "$metadata": {"requestId": "test-request-id"}
        }
        
        bedrock_provider._mock_client.converse.return_value = mock_response
        
        # Make the request
        response = await bedrock_provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-opus",
            max_tokens=100
        )
        
        # Verify the response
        assert response.choices[0].message["content"] == "Hello! How can I help you?"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 8
        assert response.model == "claude-3-opus"
        
        # Verify the Bedrock API was called correctly
        bedrock_provider._mock_client.converse.assert_called_once()
        call_args = bedrock_provider._mock_client.converse.call_args[1]
        assert call_args["modelId"] == "anthropic.claude-3-opus-20240229-v1:0"
        assert call_args["inferenceConfig"]["maxTokens"] == 100
    
    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, bedrock_provider):
        """Test streaming chat completion."""
        # Mock the streaming response
        mock_stream = {
            'stream': [
                {'contentBlockStart': {'contentBlockIndex': 0}},
                {'contentBlockDelta': {'delta': {'text': 'Hello'}, 'contentBlockIndex': 0}},
                {'contentBlockDelta': {'delta': {'text': ' there!'}, 'contentBlockIndex': 0}},
                {'contentBlockStop': {'contentBlockIndex': 0}},
                {'messageStop': {'stopReason': 'stop'}},
            ]
        }
        
        bedrock_provider._mock_client.converse_stream.return_value = mock_stream
        
        # Make the streaming request
        stream = await bedrock_provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-haiku",
            stream=True
        )
        
        # Collect chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Verify chunks
        assert len(chunks) == 3  # Two content chunks and one finish chunk
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " there!"
        assert chunks[2].choices[0].finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_embedding(self, bedrock_provider):
        """Test embedding creation."""
        # Mock the Bedrock response for Titan embeddings
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "inputTextTokenCount": 10
        }).encode('utf-8')
        
        bedrock_provider._mock_client.invoke_model.return_value = mock_response
        
        # Make the request
        response = await bedrock_provider.create_embedding(
            input="Test text",
            model="titan-embed-text-v2"
        )
        
        # Verify the response
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.usage["prompt_tokens"] == 10
        
        # Verify the API call
        bedrock_provider._mock_client.invoke_model.assert_called_once()
        call_args = bedrock_provider._mock_client.invoke_model.call_args[1]
        assert call_args["modelId"] == "amazon.titan-embed-text-v2:0"
    
    @pytest.mark.asyncio
    async def test_embedding_with_non_embedding_model(self, bedrock_provider):
        """Test embedding creation with non-embedding model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await bedrock_provider.create_embedding(
                input="Test text",
                model="claude-3-opus"
            )
        
        assert "does not support embeddings" in str(exc_info.value)
    
    def test_error_handling(self, bedrock_provider):
        """Test error handling for various AWS errors."""
        # Test AccessDeniedException
        error = ClientError(
            {
                'Error': {
                    'Code': 'AccessDeniedException',
                    'Message': 'Model not supported for inference in your account'
                },
                'ResponseMetadata': {'HTTPStatusCode': 403}
            },
            'Converse'
        )
        
        with pytest.raises(PermissionError) as exc_info:
            bedrock_provider._handle_bedrock_error(error)
        assert "Model access denied" in str(exc_info.value)
        
        # Test ThrottlingException
        error = ClientError(
            {
                'Error': {
                    'Code': 'ThrottlingException',
                    'Message': 'Too many requests'
                },
                'ResponseMetadata': {'HTTPStatusCode': 429}
            },
            'Converse'
        )
        
        with pytest.raises(RateLimitError):
            bedrock_provider._handle_bedrock_error(error)
        
        # Test ValidationException
        error = ClientError(
            {
                'Error': {
                    'Code': 'ValidationException',
                    'Message': 'Invalid parameter'
                },
                'ResponseMetadata': {'HTTPStatusCode': 400}
            },
            'Converse'
        )
        
        with pytest.raises(InvalidRequestError):
            bedrock_provider._handle_bedrock_error(error)
    
    @pytest.mark.asyncio
    async def test_completion(self, bedrock_provider):
        """Test text completion (converted to chat)."""
        # Mock the chat completion response
        mock_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "The capital of France is Paris."}]
                }
            },
            "usage": {
                "inputTokens": 8,
                "outputTokens": 7,
                "totalTokens": 15
            },
            "stopReason": "stop",
            "$metadata": {"requestId": "test-request-id"}
        }
        
        bedrock_provider._mock_client.converse.return_value = mock_response
        
        # Make the request
        response = await bedrock_provider.create_completion(
            prompt="The capital of France is",
            model="claude-3-haiku"
        )
        
        # Verify the response
        assert response.choices[0].text == "The capital of France is Paris."
        assert response.object == "text_completion"
    
    @pytest.mark.asyncio
    async def test_unsupported_operations(self, bedrock_provider):
        """Test unsupported operations."""
        # Test file upload
        with pytest.raises(InvalidRequestError) as exc_info:
            await bedrock_provider.upload_file("test.txt", "test")
        assert "does not support file uploads" in str(exc_info.value)
        
        # Test file download
        with pytest.raises(InvalidRequestError) as exc_info:
            await bedrock_provider.download_file("file-123")
        assert "does not support file downloads" in str(exc_info.value)
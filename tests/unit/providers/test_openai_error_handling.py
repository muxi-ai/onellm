import pytest
from unittest import mock

from onellm.providers.openai import OpenAIProvider
from onellm.errors import (
    TimeoutError,
    APIError,
    AuthenticationError,
    PermissionError,
    ResourceNotFoundError,
    RateLimitError,
    InvalidRequestError,
    ServiceUnavailableError,
    BadGatewayError,
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


class TestOpenAIErrorHandling:
    """Test class focusing on error handling in the OpenAI provider."""

    def setup_method(self):
        """Set up the test environment."""
        # Use a patcher to completely mock get_provider_config
        self.config_patcher = mock.patch("onellm.config.get_provider_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {"api_key": "test-api-key"}

        # Create provider instance with mocked config
        self.provider = OpenAIProvider()

        # Override the api_key directly to ensure consistency in tests
        self.provider.api_key = "test-api-key"

        # Patch _make_request to bypass the actual HTTP request
        self.request_patcher = mock.patch.object(self.provider, "_make_request")
        self.mock_make_request = self.request_patcher.start()

    def teardown_method(self):
        """Clean up patchers."""
        self.config_patcher.stop()
        self.request_patcher.stop()

    @pytest.mark.asyncio
    async def test_completion_timeout_error(self):
        """Test timeout error handling in text completion."""
        # Create a simulated timeout error
        error = TimeoutError("Request timed out", provider="openai", status_code=408)

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the completion method and expect a TimeoutError
        with pytest.raises(TimeoutError) as exc_info:
            await self.provider.create_completion(prompt="Hello, world!", model="text-davinci-003")

        # Verify error details
        assert "Request timed out" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_completion_authentication_error(self):
        """Test authentication error handling in text completion."""
        # Create a simulated authentication error
        error = AuthenticationError("Invalid API key", provider="openai", status_code=401)

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the completion method and expect an AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            await self.provider.create_completion(prompt="Hello, world!", model="text-davinci-003")

        # Verify error details
        assert "Invalid API key" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_completion_permission_error(self):
        """Test permission error handling in text completion."""
        # Create a simulated permission error
        error = PermissionError(
            "You do not have permission to access this resource", provider="openai", status_code=403
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the completion method and expect a PermissionError
        with pytest.raises(PermissionError) as exc_info:
            await self.provider.create_completion(prompt="Hello, world!", model="text-davinci-003")

        # Verify error details
        assert "permission" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_completion_resource_not_found_error(self):
        """Test resource not found error handling in text completion."""
        # Create a simulated resource not found error
        error = ResourceNotFoundError(
            "The model 'text-davinci-999' does not exist", provider="openai", status_code=404
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the completion method and expect a ResourceNotFoundError
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await self.provider.create_completion(
                prompt="Hello, world!", model="text-davinci-999"  # Non-existent model
            )

        # Verify error details
        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_embedding_rate_limit_error(self):
        """Test rate limit error handling in embeddings."""
        # Create a simulated rate limit error
        error = RateLimitError("Rate limit exceeded", provider="openai", status_code=429)

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the embedding method and expect a RateLimitError
        with pytest.raises(RateLimitError) as exc_info:
            await self.provider.create_embedding(
                input="Hello, world!", model="text-embedding-ada-002"
            )

        # Verify error details
        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_embedding_invalid_request_error(self):
        """Test invalid request error handling in embeddings."""
        # Create a simulated invalid request error
        error = InvalidRequestError(
            "Invalid input: input must be a string", provider="openai", status_code=400
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the embedding method and expect an InvalidRequestError
        with pytest.raises(InvalidRequestError) as exc_info:
            await self.provider.create_embedding(
                input=None, model="text-embedding-ada-002"  # Invalid input
            )

        # Verify error details
        assert "invalid" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self):
        """Test service unavailable error handling."""
        # Create a simulated service unavailable error
        error = ServiceUnavailableError(
            "Service is currently unavailable", provider="openai", status_code=500
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the chat completion method and expect a ServiceUnavailableError
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await self.provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
            )

        # Verify error details
        assert "unavailable" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_bad_gateway_error(self):
        """Test bad gateway error handling."""
        # Create a simulated bad gateway error
        error = BadGatewayError("Bad gateway", provider="openai", status_code=502)

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the chat completion method and expect a BadGatewayError
        with pytest.raises(BadGatewayError) as exc_info:
            await self.provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
            )

        # Verify error details
        assert "gateway" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_generic_api_error(self):
        """Test handling of generic API errors."""
        # Create a simulated API error with a status code that doesn't match specific errors
        error = APIError(
            "Unknown error occurred",
            provider="openai",
            status_code=418,  # I'm a teapot
            error_data={"error": "teapot"},
        )

        # Make the mock _make_request raise the error
        self.mock_make_request.side_effect = error

        # Call the chat completion method and expect an APIError
        with pytest.raises(APIError) as exc_info:
            await self.provider.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
            )

        # Verify error details
        assert "unknown error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 418
        assert exc_info.value.error_data == {"error": "teapot"}

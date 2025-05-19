#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import base64

import muxi_llm
from muxi_llm.errors import InvalidRequestError
from muxi_llm.validators import (
    validate_type,
    validate_dict,
    validate_list,
    validate_string,
    validate_number,
    validate_boolean,
    validate_url,
    validate_base64,
    validate_json,
    validate_model_name,
    validate_messages,
)


class TestTypeValidators:
    """Test the new type validation system."""

    def test_validate_type(self):
        # Test valid cases
        assert validate_type("test", str, "test_var") == "test"
        assert validate_type(123, int, "test_var") == 123
        assert validate_type(None, str, "test_var", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_var must be a str"):
            validate_type(123, str, "test_var")

        with pytest.raises(InvalidRequestError, match="test_var cannot be None"):
            validate_type(None, str, "test_var")

    def test_validate_dict(self):
        # Test valid cases
        test_dict = {"a": 1, "b": 2}
        assert validate_dict(test_dict, "test_dict") == test_dict
        assert validate_dict(test_dict, "test_dict", required_keys=["a"]) == test_dict
        assert (
            validate_dict(test_dict, "test_dict", required_keys=["a", "b"]) == test_dict
        )
        assert (
            validate_dict(
                test_dict, "test_dict", required_keys=["a"], optional_keys=["b"]
            )
            == test_dict
        )
        assert validate_dict(None, "test_dict", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_dict must be a dictionary"):
            validate_dict("not a dict", "test_dict")

        with pytest.raises(
            InvalidRequestError, match="test_dict is missing required key 'c'"
        ):
            validate_dict(test_dict, "test_dict", required_keys=["a", "c"])

        with pytest.raises(
            InvalidRequestError, match="test_dict contains unexpected key 'b'"
        ):
            validate_dict(test_dict, "test_dict", required_keys=["a"], optional_keys=[])

        with pytest.raises(InvalidRequestError, match="test_dict cannot be None"):
            validate_dict(None, "test_dict")

    def test_validate_list(self):
        # Test valid cases
        test_list = [1, 2, 3]
        assert validate_list(test_list, "test_list") == test_list
        assert validate_list(test_list, "test_list", min_length=2) == test_list
        assert validate_list(test_list, "test_list", max_length=5) == test_list
        assert validate_list(None, "test_list", allow_none=True) is None

        # Item validator
        def validate_int_item(item, name):
            if not isinstance(item, int):
                raise InvalidRequestError(f"{name} must be an integer")
            return item

        assert validate_list(
            [1, 2, 3], "test_list", item_validator=validate_int_item
        ) == [1, 2, 3]

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_list must be a list"):
            validate_list("not a list", "test_list")

        with pytest.raises(
            InvalidRequestError, match="test_list must have at least 5 items"
        ):
            validate_list(test_list, "test_list", min_length=5)

        with pytest.raises(
            InvalidRequestError, match="test_list must have at most 2 items"
        ):
            validate_list(test_list, "test_list", max_length=2)

        with pytest.raises(InvalidRequestError, match="test_list cannot be None"):
            validate_list(None, "test_list")

        # Test item validator
        with pytest.raises(
            InvalidRequestError, match="test_list\\[1\\] must be an integer"
        ):
            validate_list(
                [1, "not an int", 3], "test_list", item_validator=validate_int_item
            )

    def test_validate_string(self):
        # Test valid cases
        assert validate_string("test", "test_var") == "test"
        assert validate_string("test", "test_var", min_length=2) == "test"
        assert validate_string("test", "test_var", max_length=10) == "test"
        assert validate_string("test", "test_var", pattern=r"^te") == "test"
        assert (
            validate_string("test", "test_var", allowed_values=["test", "other"])
            == "test"
        )
        assert validate_string(None, "test_var", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_var must be a string"):
            validate_string(123, "test_var")

        with pytest.raises(
            InvalidRequestError, match="test_var must be at least 5 characters long"
        ):
            validate_string("test", "test_var", min_length=5)

        with pytest.raises(
            InvalidRequestError, match="test_var must be at most 3 characters long"
        ):
            validate_string("test", "test_var", max_length=3)

        with pytest.raises(
            InvalidRequestError, match="test_var does not match the required pattern"
        ):
            validate_string("test", "test_var", pattern=r"^x")

        with pytest.raises(
            InvalidRequestError, match="test_var must be one of: 'a', 'b'"
        ):
            validate_string("test", "test_var", allowed_values=["a", "b"])

        with pytest.raises(InvalidRequestError, match="test_var cannot be None"):
            validate_string(None, "test_var")

    def test_validate_number(self):
        # Test valid cases
        assert validate_number(123, "test_var") == 123
        assert validate_number(123.45, "test_var") == 123.45
        assert validate_number(5, "test_var", min_value=5) == 5
        assert validate_number(5, "test_var", max_value=5) == 5
        assert validate_number(5, "test_var", integer_only=True) == 5
        assert validate_number(None, "test_var", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_var must be a number"):
            validate_number("not a number", "test_var")

        with pytest.raises(InvalidRequestError, match="test_var must be at least 10"):
            validate_number(5, "test_var", min_value=10)

        with pytest.raises(InvalidRequestError, match="test_var must be at most 3"):
            validate_number(5, "test_var", max_value=3)

        with pytest.raises(InvalidRequestError, match="test_var must be an integer"):
            validate_number(5.5, "test_var", integer_only=True)

        with pytest.raises(InvalidRequestError, match="test_var cannot be None"):
            validate_number(None, "test_var")

        # Booleans are a subclass of int, should be rejected
        with pytest.raises(InvalidRequestError, match="test_var must be a number"):
            validate_number(True, "test_var")

        with pytest.raises(InvalidRequestError, match="test_var must be an integer"):
            validate_number(True, "test_var", integer_only=True)

    def test_validate_boolean(self):
        # Test valid cases
        assert validate_boolean(True, "test_var") is True
        assert validate_boolean(False, "test_var") is False
        assert validate_boolean(None, "test_var", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_var must be a boolean"):
            validate_boolean("not a boolean", "test_var")

        with pytest.raises(InvalidRequestError, match="test_var must be a boolean"):
            validate_boolean(0, "test_var")

        with pytest.raises(InvalidRequestError, match="test_var cannot be None"):
            validate_boolean(None, "test_var")

    def test_validate_url(self):
        # Test valid cases
        assert validate_url("https://example.com", "test_url") == "https://example.com"
        assert (
            validate_url("https://example.com", "test_url", allowed_schemes=["https"])
            == "https://example.com"
        )
        assert validate_url(None, "test_url", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_url must be a string"):
            validate_url(123, "test_url")

        with pytest.raises(InvalidRequestError, match="test_url is not a valid URL"):
            validate_url("not a url", "test_url")

        with pytest.raises(
            InvalidRequestError,
            match="test_url must use one of the following schemes: https",
        ):
            validate_url("http://example.com", "test_url", allowed_schemes=["https"])

        with pytest.raises(InvalidRequestError, match="test_url cannot be None"):
            validate_url(None, "test_url")

    def test_validate_base64(self):
        # Test valid cases
        b64_string = base64.b64encode(b"test").decode("utf-8")
        assert validate_base64(b64_string, "test_b64") == b64_string
        assert validate_base64(b64_string, "test_b64", max_size_bytes=100) == b64_string
        assert validate_base64(None, "test_b64", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_b64 must be a string"):
            validate_base64(123, "test_b64")

        with pytest.raises(
            InvalidRequestError, match="test_b64 is not a valid base64-encoded string"
        ):
            validate_base64("not base64!", "test_b64")

        large_b64 = base64.b64encode(b"x" * 101).decode("utf-8")
        with pytest.raises(
            InvalidRequestError,
            match="test_b64 exceeds maximum allowed size of 100 bytes",
        ):
            validate_base64(large_b64, "test_b64", max_size_bytes=100)

        with pytest.raises(InvalidRequestError, match="test_b64 cannot be None"):
            validate_base64(None, "test_b64")

    def test_validate_json(self):
        # Test valid cases
        assert validate_json({"a": 1}, "test_json") == {"a": 1}
        assert validate_json([1, 2, 3], "test_json") == [1, 2, 3]
        assert validate_json('{"a": 1}', "test_json") == {"a": 1}
        assert validate_json("[1, 2, 3]", "test_json") == [1, 2, 3]
        assert validate_json(None, "test_json", allow_none=True) is None

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="test_json is not valid JSON"):
            validate_json("{invalid json", "test_json")

        with pytest.raises(
            InvalidRequestError, match="test_json is not JSON-serializable"
        ):
            validate_json({"a": object()}, "test_json")

        with pytest.raises(InvalidRequestError, match="test_json cannot be None"):
            validate_json(None, "test_json")


class TestChatCompletionValidation:
    """Test the integration of validators with ChatCompletion."""

    def test_validate_model_name(self):
        # Test valid cases
        assert validate_model_name("openai/gpt-4") == "openai/gpt-4"

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="Model must be a string"):
            validate_model_name(123)

        with pytest.raises(InvalidRequestError, match="Model name cannot be empty"):
            validate_model_name("")

        with pytest.raises(
            InvalidRequestError, match="does not contain a provider prefix"
        ):
            validate_model_name("gpt-4")

        with pytest.raises(InvalidRequestError, match="Invalid provider name"):
            validate_model_name("OPENAI/gpt-4")

    def test_validate_messages(self):
        # Test valid cases
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        assert validate_messages(messages) == messages

        # Test invalid cases
        with pytest.raises(InvalidRequestError, match="Messages must be a list"):
            validate_messages("not a list")

        with pytest.raises(InvalidRequestError, match="Messages list cannot be empty"):
            validate_messages([])

        with pytest.raises(InvalidRequestError, match="Message 0 must be a dictionary"):
            validate_messages(["not a dict"])

        with pytest.raises(
            InvalidRequestError, match="Message 0 is missing required field 'role'"
        ):
            validate_messages([{"content": "No role"}])

        with pytest.raises(
            InvalidRequestError, match="Message 0 is missing required field 'content'"
        ):
            validate_messages([{"role": "user"}])

        with pytest.raises(
            InvalidRequestError, match="Message 0 has invalid role 'invalid'"
        ):
            validate_messages([{"role": "invalid", "content": "Invalid role"}])

    @pytest.mark.asyncio
    async def test_chat_completion_validation(self, monkeypatch):
        # Mock provider to avoid actual API calls
        class MockProvider:
            json_mode_support = True
            vision_support = True
            audio_input_support = True
            streaming_support = True
            token_by_token_support = True
            realtime_support = True

            async def create_chat_completion(
                self, messages, model, stream=False, **kwargs
            ):
                return muxi_llm.models.ChatCompletionResponse(
                    id="test-id",
                    object="chat.completion",
                    created=1234567890,
                    model=f"test-provider/{model}",
                    choices=[
                        muxi_llm.models.Choice(
                            index=0,
                            message={"role": "assistant", "content": "Test response"},
                            finish_reason="stop",
                        )
                    ],
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                )

        # Create a more reliable mock that will always be used
        mock_provider_instance = MockProvider()

        def mock_get_provider_with_fallbacks(primary_model, **kwargs):
            # Always return our mock provider instance regardless of model
            if "/" in primary_model:
                _, model_name = primary_model.split("/")
            else:
                model_name = primary_model  # Handle invalid model names for testing
            return mock_provider_instance, model_name

        # Apply monkeypatch at the correct import path
        monkeypatch.setattr(
            "muxi_llm.chat_completion.get_provider_with_fallbacks",
            mock_get_provider_with_fallbacks,
        )

        # Also patch it at the providers.base path to be safe
        monkeypatch.setattr(
            "muxi_llm.providers.base.get_provider_with_fallbacks",
            mock_get_provider_with_fallbacks,
        )

        # Test with valid inputs
        valid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        response = await muxi_llm.ChatCompletion.acreate(
            model="openai/gpt-4", messages=valid_messages
        )
        assert response.choices[0].message["content"] == "Test response"

        # Test with invalid model name
        with pytest.raises(
            InvalidRequestError, match="does not contain a provider prefix"
        ):
            await muxi_llm.ChatCompletion.acreate(
                model="gpt-4", messages=valid_messages
            )

        # Test with invalid messages
        with pytest.raises(InvalidRequestError, match="Messages list cannot be empty"):
            await muxi_llm.ChatCompletion.acreate(model="openai/gpt-4", messages=[])


if __name__ == "__main__":
    pytest.main()

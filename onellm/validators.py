#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validation utilities for OneLLM.

This module provides validation functions for ensuring that the data passed
to the API methods is correctly formatted and within expected ranges.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, Type
import re
import json
import base64
from urllib.parse import urlparse
from .errors import InvalidRequestError


# Type variable for generic type validators
T = TypeVar("T")


# -------------------- Type Validation System --------------------


def validate_type(
    value: Any, expected_type: Type[T], name: str, allow_none: bool = False
) -> T:
    """
    Validate that a value has the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type of the value
        name: Name of the value (for error messages)
        allow_none: Whether None is allowed

    Returns:
        The validated value

    Raises:
        InvalidRequestError: If the value has an incorrect type
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is of the expected type
    if not isinstance(value, expected_type):
        raise InvalidRequestError(
            f"{name} must be a {expected_type.__name__}, got {type(value).__name__}"
        )

    return value


def validate_dict(
    value: Any,
    name: str,
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[List[str]] = None,
    allow_none: bool = False,
) -> Dict[str, Any]:
    """
    Validate that a value is a dictionary with the expected keys.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        required_keys: Keys that must be present
        optional_keys: Keys that may be present
        allow_none: Whether None is allowed

    Returns:
        The validated dictionary

    Raises:
        InvalidRequestError: If the value is not a valid dictionary
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a dictionary
    if not isinstance(value, dict):
        raise InvalidRequestError(
            f"{name} must be a dictionary, got {type(value).__name__}"
        )

    # Default to empty lists if not provided
    required_keys = required_keys or []

    # Check required keys
    for key in required_keys:
        if key not in value:
            raise InvalidRequestError(f"{name} is missing required key '{key}'")

    # Check for unexpected keys in two cases:
    # 1. When optional_keys is explicitly provided (even if empty)
    # 2. When required_keys are specified and we want to enforce only those keys
    if optional_keys is not None:
        allowed_keys = set(required_keys) | set(optional_keys)
        for key in value:
            if key not in allowed_keys:
                raise InvalidRequestError(f"{name} contains unexpected key '{key}'")

    return value


def validate_list(
    value: Any,
    name: str,
    item_validator: Optional[Callable[[Any, str], Any]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_none: bool = False,
) -> List[Any]:
    """
    Validate that a value is a list with items of the expected type.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        item_validator: Optional validator for list items
        min_length: Minimum list length
        max_length: Maximum list length
        allow_none: Whether None is allowed

    Returns:
        The validated list

    Raises:
        InvalidRequestError: If the value is not a valid list
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a list
    if not isinstance(value, list):
        raise InvalidRequestError(f"{name} must be a list, got {type(value).__name__}")

    # Check minimum length constraint
    if min_length is not None and len(value) < min_length:
        raise InvalidRequestError(f"{name} must have at least {min_length} items")

    # Check maximum length constraint
    if max_length is not None and len(value) > max_length:
        raise InvalidRequestError(f"{name} must have at most {max_length} items")

    # Validate each item in the list if a validator is provided
    if item_validator:
        for i, item in enumerate(value):
            # Apply validator to each item, use indexed name for error context
            value[i] = item_validator(item, f"{name}[{i}]")

    return value


def validate_string(
    value: Any,
    name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    allowed_values: Optional[List[str]] = None,
    allow_none: bool = False,
) -> Optional[str]:
    """
    Validate that a value is a string with the expected properties.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern the string must match
        allowed_values: List of allowed values
        allow_none: Whether None is allowed

    Returns:
        The validated string

    Raises:
        InvalidRequestError: If the value is not a valid string
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a string
    if not isinstance(value, str):
        raise InvalidRequestError(
            f"{name} must be a string, got {type(value).__name__}"
        )

    # Check minimum length constraint
    if min_length is not None and len(value) < min_length:
        raise InvalidRequestError(
            f"{name} must be at least {min_length} characters long"
        )

    # Check maximum length constraint
    if max_length is not None and len(value) > max_length:
        raise InvalidRequestError(
            f"{name} must be at most {max_length} characters long"
        )

    # Check if string matches the required pattern
    if pattern is not None and not re.match(pattern, value):
        raise InvalidRequestError(f"{name} does not match the required pattern")

    # Check if string is one of the allowed values
    if allowed_values is not None and value not in allowed_values:
        allowed_str = ", ".join(f"'{v}'" for v in allowed_values)
        raise InvalidRequestError(f"{name} must be one of: {allowed_str}")

    return value


def validate_number(
    value: Any,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer_only: bool = False,
    allow_none: bool = False,
) -> Optional[Union[int, float]]:
    """
    Validate that a value is a number with the expected properties.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer_only: Whether only integers are allowed
        allow_none: Whether None is allowed

    Returns:
        The validated number

    Raises:
        InvalidRequestError: If the value is not a valid number
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is an integer (but not a boolean) when integer_only is True
    if integer_only:
        # Exclude booleans which are a subclass of int
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidRequestError(
                f"{name} must be an integer, got {type(value).__name__}"
            )
    else:
        # Check if value is a number (int or float, but not a boolean)
        # Exclude booleans which are a subclass of int
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidRequestError(
                f"{name} must be a number, got {type(value).__name__}"
            )

    # Check minimum value constraint
    if min_value is not None and value < min_value:
        raise InvalidRequestError(f"{name} must be at least {min_value}")

    # Check maximum value constraint
    if max_value is not None and value > max_value:
        raise InvalidRequestError(f"{name} must be at most {max_value}")

    return value


def validate_boolean(value: Any, name: str, allow_none: bool = False) -> Optional[bool]:
    """
    Validate that a value is a boolean.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        allow_none: Whether None is allowed

    Returns:
        The validated boolean

    Raises:
        InvalidRequestError: If the value is not a valid boolean
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a boolean
    if not isinstance(value, bool):
        raise InvalidRequestError(
            f"{name} must be a boolean, got {type(value).__name__}"
        )

    return value


def validate_url(
    value: Any,
    name: str,
    allowed_schemes: Optional[List[str]] = None,
    allow_none: bool = False,
) -> Optional[str]:
    """
    Validate that a value is a valid URL.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        allowed_schemes: List of allowed URL schemes
        allow_none: Whether None is allowed

    Returns:
        The validated URL

    Raises:
        InvalidRequestError: If the value is not a valid URL
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a string
    if not isinstance(value, str):
        raise InvalidRequestError(
            f"{name} must be a string, got {type(value).__name__}"
        )

    try:
        # Parse the URL to check its validity
        result = urlparse(value)
        # A valid URL must have both scheme and netloc (domain) parts
        if not all([result.scheme, result.netloc]):
            raise ValueError("Missing scheme or netloc")

        # Check if the URL scheme is in the list of allowed schemes
        if allowed_schemes and result.scheme not in allowed_schemes:
            schemes_str = ", ".join(allowed_schemes)
            raise InvalidRequestError(
                f"{name} must use one of the following schemes: {schemes_str}"
            )
    except Exception as e:
        # Re-raise InvalidRequestError exceptions, wrap others
        if isinstance(e, InvalidRequestError):
            raise
        raise InvalidRequestError(f"{name} is not a valid URL: {str(e)}")

    return value


def validate_base64(
    value: Any,
    name: str,
    allow_none: bool = False,
    max_size_bytes: Optional[int] = None,
) -> Optional[str]:
    """
    Validate that a value is a valid base64-encoded string.

    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        allow_none: Whether None is allowed
        max_size_bytes: Maximum allowed decoded size in bytes

    Returns:
        The validated base64 string

    Raises:
        InvalidRequestError: If the value is not valid base64
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # Check if value is a string
    if not isinstance(value, str):
        raise InvalidRequestError(
            f"{name} must be a string, got {type(value).__name__}"
        )

    try:
        # Add padding if necessary to make the base64 string valid
        # Base64 strings should have a length that is a multiple of 4
        padding = 4 - (len(value) % 4) if len(value) % 4 != 0 else 0
        value_padded = value + ("=" * padding)

        # Try to decode the base64 string to verify it's valid
        decoded = base64.b64decode(value_padded)

        # Check size if a maximum size is specified
        if max_size_bytes is not None and len(decoded) > max_size_bytes:
            raise InvalidRequestError(
                f"{name} exceeds maximum allowed size of {max_size_bytes} bytes"
            )
    except Exception as e:
        # Re-raise InvalidRequestError exceptions, wrap others
        if isinstance(e, InvalidRequestError):
            raise
        raise InvalidRequestError(
            f"{name} is not a valid base64-encoded string: {str(e)}"
        )

    return value


def validate_json(
    value: Any,
    name: str,
    schema: Optional[Dict[str, Any]] = None,
    allow_none: bool = False,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Validate that a value is valid JSON or a JSON-serializable object.

    Args:
        value: Value to validate (string or object)
        name: Name of the value (for error messages)
        schema: Optional JSON schema to validate against
        allow_none: Whether None is allowed

    Returns:
        The validated JSON object

    Raises:
        InvalidRequestError: If the value is not valid JSON
    """
    # Check if value is None and handle according to allow_none parameter
    if value is None:
        if allow_none:
            return None
        raise InvalidRequestError(f"{name} cannot be None")

    # If the value is a string, try to parse it as JSON
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise InvalidRequestError(f"{name} is not valid JSON: {str(e)}")

    # Check if the value is JSON-serializable by attempting to serialize it
    try:
        json.dumps(value)
    except (TypeError, OverflowError) as e:
        raise InvalidRequestError(f"{name} is not JSON-serializable: {str(e)}")

    # TODO: Add JSON schema validation if needed in the future

    return value


# -------------------- Existing Validator Functions --------------------


def validate_model_name(model: str) -> str:
    """
    Validate that the model name has the expected provider/model format.

    Args:
        model: Model name to validate

    Returns:
        Validated model name

    Raises:
        InvalidRequestError: If the model name is invalid
    """
    # Check if model is a string
    if not isinstance(model, str):
        raise InvalidRequestError(f"Model must be a string, got {type(model).__name__}")

    # Check if model is empty
    if not model:
        raise InvalidRequestError("Model name cannot be empty")

    # Check if model has provider prefix (format: provider/model-name)
    if "/" not in model:
        raise InvalidRequestError(
            f"Model name '{model}' does not contain a provider prefix. "
            f"Use format 'provider/model-name' (e.g., 'openai/gpt-4')."
        )

    # Split the model string into provider and model name parts
    provider, model_name = model.split("/", 1)

    # Validate provider name format (lowercase letters, numbers, underscores, hyphens)
    if not re.match(r"^[a-z0-9_-]+$", provider):
        raise InvalidRequestError(
            f"Invalid provider name '{provider}'. Provider names should contain only "
            f"lowercase letters, numbers, underscores, and hyphens."
        )

    # Validate model name (non-empty)
    if not model_name:
        raise InvalidRequestError("Model name part cannot be empty")

    return model


def validate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate chat messages format.

    Args:
        messages: List of message dictionaries to validate

    Returns:
        Validated messages

    Raises:
        InvalidRequestError: If the messages are invalid
    """
    # Check if messages is a list
    if not isinstance(messages, list):
        raise InvalidRequestError(
            f"Messages must be a list, got {type(messages).__name__}"
        )

    # Check if messages list is empty
    if not messages:
        raise InvalidRequestError("Messages list cannot be empty")

    # Define valid roles for messages
    valid_roles = {"system", "user", "assistant", "tool", "function"}

    # Validate each message in the list
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidRequestError(
                f"Message {i} must be a dictionary, got {type(message).__name__}"
            )

        # Check required fields
        if "role" not in message:
            raise InvalidRequestError(f"Message {i} is missing required field 'role'")

        if (
            "content" not in message
            and "function_call" not in message
            and "tool_calls" not in message
        ):
            raise InvalidRequestError(
                f"Message {i} is missing required field 'content' (or function_call/tool_calls)"
            )

        # Validate role
        role = message["role"]
        if not isinstance(role, str):
            raise InvalidRequestError(
                f"Message {i} role must be a string, got {type(role).__name__}"
            )

        if role not in valid_roles:
            raise InvalidRequestError(
                f"Message {i} has invalid role '{role}'. Valid roles are: {', '.join(valid_roles)}"
            )

        # Validate content
        content = message.get("content")
        if content is not None:
            # Content can be a string or a list of content parts (for multi-modal)
            if not isinstance(content, (str, list)):
                raise InvalidRequestError(
                    f"Message {i} content must be a string or list, got {type(content).__name__}"
                )

                # Validate multi-modal content
                if isinstance(content, list):
                    validate_multimodal_content(content, message_index=i)

        # Validate function_call if present
        function_call = message.get("function_call")
        if function_call is not None:
            if not isinstance(function_call, dict):
                raise InvalidRequestError(
                    f"Message {i} function_call must be a dictionary, "
                    f"got {type(function_call).__name__}"
                )

            if "name" not in function_call:
                raise InvalidRequestError(
                    f"Message {i} function_call is missing required field 'name'"
                )

            if not isinstance(function_call["name"], str):
                raise InvalidRequestError(
                    f"Message {i} function_call name must be a string, "
                    f"got {type(function_call['name']).__name__}"
                )

        # Validate tool_calls if present
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise InvalidRequestError(
                    f"Message {i} tool_calls must be a list, got {type(tool_calls).__name__}"
                )

            for j, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    raise InvalidRequestError(
                        f"Message {i} tool_call {j} must be a dictionary, "
                        f"got {type(tool_call).__name__}"
                    )

                if "id" not in tool_call:
                    raise InvalidRequestError(
                        f"Message {i} tool_call {j} is missing required field 'id'"
                    )

                if "type" not in tool_call:
                    raise InvalidRequestError(
                        f"Message {i} tool_call {j} is missing required field 'type'"
                    )

                if "function" not in tool_call:
                    raise InvalidRequestError(
                        f"Message {i} tool_call {j} is missing required field 'function'"
                    )

    return messages


def validate_multimodal_content(
    content: List[Dict[str, Any]], message_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Validate multi-modal content format.

    Args:
        content: List of content parts to validate
        message_index: Index of the message containing this content (for error messages)

    Returns:
        Validated content

    Raises:
        InvalidRequestError: If the content is invalid
    """
    # Define valid content types for multi-modal messages
    valid_types = {"text", "image_url", "image", "audio_url", "audio"}

    # Validate each content part
    for i, part in enumerate(content):
        if not isinstance(part, dict):
            raise InvalidRequestError(
                f"Message {message_index} content part {i} must be a dictionary, "
                f"got {type(part).__name__}"
            )

        if "type" not in part:
            raise InvalidRequestError(
                f"Message {message_index} content part {i} is missing required field 'type'"
            )

        part_type = part["type"]
        if not isinstance(part_type, str):
            raise InvalidRequestError(
                f"Message {message_index} content part {i} type must be a string, "
                f"got {type(part_type).__name__}"
            )

        if part_type not in valid_types:
            raise InvalidRequestError(
                f"Message {message_index} content part {i} has invalid type '{part_type}'. "
                f"Valid types are: {', '.join(valid_types)}"
            )

        # Validate type-specific fields
        if part_type == "text":
            # Text type requires a 'text' field with a string value
            if "text" not in part:
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} is missing required field 'text'"
                )

            if not isinstance(part["text"], str):
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} text must be a string, "
                    f"got {type(part['text']).__name__}"
                )

        elif part_type == "image_url":
            # Image URL type requires 'image_url' field with a dictionary containing a 'url' field
            if "image_url" not in part:
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} "
                    f"is missing required field 'image_url'"
                )

            image_url = part["image_url"]
            if not isinstance(image_url, dict):
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} image_url must be a dictionary, "
                    f"got {type(image_url).__name__}"
                )

            if "url" not in image_url:
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} "
                    f"image_url is missing required field 'url'"
                )

            if not isinstance(image_url["url"], str):
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} image_url.url must be a string, "
                    f"got {type(image_url['url']).__name__}"
                )

        elif part_type == "audio_url":
            # Audio URL type requires 'audio_url' field with a dictionary containing a 'url' field
            if "audio_url" not in part:
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} "
                    f"is missing required field 'audio_url'"
                )

            audio_url = part["audio_url"]
            if not isinstance(audio_url, dict):
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} audio_url must be a dictionary, "
                    f"got {type(audio_url).__name__}"
                )

            if "url" not in audio_url:
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} "
                    f"audio_url is missing required field 'url'"
                )

            if not isinstance(audio_url["url"], str):
                raise InvalidRequestError(
                    f"Message {message_index} content part {i} audio_url.url must be a string, "
                    f"got {type(audio_url['url']).__name__}"
                )

    return content


def validate_temperature(temperature: Optional[float]) -> Optional[float]:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value to validate

    Returns:
        Validated temperature

    Raises:
        InvalidRequestError: If the temperature is invalid
    """
    # Return None if temperature is None (use default)
    if temperature is None:
        return None

    # Check if temperature is a number (int or float)
    if not isinstance(temperature, (int, float)):
        raise InvalidRequestError(
            f"Temperature must be a number, got {type(temperature).__name__}"
        )

    # Convert to float for consistency
    temperature = float(temperature)

    # Check range (typically between 0 and 2 for LLM temperature)
    if temperature < 0 or temperature > 2:
        raise InvalidRequestError("Temperature must be between 0 and 2")

    return temperature


def validate_top_p(top_p: Optional[float]) -> Optional[float]:
    """
    Validate top_p parameter.

    Args:
        top_p: Top-p value to validate

    Returns:
        Validated top_p

    Raises:
        InvalidRequestError: If the top_p is invalid
    """
    # Return None if top_p is None (use default)
    if top_p is None:
        return None

    # Check if top_p is a number (int or float)
    if not isinstance(top_p, (int, float)):
        raise InvalidRequestError(f"Top_p must be a number, got {type(top_p).__name__}")

    # Convert to float for consistency
    top_p = float(top_p)

    # Check range (top_p must be between 0 and 1)
    if top_p < 0 or top_p > 1:
        raise InvalidRequestError("Top_p must be between 0 and 1")

    return top_p


def validate_presence_penalty(presence_penalty: Optional[float]) -> Optional[float]:
    """
    Validate presence_penalty parameter.

    Args:
        presence_penalty: Presence penalty value to validate

    Returns:
        Validated presence_penalty

    Raises:
        InvalidRequestError: If the presence_penalty is invalid
    """
    # Return None if presence_penalty is None (use default)
    if presence_penalty is None:
        return None

    # Check if presence_penalty is a number (int or float)
    if not isinstance(presence_penalty, (int, float)):
        raise InvalidRequestError(
            f"Presence_penalty must be a number, got {type(presence_penalty).__name__}"
        )

    # Convert to float for consistency
    presence_penalty = float(presence_penalty)

    # Check range (typically between -2 and 2 for presence penalty)
    if presence_penalty < -2 or presence_penalty > 2:
        raise InvalidRequestError("Presence_penalty must be between -2 and 2")

    return presence_penalty


def validate_frequency_penalty(frequency_penalty: Optional[float]) -> Optional[float]:
    """
    Validate frequency_penalty parameter.

    Args:
        frequency_penalty: Frequency penalty value to validate

    Returns:
        Validated frequency_penalty

    Raises:
        InvalidRequestError: If the frequency_penalty is invalid
    """
    # Return None if frequency_penalty is None (use default)
    if frequency_penalty is None:
        return None

    # Check if frequency_penalty is a number (int or float)
    if not isinstance(frequency_penalty, (int, float)):
        raise InvalidRequestError(
            f"Frequency_penalty must be a number, got {type(frequency_penalty).__name__}"
        )

    # Convert to float for consistency
    frequency_penalty = float(frequency_penalty)

    # Check range (typically between -2 and 2 for frequency penalty)
    if frequency_penalty < -2 or frequency_penalty > 2:
        raise InvalidRequestError("Frequency_penalty must be between -2 and 2")

    return frequency_penalty


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Max tokens value to validate

    Returns:
        Validated max_tokens

    Raises:
        InvalidRequestError: If the max_tokens is invalid
    """
    # Return None if max_tokens is None (use default)
    if max_tokens is None:
        return None

    # Check if max_tokens is an integer
    if not isinstance(max_tokens, int):
        raise InvalidRequestError(
            f"Max_tokens must be an integer, got {type(max_tokens).__name__}"
        )

    # Check range (must be positive)
    if max_tokens < 1:
        raise InvalidRequestError("Max_tokens must be greater than 0")

    return max_tokens


def validate_n(n: Optional[int]) -> Optional[int]:
    """
    Validate n parameter (number of completions).

    Args:
        n: Number of completions to validate

    Returns:
        Validated n

    Raises:
        InvalidRequestError: If n is invalid
    """
    # Return None if n is None (use default)
    if n is None:
        return None

    # Check if n is an integer
    if not isinstance(n, int):
        raise InvalidRequestError(f"N must be an integer, got {type(n).__name__}")

    # Check range (must be positive)
    if n < 1:
        raise InvalidRequestError("N must be greater than 0")

    return n


def validate_stop(
    stop: Optional[Union[str, List[str]]],
) -> Optional[Union[str, List[str]]]:
    """
    Validate stop parameter.

    Args:
        stop: Stop sequence(s) to validate

    Returns:
        Validated stop

    Raises:
        InvalidRequestError: If stop is invalid
    """
    # Return None if stop is None (use default)
    if stop is None:
        return None

    # Handle single string stop sequence
    if isinstance(stop, str):
        # Single string is fine
        return stop

    # Handle list of stop sequences
    if isinstance(stop, list):
        # Check that all elements are strings
        for i, s in enumerate(stop):
            if not isinstance(s, str):
                raise InvalidRequestError(
                    f"Stop sequence at index {i} must be a string, got {type(s).__name__}"
                )

        # Check list length (OpenAI API typically limits to 4 stop sequences)
        if len(stop) > 4:
            raise InvalidRequestError("Maximum of 4 stop sequences allowed")

        return stop

    # If we get here, stop is neither None, a string, nor a list
    raise InvalidRequestError(
        f"Stop must be a string or list of strings, got {type(stop).__name__}"
    )


def validate_prompt(prompt: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Validate prompt parameter for text completions.

    This function ensures that the prompt is either a non-empty string or a non-empty
    list of strings, which are the only valid formats for prompts in LLM requests.

    Args:
        prompt: Prompt to validate (string or list of strings)

    Returns:
        Validated prompt (unchanged if valid)

    Raises:
        InvalidRequestError: If prompt is empty, contains non-string elements, or has invalid type
    """
    # Handle single string prompt
    if isinstance(prompt, str):
        # Check that string is not empty
        if not prompt:
            raise InvalidRequestError("Prompt cannot be empty")
        return prompt

    # Handle list of prompts
    if isinstance(prompt, list):
        # Check that all elements are strings
        for i, p in enumerate(prompt):
            if not isinstance(p, str):
                raise InvalidRequestError(
                    f"Prompt at index {i} must be a string, got {type(p).__name__}"
                )

        # Check that list is not empty
        if not prompt:
            raise InvalidRequestError("Prompt list cannot be empty")

        return prompt

    # If we get here, prompt is neither a string nor a list
    raise InvalidRequestError(
        f"Prompt must be a string or list of strings, got {type(prompt).__name__}"
    )


def validate_response_format(
    response_format: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Validate response_format parameter.

    This function validates the response_format parameter used in completion requests,
    ensuring it has the correct structure and valid values for the 'type' field.

    Args:
        response_format: Response format to validate (dictionary or None)

    Returns:
        Validated response_format (unchanged if valid)

    Raises:
        InvalidRequestError: If response_format has invalid structure or values
    """
    # None is a valid value (use default response format)
    if response_format is None:
        return None

    # Must be a dictionary
    if not isinstance(response_format, dict):
        raise InvalidRequestError(
            f"Response_format must be a dictionary, got {type(response_format).__name__}"
        )

    # Check required fields
    if "type" not in response_format:
        raise InvalidRequestError("Response_format is missing required field 'type'")

    # Validate type field
    type_value = response_format["type"]
    if not isinstance(type_value, str):
        raise InvalidRequestError(
            f"Response_format type must be a string, got {type(type_value).__name__}"
        )

    # Check that type is one of the allowed values
    valid_types = {"text", "json_object"}
    if type_value not in valid_types:
        raise InvalidRequestError(
            f"Response_format type '{type_value}' is invalid. "
            f"Valid types are: {', '.join(valid_types)}"
        )

    return response_format


def validate_input_for_embeddings(
    input_value: Union[str, List[str]],
) -> Union[str, List[str]]:
    """
    Validate input parameter for embeddings.

    This function ensures that the input for embedding requests is either a non-empty
    string or a non-empty list of strings, which are the only valid formats for
    embedding inputs.

    Args:
        input_value: Input to validate (string or list of strings)

    Returns:
        Validated input (unchanged if valid)

    Raises:
        InvalidRequestError: If input is empty, contains non-string elements, or has invalid type
    """
    # Handle single string input
    if isinstance(input_value, str):
        # Check that string is not empty
        if not input_value:
            raise InvalidRequestError("Input for embeddings cannot be empty")
        return input_value

    # Handle list of inputs
    if isinstance(input_value, list):
        # Check that all elements are strings
        for i, item in enumerate(input_value):
            if not isinstance(item, str):
                raise InvalidRequestError(
                    f"Input at index {i} must be a string, got {type(item).__name__}"
                )

        # Check that list is not empty
        if not input_value:
            raise InvalidRequestError("Input list for embeddings cannot be empty")

        return input_value

    # If we get here, input is neither a string nor a list
    raise InvalidRequestError(
        f"Input must be a string or list of strings, got {type(input_value).__name__}"
    )


def validate_stream(stream: Optional[bool]) -> Optional[bool]:
    """
    Validate stream parameter.

    This function validates that the stream parameter is either None or a boolean value,
    which determines whether responses should be streamed or returned as a complete response.

    Args:
        stream: Stream value to validate (boolean or None)

    Returns:
        Validated stream value (unchanged if valid)

    Raises:
        InvalidRequestError: If stream is not a boolean or None
    """
    # None is a valid value (use default streaming behavior)
    if stream is None:
        return None

    # Must be a boolean
    if not isinstance(stream, bool):
        raise InvalidRequestError(
            f"Stream must be a boolean, got {type(stream).__name__}"
        )

    return stream


def validate_functions(
    functions: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Validate functions parameter.

    This function validates the functions parameter used in function calling requests,
    ensuring each function has the required fields and correct structure.

    Args:
        functions: Functions to validate (list of dictionaries or None)

    Returns:
        Validated functions (unchanged if valid)

    Raises:
        InvalidRequestError: If functions has invalid structure or missing required fields
    """
    # None is a valid value (no functions)
    if functions is None:
        return None

    # Must be a list
    if not isinstance(functions, list):
        raise InvalidRequestError(
            f"Functions must be a list, got {type(functions).__name__}"
        )

    # Validate each function in the list
    for i, function in enumerate(functions):
        # Each function must be a dictionary
        if not isinstance(function, dict):
            raise InvalidRequestError(
                f"Function {i} must be a dictionary, got {type(function).__name__}"
            )

        # Check required fields
        if "name" not in function:
            raise InvalidRequestError(f"Function {i} is missing required field 'name'")

        # Name must be a string
        if not isinstance(function["name"], str):
            raise InvalidRequestError(
                f"Function {i} name must be a string, got {type(function['name']).__name__}"
            )

        # Validate parameters if present
        parameters = function.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise InvalidRequestError(
                    f"Function {i} parameters must be a dictionary, got {type(parameters).__name__}"
                )

    return functions


def validate_tools(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Validate tools parameter.

    This function validates the tools parameter used in tool-using requests,
    ensuring each tool has the required fields and correct structure. Currently,
    only 'function' type tools are supported.

    Args:
        tools: Tools to validate (list of dictionaries or None)

    Returns:
        Validated tools (unchanged if valid)

    Raises:
        InvalidRequestError: If tools has invalid structure or missing required fields
    """
    # None is a valid value (no tools)
    if tools is None:
        return None

    # Must be a list
    if not isinstance(tools, list):
        raise InvalidRequestError(f"Tools must be a list, got {type(tools).__name__}")

    # Validate each tool in the list
    for i, tool in enumerate(tools):
        # Each tool must be a dictionary
        if not isinstance(tool, dict):
            raise InvalidRequestError(
                f"Tool {i} must be a dictionary, got {type(tool).__name__}"
            )

        # Check required fields
        if "type" not in tool:
            raise InvalidRequestError(f"Tool {i} is missing required field 'type'")

        # Type must be a string
        if not isinstance(tool["type"], str):
            raise InvalidRequestError(
                f"Tool {i} type must be a string, got {type(tool['type']).__name__}"
            )

        # Currently only 'function' type is supported
        if tool["type"] != "function":
            raise InvalidRequestError(
                f"Tool {i} has unsupported type '{tool['type']}'. "
                f"Currently only 'function' is supported."
            )

        # Function field is required for function-type tools
        if "function" not in tool:
            raise InvalidRequestError(f"Tool {i} is missing required field 'function'")

        # Function must be a dictionary
        function = tool["function"]
        if not isinstance(function, dict):
            raise InvalidRequestError(
                f"Tool {i} function must be a dictionary, got {type(function).__name__}"
            )

        # Function name is required
        if "name" not in function:
            raise InvalidRequestError(
                f"Tool {i} function is missing required field 'name'"
            )

        # Function name must be a string
        if not isinstance(function["name"], str):
            raise InvalidRequestError(
                f"Tool {i} function name must be a string, got {type(function['name']).__name__}"
            )

    return tools

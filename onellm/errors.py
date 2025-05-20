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
Standardized error types for OneLLM.

This module provides consistent error classes across different LLM providers
to help with error handling in client code.
"""

from typing import Any, Dict, List, Optional


class MuxiLLMError(Exception):
    """
    Base exception class for OneLLM errors.

    This serves as the parent class for all custom exceptions in the library,
    providing a consistent interface for error handling.

    Attributes:
        message: Human-readable error description
        provider: Name of the LLM provider that generated the error
        status_code: HTTP status code if applicable
        request_id: Unique identifier for the request that failed
        error_data: Additional error details from the provider
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None,
    ):
        # Initialize the parent Exception class with the error message
        super().__init__(message)
        # Store all error details as instance attributes
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        # Initialize error_data as empty dict if None is provided
        self.error_data = error_data or {}

    def __str__(self) -> str:
        """
        Create a formatted string representation of the error.

        Returns:
            A string containing the error message and any available metadata.
        """
        # Add provider information if available
        provider_msg = f" (Provider: {self.provider})" if self.provider else ""
        # Add status code if available
        status_msg = f" Status: {self.status_code}" if self.status_code else ""
        # Add request ID if available
        request_msg = f" Request ID: {self.request_id}" if self.request_id else ""
        # Combine all components into a single error message
        return f"{self.message}{provider_msg}{status_msg}{request_msg}"


class APIError(MuxiLLMError):
    """
    Raised when the provider's API returns an unexpected error.

    This is a general-purpose error for unexpected API responses that don't
    fit into more specific error categories.
    """

    pass


class AuthenticationError(MuxiLLMError):
    """
    Raised when there are authentication issues (invalid API key, etc.).

    This typically occurs when the API key is invalid, expired, or missing.
    """

    pass


class RateLimitError(MuxiLLMError):
    """
    Raised when the provider's rate limit is exceeded.

    This occurs when too many requests are made in a short period of time,
    exceeding the provider's usage limits.
    """

    pass


class InvalidRequestError(MuxiLLMError):
    """
    Raised when the request parameters are invalid.

    This occurs when the request contains invalid parameters, such as
    malformed JSON, invalid model parameters, or other client-side errors.
    """

    pass


class ServiceUnavailableError(MuxiLLMError):
    """
    Raised when the provider's service is unavailable.

    This typically occurs during provider outages or maintenance periods.
    """

    pass


class TimeoutError(MuxiLLMError):
    """
    Raised when a request times out.

    This occurs when the provider takes too long to respond to a request,
    exceeding the configured timeout threshold.
    """

    pass


class BadGatewayError(MuxiLLMError):
    """
    Raised when a bad gateway error occurs.

    This typically indicates an issue with the provider's infrastructure
    or an intermediate proxy server.
    """

    pass


class PermissionError(MuxiLLMError):
    """
    Raised when permission is denied for the requested operation.

    This occurs when the API key doesn't have sufficient permissions
    to access the requested resource or perform the requested operation.
    """

    pass


class ResourceNotFoundError(MuxiLLMError):
    """
    Raised when a requested resource is not found.

    This occurs when attempting to access a resource (model, file, etc.)
    that doesn't exist or has been deleted.
    """

    pass


class InvalidModelError(InvalidRequestError):
    """
    Raised when an invalid or unsupported model is requested.

    This is a specialized form of InvalidRequestError specifically for
    cases where the requested model doesn't exist or isn't available.
    """

    pass


class InvalidConfigurationError(MuxiLLMError):
    """
    Raised when the library is configured incorrectly.

    This occurs when there are issues with the library configuration,
    such as missing required settings or incompatible options.
    """

    pass


class FallbackExhaustionError(MuxiLLMError):
    """
    Error raised when all fallback models have been tried and failed.

    This occurs when the primary model and all specified fallback models
    have been attempted without success, leaving no further options.

    Attributes:
        primary_model: The original model that was requested
        fallback_models: List of fallback models that were specified
        models_tried: List of models that were actually attempted
        original_error: The exception from the last failed attempt
    """

    def __init__(
        self,
        message: str,
        primary_model: str,
        fallback_models: List[str],
        models_tried: List[str],
        original_error: Exception,
        **kwargs,
    ):
        # Initialize the parent class with the message and any additional kwargs
        super().__init__(message, **kwargs)
        # Store information about the models and the original error
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.models_tried = models_tried
        self.original_error = original_error

    def __str__(self) -> str:
        """
        Create a detailed string representation of the fallback exhaustion error.

        Returns:
            A multi-line string containing the error message and details about
            the models that were tried.
        """
        # Get the base error message from the parent class
        base_str = super().__str__()
        # Create comma-separated lists of fallback models and tried models
        fallbacks = ", ".join(self.fallback_models)
        tried = ", ".join(self.models_tried)
        # Return a formatted multi-line error message with all relevant details
        return (
            f"{base_str}\n"
            f"Primary model: {self.primary_model}\n"
            f"Fallback models: {fallbacks}\n"
            f"Models tried: {tried}"
        )

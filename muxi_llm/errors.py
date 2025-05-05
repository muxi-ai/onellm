#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi_llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
Standardized error types for muxi-llm.

This module provides consistent error classes across different LLM providers
to help with error handling in client code.
"""

from typing import Any, Dict, List, Optional


class MuxiLLMError(Exception):
    """Base exception class for muxi-llm errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.error_data = error_data or {}

    def __str__(self) -> str:
        provider_msg = f" (Provider: {self.provider})" if self.provider else ""
        status_msg = f" Status: {self.status_code}" if self.status_code else ""
        request_msg = f" Request ID: {self.request_id}" if self.request_id else ""
        return f"{self.message}{provider_msg}{status_msg}{request_msg}"


class APIError(MuxiLLMError):
    """Raised when the provider's API returns an unexpected error."""
    pass


class AuthenticationError(MuxiLLMError):
    """Raised when there are authentication issues (invalid API key, etc.)."""
    pass


class RateLimitError(MuxiLLMError):
    """Raised when the provider's rate limit is exceeded."""
    pass


class InvalidRequestError(MuxiLLMError):
    """Raised when the request parameters are invalid."""
    pass


class ServiceUnavailableError(MuxiLLMError):
    """Raised when the provider's service is unavailable."""
    pass


class TimeoutError(MuxiLLMError):
    """Raised when a request times out."""
    pass


class BadGatewayError(MuxiLLMError):
    """Raised when a bad gateway error occurs."""
    pass


class PermissionError(MuxiLLMError):
    """Raised when permission is denied for the requested operation."""
    pass


class ResourceNotFoundError(MuxiLLMError):
    """Raised when a requested resource is not found."""
    pass


class InvalidModelError(InvalidRequestError):
    """Raised when an invalid or unsupported model is requested."""
    pass


class InvalidConfigurationError(MuxiLLMError):
    """Raised when the library is configured incorrectly."""
    pass


class FallbackExhaustionError(MuxiLLMError):
    """Error raised when all fallback models have been tried and failed."""

    def __init__(
        self,
        message: str,
        primary_model: str,
        fallback_models: List[str],
        models_tried: List[str],
        original_error: Exception,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.models_tried = models_tried
        self.original_error = original_error

    def __str__(self) -> str:
        base_str = super().__str__()
        fallbacks = ', '.join(self.fallback_models)
        tried = ', '.join(self.models_tried)
        return (
            f"{base_str}\n"
            f"Primary model: {self.primary_model}\n"
            f"Fallback models: {fallbacks}\n"
            f"Models tried: {tried}"
        )

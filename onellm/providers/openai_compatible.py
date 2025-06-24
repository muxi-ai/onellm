#!/usr/bin/env python3
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
Base class for OpenAI-compatible providers.

This module provides a base implementation for providers that are compatible
with the OpenAI API format. It inherits from OpenAIProvider and allows
providers to customize only the necessary parts (API key, base URL, etc.)
while reusing all the OpenAI implementation logic.
"""

from ..config import get_provider_config
from ..errors import AuthenticationError
from .openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """
    Base class for OpenAI-compatible providers.

    This class extends OpenAIProvider and allows subclasses to customize
    provider-specific details while inheriting all OpenAI functionality.
    """

    # Provider name to be set by subclasses
    provider_name: str = None

    # Default API base URL (can be overridden by subclasses)
    default_api_base: str = None

    # Whether this provider requires special headers
    requires_special_headers: bool = False

    def __init__(self, **kwargs):
        """
        Initialize the OpenAI-compatible provider.

        Args:
            api_key: Optional API key
            **kwargs: Additional configuration options
        """
        if self.provider_name is None:
            raise NotImplementedError("Subclasses must set provider_name")

        # Get configuration for the specific provider
        self.config = get_provider_config(self.provider_name)

        # Extract credential parameters
        api_key = kwargs.pop("api_key", None)

        # Filter out any credential parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key"]}

        # Update non-credential configuration
        self.config.update(filtered_kwargs)

        # Apply credentials explicitly provided to the constructor
        if api_key:
            self.config["api_key"] = api_key

        # Check for required configuration (only if API key is required)
        if getattr(self, "requires_api_key", True) and not self.config.get("api_key"):
            env_var_name = f"{self.provider_name.upper()}_API_KEY"
            raise AuthenticationError(
                f"{self.provider_name.title()} API key is required. "
                f"Set it via environment variable {env_var_name} "
                f"or with onellm.{self.provider_name}_api_key = 'your-key'.",
                provider=self.provider_name,
            )

        # Store relevant configuration as instance variables
        self.api_key = self.config.get(
            "api_key", "not-required" if not getattr(self, "requires_api_key", True) else None
        )
        self.api_base = self.config.get("api_base", self.default_api_base)
        self.timeout = self.config.get("timeout", 30.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.organization_id = self.config.get("organization_id", None)

        # Skip the parent __init__ since we're handling everything here
        # Instead, initialize retry config directly
        from ..utils.retry import RetryConfig

        self.retry_config = RetryConfig(
            max_retries=self.max_retries, initial_backoff=1.0, max_backoff=60.0
        )

    def _get_headers(self) -> dict[str, str]:
        """
        Get headers for API requests.

        Can be overridden by subclasses to add provider-specific headers.

        Returns:
            Dict of headers
        """
        headers = super()._get_headers()

        # Allow subclasses to add custom headers
        if self.requires_special_headers:
            headers.update(self._get_special_headers())

        return headers

    def _get_special_headers(self) -> dict[str, str]:
        """
        Get provider-specific headers.

        To be overridden by subclasses that need special headers.

        Returns:
            Dict of special headers
        """
        return {}

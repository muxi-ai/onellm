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
Provider implementations for muxi-llm.

This module imports all available provider implementations,
ensuring they are registered with the provider registry.
"""

from .base import get_provider, list_providers, parse_model_name, register_provider
from .fallback import FallbackProviderProxy
from .openai import OpenAIProvider

# Register provider implementations
register_provider("openai", OpenAIProvider)

# Convenience export
__all__ = [
    "get_provider",
    "parse_model_name",
    "register_provider",
    "list_providers",
    "FallbackProviderProxy",
]

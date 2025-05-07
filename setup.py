#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/ranaroussi/muxi-llm
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
#

import os
from setuptools import setup, find_packages

# Read version from .version file in the muxi_llm package
with open(os.path.join(os.path.dirname(__file__), 'muxi_llm', '.version'), 'r') as f:
    version = f.read().strip()

setup(
    name="muxi-llm",
    version=version,
    description="MUXI LLM provides a unified interface for LLM providers using OpenAI format",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "PyYAML>=6.0.0",
        # Optional (but recommended) dependencies
        "tiktoken>=0.3.0; python_version >= '3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "ruff>=0.0.100",
        ],
    },
    python_requires=">=3.10"
)

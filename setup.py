#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/llm
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
#

import os
from setuptools import setup, find_packages

# Read version from .version file in the muxi_llm package
with open(os.path.join(os.path.dirname(__file__), 'muxi_llm', '.version'), 'r') as f:
    version = f.read().strip()

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="muxi-llm",
    version=version,
    description="A unified interface for interacting with large language models from "
                "various providers - a complete drop-in replacement for OpenAI's client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    url="https://github.com/muxi-ai/llm",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Core dependencies
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "PyYAML>=6.0.0",
        # OpenAI dependency (for drop-in replacement compatibility)
        "openai>=1.0.0",
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
        "all": [
            "anthropic>=0.5.0",  # For future Anthropic provider support
            "google-generativeai>=0.3.0",  # For future Google Gemini support
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="llm, ai, openai, gpt, chatgpt, api, client, claude, gemini, mistral, "
             "multimodal, embeddings, vector",
    project_urls={
        "Source": "https://github.com/muxi-ai/llm",
        "Bug Reports": "https://github.com/muxi-ai/llm/issues",
        "Documentation": "https://github.com/muxi-ai/llm",
        "Changelog": "https://github.com/muxi-ai/llm/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/ranaroussi",
    },
    python_requires=">=3.10"
)

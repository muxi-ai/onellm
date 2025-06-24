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
#

import os
from setuptools import setup, find_packages

# Read version from .version file in the onellm package
with open(os.path.join(os.path.dirname(__file__), 'onellm', '.version'), 'r') as f:
    version = f.read().strip()

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="onellm",
    version=version,
    description="A unified interface for interacting with large language models from "
                "various providers - a complete drop-in replacement for OpenAI's client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    url="https://github.com/muxi-ai/onellm",
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
        # HuggingFace Hub for model downloads
        "huggingface-hub>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "ruff>=0.0.100",
            "python-dotenv>=0.19.0",
        ],
        "all": [
            "anthropic>=0.5.0",  # For Anthropic provider support
            "google-generativeai>=0.3.0",  # For Google Gemini support
            "boto3>=1.26.0",  # For AWS Bedrock support
            "llama-cpp-python>=0.2.0",  # For llama.cpp local models
            "google-auth>=2.16.0",  # For Google Vertex AI
            "google-cloud-aiplatform>=1.38.0",  # For Google Vertex AI
        ],
        "vertexai": [
            "google-auth>=2.16.0",  # For Google Cloud authentication
            "google-cloud-aiplatform>=1.38.0",  # For Vertex AI
        ],
        "bedrock": [
            "boto3>=1.26.0",  # AWS SDK for Python
        ],
        "llama": [
            "llama-cpp-python>=0.2.0",  # For llama.cpp local models
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
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
        "Source": "https://github.com/muxi-ai/onellm",
        "Bug Reports": "https://github.com/muxi-ai/onellm/issues",
        "Documentation": "https://github.com/muxi-ai/onellm",
        "Changelog": "https://github.com/muxi-ai/onellm/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/ranaroussi",
    },
    entry_points={
        'console_scripts': [
            'onellm=onellm.cli:main',
        ],
    },
    python_requires=">=3.10"
)

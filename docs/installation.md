---
layout: default
title: Installation
nav_order: 2
---

# Installation

This guide covers how to install OneLLM and its dependencies.

## Requirements

- Python 3.8 or higher
- pip package manager

## Basic Installation

Install OneLLM using pip:

```bash
pip install onellm
```

This installs the core package with support for all providers.

## Installation Options

### Install with All Dependencies

To install OneLLM with all optional dependencies:

```bash
pip install "onellm[all]"
```

### Install with Specific Provider Dependencies

Install only the dependencies you need:

```bash
# For llama.cpp support
pip install "onellm[llama]"

# For AWS Bedrock support
pip install "onellm[bedrock]"

# For Google Vertex AI support
pip install "onellm[vertexai]"

# Combine multiple extras
pip install "onellm[llama,bedrock,vertexai]"
```

### Development Installation

For development or contributing:

```bash
# Clone the repository
git clone https://github.com/muxi-ai/onellm.git
cd onellm

# Install in development mode
pip install -e ".[dev]"
```

## Verify Installation

Check that OneLLM is installed correctly:

```python
import onellm
print(onellm.__version__)
```

Or from the command line:

```bash
python -c "import onellm; print(onellm.__version__)"
```

## CLI Tools

OneLLM includes a command-line tool for downloading GGUF models for local use:

### Download Models

Download GGUF models directly from HuggingFace:

```bash
# Download a model (saves to ~/llama_models by default)
onellm download --repo-id "TheBloke/Llama-2-7B-GGUF" --filename "llama-2-7b.Q4_K_M.gguf"

# Download to a custom directory
onellm download -r "microsoft/Phi-3-mini-4k-instruct-gguf" -f "Phi-3-mini-4k-instruct-q4.gguf" -o /path/to/models

# Short options also work
onellm download -r "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" -f "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
```

This command is particularly useful when working with the llama.cpp provider, which requires local GGUF model files.

## Dependencies

### Core Dependencies

- `aiohttp>=3.8.0` - Async HTTP client
- `pydantic>=2.0.0` - Data validation
- `typing-extensions>=4.0.0` - Type hints backport

### Optional Dependencies

OneLLM uses optional dependencies to keep the core package lightweight. Install only what you need:

#### Basic Providers (No Extra Dependencies)
These providers work with just the core installation:
- OpenAI, Anthropic, Mistral, Groq, Together AI, Fireworks
- Anyscale, X.AI, Perplexity, DeepSeek, Cohere, OpenRouter
- Azure OpenAI (configuration-based)

#### llama.cpp Support (`onellm[llama]`)
- `llama-cpp-python>=0.2.0` - Direct inference with GGUF models
- Note: Ollama works without extra dependencies (uses HTTP API)

#### AWS Bedrock Support (`onellm[bedrock]`)
- `boto3>=1.26.0` - AWS SDK for Python

#### Google Vertex AI Support (`onellm[vertexai]`)
- `google-auth>=2.16.0` - Google Cloud authentication
- `google-cloud-aiplatform>=1.38.0` - Vertex AI client library

#### All Providers (`onellm[all]`)
Includes all optional dependencies for maximum compatibility

## Environment Setup

### Setting API Keys

OneLLM reads API keys from environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google AI Studio
export GOOGLE_API_KEY="..."

# Add other provider keys as needed
```

### Using a .env File

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

Then load it in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()

from onellm import OpenAI
client = OpenAI()
```

## Platform-Specific Notes

### macOS

On macOS with Apple Silicon (M1/M2), you may need to install Rust for some dependencies:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Windows

On Windows, you might need Visual C++ Build Tools for some dependencies. Install from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Linux

Most Linux distributions work out of the box. For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install python3-dev
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the correct Python version:

```bash
python --version  # Should be 3.8 or higher
```

### SSL Certificate Errors

If you encounter SSL errors, update certificates:

```bash
pip install --upgrade certifi
```

### Permission Errors

On some systems, you may need to use `--user` flag:

```bash
pip install --user onellm
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade onellm
```

To upgrade to a specific version:

```bash
pip install onellm==0.1.0
```

## Uninstalling

To uninstall OneLLM:

```bash
pip uninstall onellm
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with OneLLM
- [Configuration](configuration.md) - Configure providers and settings
- [Provider Setup](providers/setup.md) - Set up specific providers
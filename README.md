# muxi-llm

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/ranaroussi/muxi-llm)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A unified interface for interacting with large language models from various providers - a complete drop-in replacement for OpenAI's client with support for hundreds of models.

## Overview

muxi-llm is a lightweight, provider-agnostic Python library that offers a unified interface for interacting with large language models (LLMs) from various providers. It simplifies the integration of LLMs into applications by providing a consistent API while abstracting away provider-specific implementation details.

The library follows the OpenAI client API design pattern, making it familiar to developers already using OpenAI and enabling easy migration for existing applications. **Simply change your import statements and instantly gain access to hundreds of models** across dozens of providers while maintaining your existing code structure.

With support for 25+ providers, muxi-llm gives you access to approximately 300+ unique language models through a single, consistent interface - from the latest proprietary models to open-source alternatives, all accessible through familiar OpenAI-compatible patterns.

## Getting Started

### Installation

```bash
# Basic installation (includes OpenAI compatibility)
pip install muxi-llm

# For all providers (includes dependencies for future provider support)
pip install "muxi-llm[all]"
```

### Quick Start

```python
# Basic usage with OpenAI-compatible syntax
from muxi_llm import ChatCompletion

response = ChatCompletion.create(
    model="openai/gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message["content"])
```

For more detailed examples, check out the [examples directory](./examples).

## Key Features

- **Drop-in replacement for OpenAI** - Use your existing OpenAI code with minimal changes
- **Provider-agnostic** - Support for 300+ models across 25+ LLM providers and services
- **Automatic model fallback** - Seamlessly switch to alternative models when a provider is unavailable
- **Auto-retry mechanism** - Automatically retry the same model multiple times before failing or falling back
- **OpenAI-compatible API** - Familiar interface for developers accustomed to OpenAI's client library
- **Streaming support** - Real-time streaming responses from supported providers
- **Multi-modal capabilities** - Full support for text, images, audio, and video across compatible models
- **Model naming convention** - Consistent `provider/model-name` format for clear attribution
- **Comprehensive test coverage** - Extensive test suite ensuring reliability and compatibility
- **AGPL v3 license** - Open-source license that ensures all improvements remain available to the community

## Supported Providers

muxi-llm supports hundreds of models through:

1. **Direct integration**
2. **OpenRouter connectivity**
3. **Local model support via Ollama**

### Notable Providers

* **OpenAI** - Base implementation and reference API
* **Google Gemini** - OpenAI-compatible API for Gemini models
* **Together AI** - Full compatibility for chat/completion/embedding
* **Anyscale** - OpenAI-compatible endpoints
* **Fireworks AI** - Drop-in replacement for OpenAI
* **Groq** - Compatible API format
* **Mistral AI** - La Plateforme offers OpenAI compatibility
* **Azure OpenAI** - Microsoft's hosted version of OpenAI
* **Perplexity AI** - Compatible chat completions
* **DeepInfra** - OpenAI-compatible API
* **DeepSeek** - OpenAI-compatible API for DeepSeek models
* **Lepton AI** - Compatible endpoints
* **OctoAI** - OpenAI-compatible for selected models
* **Modal** - Hosts compatible endpoints
* **Replicate** - OpenAI compatibility layer for various models
* **NexusFlow** - OpenAI-compatible
* **Voyage AI** - Compatible for embeddings
* **Databricks** - MosaicML offers compatible endpoints
* **OpenRouter** - OpenAI-compatible API for multiple providers
* **Anthropic** - Has its own API structure
* **Ollama** - Custom API for local models
* **llama.cpp** - C/C++ implementation for running LLMs locally with direct hardware access
* **HuggingFace** - Different API structure
* **Sagemaker** - Custom API for embeddings/completions
* **Cohere** - Offers both unique API format and OpenAI-compatible endpoints

### Notable Models Available

Through these providers, you gain access to hundreds of models, including:

- **GPT Family**: GPT-4, GPT-4o, GPT-4 Turbo
- **Claude Family**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Llama Family**: Llama 2, Llama 3, Code Llama
- **Mistral Family**: Mistral 7B, Mistral Large
- **Gemini Family**: Gemini Pro, Gemini Ultra
- **Specialized Models**: Stable Diffusion XL, Command R, Phi-3, Mixtral
- **Embeddings**: Ada-002, text-embedding-3-small/large, Cohere embeddings
- **Multimodal Models**: GPT-4 Vision, Claude 3 Vision, Gemini Pro Vision

## Architecture

muxi-llm follows a modular architecture with clear separation of concerns:

```
muxi_llm/
├── __init__.py                 # Public API exports
├── chat_completion.py          # ChatCompletion class
├── completion.py               # Completion class
├── embedding.py                # Embedding class
├── models.py                   # Response and request model definitions
├── errors.py                   # Error definitions
├── config.py                   # Configuration handling
├── providers/
│   ├── __init__.py
│   ├── base.py                 # Base provider interface
│   ├── openai.py               # OpenAI implementation
│   ├── anthropic.py            # Anthropic implementation
│   ├── ollama.py               # Ollama implementation
│   └── ...                     # Other provider implementations
├── utils/
│   ├── __init__.py
│   ├── streaming.py            # Streaming utilities
│   ├── retry.py                # Retry mechanisms
│   └── token_counter.py        # Token counting utilities
└── types/
    ├── __init__.py
    └── common.py               # Type definitions
```

### Core Components

1. **Public Interface Classes**
   - `ChatCompletion` - For chat-based interactions
   - `Completion` - For text completion
   - `Embedding` - For generating embeddings

2. **Provider Interface**
   - Abstract base class defining required methods
   - Provider-specific implementations

3. **Configuration System**
   - Environment variable support
   - Runtime configuration options

4. **Error Handling**
   - Standardized error types
   - Provider-specific error mapping

5. **Fallback System**
   - Automatic retries with alternative models
   - Configurable fallback chains
   - Graceful degradation options

## API Design

muxi-llm mirrors the OpenAI Python client library API for familiarity:

### Chat Completions

```python
from muxi_llm import ChatCompletion

# Basic usage (identical to OpenAI's client)
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Streaming
for chunk in ChatCompletion.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True
):
    print(chunk.choices[0].delta.content, end="", flush=True)

# With fallback options
response = ChatCompletion.create(
    model=["openai/gpt-4", "anthropic/claude-3-opus", "mistral/mistral-large"],
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# With retries and fallbacks for enhanced reliability
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    retries=3,  # Try gpt-4 up to 3 additional times before failing or using fallbacks
    fallback_models=["anthropic/claude-3-opus", "mistral/mistral-large"]
)

# JSON mode for structured outputs
response = ChatCompletion.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "user", "content": "List the top 3 planets by size"}
    ],
    response_format={"type": "json_object"}  # Request structured JSON output
)
print(response.choices[0].message["content"])  # Outputs valid, parseable JSON

# Multi-modal with images
response = ChatCompletion.create(
    model="openai/gpt-4-vision",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]}
    ]
)

# Multi-modal with audio
response = ChatCompletion.create(
    model="anthropic/claude-3-sonnet",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Transcribe and analyze this audio clip"},
            {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
        ]}
    ]
)
```

### Completions

```python
from muxi_llm import Completion

response = Completion.create(
    model="groq/llama3-70b",
    prompt="Once upon a time",
    max_tokens=100
)
```

### Embeddings

```python
from muxi_llm import Embedding

response = Embedding.create(
    model="openai/text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog"
)
```

## Advanced Features

### Fallback Chains for Enhanced Reliability

muxi-llm includes built-in fallback support to handle API errors gracefully:

```python
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    fallback_models=[
        "anthropic/claude-3-haiku",  # Try Claude if GPT-4 fails
        "openai/gpt-3.5-turbo"       # Try GPT-3.5 if Claude fails
    ]
)
```

If the primary model fails due to service unavailability, rate limiting, or other retriable errors, muxi-llm automatically tries the fallback models in sequence.

### Automatic Retries

For transient errors, you can configure muxi-llm to retry the same model multiple times before falling back to alternatives:

```python
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    retries=3,  # Will try the same model up to 3 additional times if it fails
    fallback_models=["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"]
)
```

This is implemented using the fallback mechanism under the hood, making it both powerful and efficient.

### JSON Mode for Structured Outputs

For applications that require structured data, muxi-llm supports JSON mode with compatible providers:

```python
response = ChatCompletion.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "user", "content": "List the top 3 programming languages with their key features"}
    ],
    response_format={"type": "json_object"}  # Request JSON output
)

# The response contains valid, parseable JSON
json_response = response.choices[0].message["content"]
print(json_response)  # Structured JSON data

# Parse it with standard libraries
import json
structured_data = json.loads(json_response)
```

When using providers that don't natively support JSON mode, muxi-llm automatically adds system instructions requesting JSON-formatted responses.

### Asynchronous Support

Both synchronous and asynchronous APIs are available:

```python
# Synchronous
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Asynchronous with fallbacks and retries
response = await ChatCompletion.acreate(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    retries=2,
    fallback_models=["anthropic/claude-3-opus"]
)
```

## Migration from OpenAI

muxi-llm provides multiple ways to migrate from the OpenAI client, including a fully compatible client interface:

### Option 1: Complete Drop-in Replacement (Identical Interface)

muxi-llm is a complete drop-in replacement for the OpenAI client with the OpenAI library included as a default dependency:

```python
# Before
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello world"}]
)

# After - 100% identical client interface
from muxi_llm import OpenAI  # or Client
client = OpenAI()            # completely compatible with OpenAI's client
response = client.chat.completions.create(
    model="gpt-4",  # automatically adds "openai/" prefix when needed
    messages=[{"role": "user", "content": "Hello world"}]
)
```

### Option 2: Streamlined Direct API (Fewer Lines)

```python
# Before
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello world"}]
)

# After - more concise
from muxi_llm import ChatCompletion
response = ChatCompletion.create(
    model="openai/gpt-4",  # explicitly using provider prefix
    messages=[{"role": "user", "content": "Hello world"}]
)
```

### Option 3: Model Fallback (Enhanced Reliability)

```python
# Adding fallback options with ChatCompletion
from muxi_llm import ChatCompletion
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello world"}],
    fallback_models=[
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo"
    ],
    # optional config
    fallback_config={
        "log_fallbacks": True
    }
)

# Using retries with fallbacks for enhanced reliability
from muxi_llm import ChatCompletion
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello world"}],
    retries=3,  # Will retry gpt-4 up to 3 additional times before using fallbacks
    fallback_models=[
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo"
    ]
)

# Using fallback with the client interface
from muxi_llm import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello world"}],
    fallback_models=[
        "anthropic/claude-3-opus",
        "groq/llama3-70b"
    ]
)
```

## Model Naming Convention

Models are specified using a provider prefix to clearly identify the source:

- `openai/gpt-4` - OpenAI's GPT-4
- `google/gemini-pro` - Google's Gemini Pro model
- `anthropic/claude-3-opus` - Anthropic's Claude model
- `groq/llama3-70b` - Llama 3 via Groq
- `mistral/mistral-large` - Mistral AI's large model
- `cohere/command-r` - Cohere's Command R model
- `deepseek/deepseek-coder` - DeepSeek's coding model
- `ollama/llama3` - Local Llama model via Ollama
- `llama_cpp/llama-3-8b` - Local Llama model via llama.cpp
- `openrouter/anthropic/claude-3` - Claude via OpenRouter

## Configuration

muxi-llm can be configured through environment variables or at runtime:

```python
# Environment variables
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-...

# Runtime configuration
import muxi_llm

muxi_llm.openai_api_key = "sk-..."  # OpenAI API key
muxi_llm.anthropic_api_key = "sk-..."  # Anthropic API key

# Configure fallback behavior
muxi_llm.config.fallback = {
    "enabled": True,
    "default_chains": {
        "chat": ["openai/gpt-4", "anthropic/claude-3-opus", "groq/llama3-70b"],
        "embedding": ["openai/text-embedding-3-small", "cohere/embed-english"]
    },
    "retry_delay": 1.0,
    "max_retries": 3
}
```

## Test Coverage

muxi-llm maintains comprehensive test coverage to ensure reliability and compatibility:

- **Unit tests** for all core components and utilities
- **Integration tests** with mock servers for each provider
- **End-to-end tests** with actual API calls (using recorded responses)
- **Compatibility tests** across different Python versions

The CI pipeline ensures all tests pass before merging changes, maintaining a high standard of quality.

### Current Coverage Metrics

As of May 2024, muxi-llm maintains exceptional test coverage:

- **96% overall package coverage** with 357 passing tests
- **16 modules with perfect 100% coverage**, including core types, models, errors, files, and embeddings
- **All modules maintain 90%+ coverage**, ensuring robust behavior across the entire codebase
- **Key provider implementations at 93-95% coverage**, with only difficult-to-test edge cases remaining uncovered
- **Comprehensive async testing** with robust handling of streaming responses and error conditions

This extensive test coverage ensures reliable operation across all supported providers and models.

## Documentation

muxi-llm uses a code-first documentation approach:

1. **Examples Directory**: The `examples/` directory contains well-documented example scripts that demonstrate all key features of the library:
   - Each example includes detailed frontmatter explaining its purpose and relationship to the codebase
   - Examples range from basic usage to advanced features like fallbacks, retries, and multi-modal capabilities
   - Running the examples is the fastest way to understand the library's capabilities

2. **Code Docstrings**: All public APIs, classes, and methods have comprehensive docstrings:
   - Detailed parameter descriptions
   - Return value documentation
   - Exception information
   - Usage examples

3. **Type Annotations**: The codebase uses Python type annotations throughout:
   - Provides IDE autocompletion support
   - Makes argument requirements clear
   - Enables static type checking

This approach keeps documentation tightly coupled with code, ensuring it stays up-to-date as the library evolves. To get started, we recommend examining the examples that match your use case.

## Call for Contributions

I welcome contributions to muxi-llm! Whether you're fixing bugs, adding features, improving documentation, or supporting new providers, your help is appreciated.

To get started:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests as necessary
5. Submit a pull request

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines and my contributor license agreement.

## License

muxi-llm is licensed under the [GNU Affero General Public License V3 (AGPL-3.0)](./LICENSE).

### Why AGPL?

I chose the AGPL license to ensure that all improvements to muxi-llm remain available to the entire community. This license:

- Ensures that modifications to this library, even when used in distributed software, are shared back with the community
- Promotes collaborative development and prevents proprietary forks that don't contribute back
- Creates a sustainable ecosystem where everyone benefits from improvements
- Allows free usage while ensuring the open-source nature is preserved

For individuals and organizations integrating muxi-llm into their applications, this means you can freely use, modify, and distribute the library, as long as you share your improvements with the community when you distribute your software.

## Developing Custom Providers

### Provider Capability Flags

When developing custom providers for muxi-llm, use capability flags to indicate supported features:

```python
from muxi_llm.providers.base import Provider

class MyCustomProvider(Provider):
    """Custom provider implementation."""

    # Set capability flags
    json_mode_support = True         # Supports structured JSON output

    # Multi-modal capabilities
    vision_support = True            # Supports image inputs
    audio_input_support = False      # No audio input support
    video_input_support = False      # No video input support

    # Streaming capabilities
    streaming_support = True         # Supports streaming responses
    token_by_token_support = True    # Supports granular token streaming

    # Realtime capabilities
    realtime_support = False         # No realtime API support

    # Implement required methods...
```

The library automatically adapts requests based on each provider's capabilities:

1. When JSON mode is requested for providers without support, a system message is added
2. For streaming requests with non-streaming providers, streaming is disabled
3. Image/audio/video content is removed for providers without multimedia support

These flags help the library gracefully handle features across different provider implementations while maintaining consistent behavior.

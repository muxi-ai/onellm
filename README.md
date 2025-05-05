# muxi-llm

A unified interface for interacting with large language models from various providers - a complete drop-in replacement for OpenAI's client with support for hundreds of models.

## Overview

muxi-llm is a lightweight, provider-agnostic Python library that offers a unified interface for interacting with large language models (LLMs) from various providers. It simplifies the integration of LLMs into applications by providing a consistent API while abstracting away provider-specific implementation details.

The library follows the OpenAI client API design pattern, making it familiar to developers already using OpenAI and enabling easy migration for existing applications. **Simply change your import statements and instantly gain access to hundreds of models** across dozens of providers while maintaining your existing code structure.

## Key Features

- **Drop-in replacement for OpenAI** - Use your existing OpenAI code with minimal changes
- **Provider-agnostic** - Support for 100+ LLM providers through direct integration or via OpenRouter
- **Automatic model fallback** - Seamlessly switch to alternative models when a provider is unavailable
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
* **Together AI** - Full compatibility for chat/completion/embedding
* **Anyscale** - OpenAI-compatible endpoints
* **Fireworks AI** - Drop-in replacement for OpenAI
* **Groq** - Compatible API format
* **Mistral AI** - La Plateforme offers OpenAI compatibility
* **Azure OpenAI** - Microsoft's hosted version of OpenAI
* **Perplexity AI** - Compatible chat completions
* **DeepInfra** - OpenAI-compatible API
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
* **HuggingFace** - Different API structure
* **Sagemaker** - Custom API for embeddings/completions
* **Cohere** - Unique API format

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

## Migration from OpenAI

muxi-llm provides multiple ways to migrate from the OpenAI client, including a fully compatible client interface:

### Option 1: Complete Drop-in Replacement (Identical Interface)

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
- `anthropic/claude-3-opus` - Anthropic's Claude model
- `groq/llama3-70b` - Llama 3 via Groq
- `ollama/llama3` - Local Llama model via Ollama
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

## Implementation Plan

The implementation will be phased to deliver value incrementally:

### Phase 1: Core Framework and OpenAI Support

- Basic architecture and interfaces
- OpenAI provider implementation
- Chat completion, text completion, and embedding support
- Unit tests and documentation

### Phase 2: OpenAI-compatible Providers

- Add support for major OpenAI-compatible providers
- Implement streaming capabilities
- Add retry mechanisms and error handling
- Expand test coverage

### Phase 3: Custom API Providers

- Implement Anthropic support
- Implement Ollama support
- Add multi-modal capabilities
- Further enhance documentation and examples

## Test Coverage

muxi-llm maintains comprehensive test coverage to ensure reliability and compatibility:

- **Unit tests** for all core components and utilities
- **Integration tests** with mock servers for each provider
- **End-to-end tests** with actual API calls (using recorded responses)
- **Compatibility tests** across different Python versions
- **Performance benchmarks** for critical operations

The CI pipeline ensures all tests pass before merging changes, maintaining a high standard of quality.

## Contributing

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




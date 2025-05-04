# muxi-llm

A unified interface for interacting with large language models from various providers.

## Overview

muxi-llm is a lightweight, provider-agnostic Python library that offers a unified interface for interacting with large language models (LLMs) from various providers. It simplifies the integration of LLMs into applications by providing a consistent API while abstracting away provider-specific implementation details.

The library follows the OpenAI client API design pattern, making it familiar to developers already using OpenAI and enabling easy migration for existing applications.

## Key Features

- **Provider-agnostic** - Support for multiple LLM providers through a single interface
- **OpenAI-compatible API** - Familiar interface for developers accustomed to OpenAI's client library
- **Streaming support** - Real-time streaming responses from supported providers
- **Multi-modal capabilities** - Support for text, images, and other modalities when available
- **Model naming convention** - Consistent `provider/model-name` format for clear attribution
- **Apache 2.0 license** - Permissive open-source license for broad adoption

## Supported Providers

### OpenAI-compatible API Providers

1. **OpenAI** - Base implementation and reference API
2. **Together AI** - Full compatibility for chat/completion/embedding
3. **Anyscale** - OpenAI-compatible endpoints
4. **Fireworks AI** - Drop-in replacement for OpenAI
5. **Groq** - Compatible API format
6. **Mistral AI** - La Plateforme offers OpenAI compatibility
7. **Azure OpenAI** - Microsoft's hosted version of OpenAI
8. **Perplexity AI** - Compatible chat completions
9. **DeepInfra** - OpenAI-compatible API
10. **Lepton AI** - Compatible endpoints
11. **OctoAI** - OpenAI-compatible for selected models
12. **Modal** - Hosts compatible endpoints
13. **Replicate** - OpenAI compatibility layer for various models
14. **NexusFlow** - OpenAI-compatible
15. **Voyage AI** - Compatible for embeddings
16. **Databricks** - MosaicML offers compatible endpoints
17. **OpenRouter** - OpenAI-compatible API for multiple providers

### Non-OpenAI-compatible Providers (Requiring Custom Adapters)

1. **Anthropic** - Has its own API structure (though moving closer to OpenAI compatibility)
2. **Ollama** - Custom API for local models
3. **HuggingFace** - Different API structure
4. **Sagemaker** - Custom API for embeddings/completions
5. **Cohere** - Unique API format

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
   - Configuration file support
   - Runtime configuration options

4. **Error Handling**
   - Standardized error types
   - Provider-specific error mapping

## API Design

muxi-llm mirrors the OpenAI Python client library API for familiarity:

### Chat Completions

```python
from muxi_llm import ChatCompletion

# Basic usage
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

# With images (multi-modal)
response = ChatCompletion.create(
    model="openai/gpt-4-vision",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
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

## Model Naming Convention

Models are specified using a provider prefix to clearly identify the source:

- `openai/gpt-4` - OpenAI's GPT-4
- `anthropic/claude-3-opus` - Anthropic's Claude model
- `groq/llama3-70b` - Llama 3 via Groq
- `ollama/llama3` - Local Llama model via Ollama
- `openrouter/anthropic/claude-3` - Claude via OpenRouter

## Configuration

muxi-llm can be configured through environment variables, a configuration file, or at runtime:

```python
# Environment variables
# MUXI_LLM_OPENAI_API_KEY=sk-...
# MUXI_LLM_ANTHROPIC_API_KEY=sk-...

# Configuration file (~/.muxi/config.yaml)
# llm:
#   providers:
#     openai:
#       api_key: sk-...
#     anthropic:
#       api_key: sk-...

# Runtime configuration
import muxi_llm

muxi_llm.api_key = "sk-..."  # Default provider (OpenAI)
muxi_llm.anthropic_api_key = "sk-..."  # Anthropic-specific

# Per-request configuration
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="sk-..."  # Override for this request only
)
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

### Phase 4: Advanced Features

- Token counting utilities
- Cost estimation
- Caching layer (optional)
- Performance optimizations

## Integration with MUXI Framework

muxi-llm will serve as a dependency for the core MUXI framework, replacing the current direct implementation of model providers. This will:

1. Simplify the core codebase
2. Provide more consistent model access
3. Allow independent evolution of the LLM interface
4. Make it easier to add support for new models

## License

muxi-llm is licensed under the Apache License 2.0 to encourage broad adoption and contributions.

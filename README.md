# OneLLM

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.python.org/pypi/onellm)
[![Version](https://img.shields.io/pypi/v/onellm.svg?maxAge=60)](https://pypi.python.org/pypi/onellm)
[![Status](https://img.shields.io/pypi/status/onellm.svg?maxAge=60)](https://pypi.python.org/pypi/onellm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Test Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)](https://github.com/muxi-ai/onellm)
&nbsp;
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/muxi-ai/onellm)

### A "drop-in" replacement for OpenAI's client that offers a unified interface for interacting with large language models from various providers,  with support for hundreds of models, built-in fallback mechanisms, and enhanced reliability features.

---

> [!IMPORTANT]
> #### Support this project by starring this repo on GitHub!
>
> More stars → more visibility → more contributors → better features → more robust tool for everyone 🎉
>
> [🌟 Star this repo on GitHub →](https://github.com/muxi-ai/onellm)
>
> Thank you for your support! 🙏

---

## 📚 Table of Contents

- [Overview](https://github.com/muxi-ai/onellm/blob/main/README.md#-overview)
- [Getting Started](https://github.com/muxi-ai/onellm/blob/main/README.md#-getting-started)
- [Key Features](https://github.com/muxi-ai/onellm/blob/main/README.md#-key-features)
- [Supported Providers](https://github.com/muxi-ai/onellm/blob/main/README.md#-supported-providers)
- [Architecture](https://github.com/muxi-ai/onellm/blob/main/README.md#-architecture)
- [API Design](https://github.com/muxi-ai/onellm/blob/main/README.md#-api-design)
- [Advanced Features](https://github.com/muxi-ai/onellm/blob/main/README.md#-advanced-features)
- [Migration from OpenAI](https://github.com/muxi-ai/onellm/blob/main/README.md#-migration-from-openai)
- [Model Naming Convention](https://github.com/muxi-ai/onellm/blob/main/README.md#-model-naming-convention)
- [Configuration](https://github.com/muxi-ai/onellm/blob/main/README.md#-configuration)
- [Test Coverage](https://github.com/muxi-ai/onellm/blob/main/README.md#-test-coverage)
- [Documentation](https://github.com/muxi-ai/onellm/blob/main/README.md#-documentation)
- [Call for Contributions](https://github.com/muxi-ai/onellm/blob/main/README.md#-call-for-contributions)
- [License](https://github.com/muxi-ai/onellm/blob/main/README.md#-license)
- [Acknowledgements](https://github.com/muxi-ai/onellm/blob/main/README.md#-acknowledgements)


## 👉 Overview

**OneLLM** is a lightweight, provider-agnostic Python library that offers a unified interface for interacting with large language models (LLMs) from various providers. It simplifies the integration of LLMs into applications by providing a consistent API while abstracting away provider-specific implementation details.

The library follows the OpenAI client API design pattern, making it familiar to developers already using OpenAI and enabling easy migration for existing applications. **Simply change your import statements and instantly gain access to hundreds of models** across dozens of providers while maintaining your existing code structure.

With support for 19 implemented providers (and more planned), OneLLM gives you access to approximately 300+ unique language models through a single, consistent interface - from the latest proprietary models to open-source alternatives, all accessible through familiar OpenAI-compatible patterns.

> [!NOTE]
> **Ready for Use**: OneLLM now supports 19 providers with 300+ models! From cloud APIs to local models, you can access them all through a single, unified interface. [Contributions are welcome](./CONTRIBUTING.md) to help add even more providers!

---

## 🚀 Getting Started

### Installation

```bash
# Basic installation (includes OpenAI compatibility and download utility)
pip install OneLLM

# For all providers (includes dependencies for future provider support)
pip install "OneLLM[all]"
```

### Download Models for Local Use

OneLLM includes a built-in utility for downloading GGUF models:

```bash
# Download a model from HuggingFace (saves to ~/llama_models by default)
onellm download --repo-id "TheBloke/Llama-2-7B-GGUF" --filename "llama-2-7b.Q4_K_M.gguf"

# Download to a custom directory
onellm download -r "microsoft/Phi-3-mini-4k-instruct-gguf" -f "Phi-3-mini-4k-instruct-q4.gguf" -o /path/to/models
```

### Quick Win: Your First LLM Call

```python
# Basic usage with OpenAI-compatible syntax
from onellm import ChatCompletion

response = ChatCompletion.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message["content"])
# Output: I'm doing well, thank you for asking! I'm here and ready to help you...
```

For more detailed examples, check out the [examples directory](./examples).

---

## ✨ Key Features


| Feature | Description |
|---------|-------------|
| **📦 Drop-in replacement** | Use your existing OpenAI code with minimal changes |
| **🔄 Provider-agnostic** | Support for 300+ models across 19 implemented providers |
| **🔁 Automatic fallback** | Seamlessly switch to alternative models when needed |
| **🔄 Auto-retry mechanism** | Retry the same model multiple times before failing |
| **🧩 OpenAI-compatible** | Familiar interface for developers used to OpenAI |
| **📺 Streaming support** | Real-time streaming responses from supported providers |
| **🖼️ Multi-modal capabilities** | Support for text, images, audio across compatible models |
| **🏠 Local LLM support** | Run models locally via Ollama and llama.cpp |
| **⬇️ Model downloads** | Built-in CLI to download GGUF models from HuggingFace |
| **🧹 Unicode artifact cleaning** | Automatic removal of invisible characters to prevent AI detection |
| **🏷️ Consistent naming** | Clear `provider/model-name` format for attribution |
| **🧪 Comprehensive tests** | 96% test coverage ensuring reliability |
| **📄 Apache-2.0 license** | Open-source license that protects contributions |

---

## 🌐 Supported Providers

OneLLM currently supports **19 providers** with more on the way:

### Cloud API Providers (17)
- **Anthropic** - Claude family of models
- **Anyscale** - Configurable AI platform
- **AWS Bedrock** - Access to multiple model families
- **Azure OpenAI** - Microsoft-hosted OpenAI models
- **Cohere** - Command models with RAG
- **DeepSeek** - Chinese LLM provider
- **Fireworks** - Fast inference platform
- **Moonshot** - Kimi models with long-context capabilities
- **Google AI Studio** - Gemini models via API key
- **Groq** - Ultra-fast inference for Llama, Mixtral
- **Mistral** - Mistral Large, Medium, Small
- **OpenAI** - GPT-4o, 3o-mini, DALL-E, Whisper, etc.
- **OpenRouter** - Gateway to 100+ models
- **Perplexity** - Search-augmented models
- **Together AI** - Open-source model hosting
- **Vertex AI** - Google Cloud's enterprise Gemini
- **X.AI** - Grok models

### Local Providers (2)
- **Ollama** - Run models locally with easy management
- **llama.cpp** - Direct GGUF model execution

### Notable Models Available

Through these providers, you gain access to hundreds of models, including:

<div align="center">

<!-- Model categories -->
<table>
  <tr>
    <th>Model Family</th>
    <th>Notable Models</th>
  </tr>
  <tr>
    <td><strong>OpenAI Family</strong></td>
    <td>GPT-4o, GPT-4 Turbo, o3</td>
  </tr>
  <tr>
    <td><strong>Claude Family</strong></td>
    <td>Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku</td>
  </tr>
  <tr>
    <td><strong>Llama Family</strong></td>
    <td>Llama 3 70B, Llama 3 8B, Code Llama</td>
  </tr>
  <tr>
    <td><strong>Mistral Family</strong></td>
    <td>Mistral Large, Mistral 7B, Mixtral</td>
  </tr>
  <tr>
    <td><strong>Gemini Family</strong></td>
    <td>Gemini Pro, Gemini Ultra, Gemini Flash</td>
  </tr>
  <tr>
    <td><strong>Embeddings</strong></td>
    <td>Ada-002, text-embedding-3-small/large, Cohere embeddings</td>
  </tr>
  <tr>
    <td><strong>Multimodal</strong></td>
    <td>GPT-4 Vision, Claude 3 Vision, Gemini Pro Vision</td>
  </tr>
</table>

</div>

---

## 🏗️ Architecture

OneLLM follows a modular architecture with clear separation of concerns:

```mermaid
---
config:
  look: handDrawn
  theme: mc
  themeVariables:
    background: 'transparent'
    primaryColor: '#fff0'
    secondaryColor: 'transparent'
    tertiaryColor: 'transparent'
    mainBkg: 'transparent'

  flowchart:
    layout: fixed
---
flowchart TD
    %% User API Layer
    User(User Application) --> ChatCompletion
    User --> Completion
    User --> Embedding
    User --> OpenAIClient["OpenAI Client Interface"]

    subgraph API["Public API Layer"]
        ChatCompletion["ChatCompletion\n.create() / .acreate()"]
        Completion["Completion\n.create() / .acreate()"]
        Embedding["Embedding\n.create() / .acreate()"]
        OpenAIClient
    end

    %% Core logic
    subgraph Core["Core Logic"]
        Router["Provider Router"]
        Config["Configuration\nEnvironment Variables\nAPI Keys"]
        FallbackManager["Fallback Manager"]
        RetryManager["Retry Manager"]
    end

    %% Provider Layer
    BaseProvider["Provider Interface<br>(Base Class)"]

    subgraph Implementations["Provider Implementations"]
        OpenAI["OpenAI"]
        Anthropic["Anthropic"]
        GoogleProvider["Google"]
        Groq["Groq"]
        Ollama["Local LLMs"]
        OtherProviders["20+ Others"]
    end

    %% Utilities
    subgraph Utilities["Utilities"]
        Streaming["Streaming<br>Handlers"]
        TokenCounting["Token<br>Counter"]
        ErrorHandling["Error<br>Handling"]
        Types["Type<br>Definitions"]
        Models["Response<br>Models"]
    end

    %% External services
    OpenAIAPI["OpenAI API"]
    AnthropicAPI["Anthropic API"]
    GoogleAPI["Google API"]
    GroqAPI["Groq API"]
    LocalModels["Ollama/llama.cpp"]
    OtherAPIs["..."]

    %% Connections
    ChatCompletion --> Router
    Completion --> Router
    Embedding --> Router
    OpenAIClient --> Router

    Router --> Config
    Router --> FallbackManager
    FallbackManager --> RetryManager

    RetryManager --> BaseProvider

    BaseProvider --> OpenAI
    BaseProvider --> Anthropic
    BaseProvider --> GoogleProvider
    BaseProvider --> Groq
    BaseProvider --> Ollama
    BaseProvider --> OtherProviders

    BaseProvider --> Streaming
    BaseProvider --> TokenCounting
    BaseProvider --> ErrorHandling

    OpenAI --> OpenAIAPI
    Anthropic --> AnthropicAPI
    GoogleProvider --> GoogleAPI
    Groq --> GroqAPI
    Ollama --> LocalModels
    OtherProviders --> OtherAPIs
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

---

## 🔌 API Design

OneLLM mirrors the OpenAI Python client library API for familiarity:

### Chat Completions

```python
from onellm import ChatCompletion

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
from onellm import Completion

response = Completion.create(
    model="groq/llama3-70b",
    prompt="Once upon a time",
    max_tokens=100
)
```

### Embeddings

```python
from onellm import Embedding

response = Embedding.create(
    model="openai/text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog"
)
```

---

## 🛠️ Advanced Features

### Fallback Chains for Enhanced Reliability

OneLLM includes built-in fallback support to handle API errors gracefully:

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

If the primary model fails due to service unavailability, rate limiting, or other retriable errors, OneLLM automatically tries the fallback models in sequence.

### Automatic Retries

For transient errors, you can configure OneLLM to retry the same model multiple times before falling back to alternatives:

```python
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    retries=3,  # Will try the same model up to 3 additional times if it fails
    fallback_models=["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"]
)
```

This is implemented using the fallback mechanism under the hood, making it both powerful and efficient.

### Fallback Chains + Automatic Retries Architecture

```mermaid
---
config:
  look: handDrawn
  theme: mc
  themeVariables:
    background: 'transparent'
    primaryColor: '#fff0'
    secondaryColor: 'transparent'
    tertiaryColor: 'transparent'
    mainBkg: 'transparent'

  flowchart:
    layout: fixed
---
flowchart TD
    START(["Client Request"]) --> REQUEST["Chat/Completion Request"]
    REQUEST --> PRIMARY["Primary Model<br>e.g., openai/gpt-4"]

    PRIMARY --> API_CHECK{"API<br>Available?"}
    API_CHECK -->|Yes| MODEL_CHECK{"Model<br>Available?"}
    MODEL_CHECK -->|Yes| QUOTA_CHECK{"Quota/Rate<br>Limits OK?"}
    QUOTA_CHECK -->|Yes| SUCCESS["Successful Response"]
    SUCCESS --> RESPONSE(["Return to Client"])

    API_CHECK -->|No| RETRY_DECISION{"Retry<br>Count < Max?"}
    MODEL_CHECK -->|No| RETRY_DECISION
    QUOTA_CHECK -->|No| RETRY_DECISION

    RETRY_DECISION -->|Yes| RETRY["Retry with Delay<br>(Same Model)"]
    RETRY --> PRIMARY

    RETRY_DECISION -->|No| FALLBACK_CHECK{"Fallbacks<br>Available?"}

    FALLBACK_CHECK -->|Yes| FALLBACK_MODEL["Next Fallback Model<br>e.g., anthropic/claude-3-haiku"]
    FALLBACK_MODEL --> FALLBACK_TRY["Try Fallback"]
    FALLBACK_TRY --> FALLBACK_API_CHECK{"API<br>Available?"}

    FALLBACK_API_CHECK -->|Yes| FALLBACK_SUCCESS["Successful Response"]
    FALLBACK_SUCCESS --> RESPONSE

    FALLBACK_API_CHECK -->|No| NEXT_FALLBACK{"More<br>Fallbacks?"}
    NEXT_FALLBACK -->|Yes| FALLBACK_MODEL
    NEXT_FALLBACK -->|No| ERROR["Error Response"]

    FALLBACK_CHECK -->|No| ERROR
    ERROR --> RESPONSE
```



### JSON Mode for Structured Outputs

For applications that require structured data, OneLLM supports JSON mode with compatible providers:

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

When using providers that don't natively support JSON mode, OneLLM automatically adds system instructions requesting JSON-formatted responses.

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

---

## 🔄 Migration from OpenAI

OneLLM provides multiple ways to migrate from the OpenAI client, including a fully compatible client interface:

### Option 1: Complete Drop-in Replacement (Identical Interface)

OneLLM is a complete drop-in replacement for the OpenAI client with the OpenAI library included as a default dependency:

```python
# Before
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello world"}]
)

# After - 100% identical client interface
from onellm import OpenAI  # or Client
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
from onellm import ChatCompletion
response = ChatCompletion.create(
    model="openai/gpt-4",  # explicitly using provider prefix
    messages=[{"role": "user", "content": "Hello world"}]
)
```

### Option 3: Model Fallback (Enhanced Reliability)

```python
# Adding fallback options with ChatCompletion
from onellm import ChatCompletion
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
from onellm import ChatCompletion
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
from onellm import OpenAI
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

---

## 🏷️ Model Naming Convention

Models are specified using a provider prefix to clearly identify the source:

<!-- Model naming examples -->
<table>
  <tr>
    <th>Provider</th>
    <th>Format</th>
    <th>Example</th>
  </tr>
  <tr>
    <td>OpenAI</td>
    <td><code>openai/{model}</code></td>
    <td><code>openai/gpt-4</code></td>
  </tr>
  <tr>
    <td>Google</td>
    <td><code>google/{model}</code></td>
    <td><code>google/gemini-pro</code></td>
  </tr>
  <tr>
    <td>Anthropic</td>
    <td><code>anthropic/{model}</code></td>
    <td><code>anthropic/claude-3-opus</code></td>
  </tr>
  <tr>
    <td>Groq</td>
    <td><code>groq/{model}</code></td>
    <td><code>groq/llama3-70b</code></td>
  </tr>
  <tr>
    <td>Mistral</td>
    <td><code>mistral/{model}</code></td>
    <td><code>mistral/mistral-large</code></td>
  </tr>
  <tr>
    <td>Ollama</td>
    <td><code>ollama/{model}@host:port</code></td>
    <td><code>ollama/llama3:8b@localhost:11434</code></td>
  </tr>
  <tr>
    <td>llama.cpp</td>
    <td><code>llama_cpp/{model.gguf}</code></td>
    <td><code>llama_cpp/llama-3-8b-q4_K_M.gguf</code></td>
  </tr>
  <tr>
    <td>XAI (Grok)</td>
    <td><code>xai/{model}</code></td>
    <td><code>xai/grok-beta</code></td>
  </tr>
  <tr>
    <td>Cohere</td>
    <td><code>cohere/{model}</code></td>
    <td><code>cohere/command-r-plus</code></td>
  </tr>
  <tr>
    <td>AWS Bedrock</td>
    <td><code>bedrock/{model}</code></td>
    <td><code>bedrock/claude-3-5-sonnet</code></td>
  </tr>
</table>

---

## ⚙️ Configuration

OneLLM can be configured through environment variables or at runtime:

```python
# Environment variables
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-...

# Runtime configuration
import onellm

onellm.openai_api_key = "sk-..."  # OpenAI API key
onellm.anthropic_api_key = "sk-..."  # Anthropic API key

# Configure fallback behavior
onellm.config.fallback = {
    "enabled": True,
    "default_chains": {
        "chat": ["openai/gpt-4", "anthropic/claude-3-opus", "groq/llama3-70b"],
        "embedding": ["openai/text-embedding-3-small", "cohere/embed-english"]
    },
    "retry_delay": 1.0,
    "max_retries": 3
}
```

---

## 🧪 Test Coverage

OneLLM maintains comprehensive test coverage to ensure reliability and compatibility:

<!-- Test coverage visualization -->
<table>
  <tr>
    <th>Coverage Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Overall Package Coverage</td>
    <td>96% (357 passing tests)</td>
  </tr>
  <tr>
    <td>Modules with 100% Coverage</td>
    <td>16 modules</td>
  </tr>
  <tr>
    <td>Minimum Module Coverage</td>
    <td>90%</td>
  </tr>
  <tr>
    <td>Provider Implementations</td>
    <td>93-95% coverage</td>
  </tr>
</table>


This extensive test coverage ensures reliable operation across all supported providers and models.

---

## 📖 Documentation

OneLLM uses a code-first documentation approach:

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

---

## 🤝 Call for Contributions

We're building something amazing with OneLLM, and we'd love your help to make it even better! There are many ways to contribute:

- **Code contributions**: Add new providers, enhance existing ones, or improve core functionality
- **Bug reports**: Help us identify and fix issues
- **Documentation**: Improve examples, clarify API usage, or fix typos
- **Feature requests**: Share your ideas for making OneLLM more powerful
- **Testing**: Help ensure reliability across different environments and use cases

**Getting started is easy:**

1. Check out our [open issues](https://github.com/muxi-ai/onellm/issues) for good first contributions
2. Fork the repository and create a feature branch
3. Make your improvements and run tests
4. Submit a pull request with a clear description of your changes

For complete details on contribution guidelines, code style, provider development, testing standards, and our contributor license agreement, please read our [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## 📄 License

OneLLM is licensed under the [Apache-2.0 license](./LICENSE).

### Why Apache 2.0?

I chose the Apache 2.0 license to make OneLLM easy to adopt, integrate, and build on. This license:

- Allows you to freely use, modify, and distribute the library in both open-source and proprietary software
- Encourages wide adoption by individuals, startups, and enterprises alike
- Includes a clear patent grant for legal peace of mind
- Enables flexible usage without the complexity of copyleft restrictions

Whether you’re building internal tools or commercial applications, Apache 2.0 gives you the freedom to use OneLLM however you need – no strings attached.

---

## 🌟 Acknowledgements

This project stands on the shoulders of many excellent open-source projects and would not be possible without the collaborative spirit of the developer community.

Special thanks to all the LLM providers whose APIs this library integrates with, and to the early adopters who tested the library and provided crucial feedback.

## 🙏 Thank You

Thank you for trying out OneLLM! Your interest and support mean a lot to this project. Whether you're using it in your applications, experimenting with different LLM providers, or just exploring the capabilities, your participation helps drive this project forward.

If you find OneLLM useful in your work:

- Consider starring the repository on GitHub
- Share your experiences or use cases with the community
- Let us know how we can make it better for your needs

Thank you for your support!

~ **Ran Aroussi**<br>
𝕏 / [@aroussi](https://x.com/aroussi)

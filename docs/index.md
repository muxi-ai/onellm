---
layout: home
title: Home
nav_order: 1
---

# OneLLM Documentation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.python.org/pypi/onellm)
[![Version](https://img.shields.io/pypi/v/onellm.svg?maxAge=60)](https://pypi.python.org/pypi/onellm)
[![Status](https://img.shields.io/pypi/status/onellm.svg?maxAge=60)](https://pypi.python.org/pypi/onellm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Test Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)](https://github.com/muxi-ai/onellm)
&nbsp;
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](./CONTRIBUTING.md)

### A "drop-in" replacement for OpenAI's client that offers a unified interface for interacting with large language models from various providers,  with support for hundreds of models, built-in fallback mechanisms, and enhanced reliability features.

---

> This project was created by [Ran Aroussi](https://x.com/aroussi), and is released under the [Apache 2.0 license](https://github.com/muxi-ai/onellm/blob/main/LICENSE).
>
> #### Support this project by [starring it on GitHub >](https://github.com/muxi-ai/onellm)
> More stars → more visibility → more contributors → better features → more robust tool for everyone 🎉

---

Welcome to the OneLLM documentation! OneLLM is a unified interface for 300+ LLMs across 19+ providers, designed as a drop-in replacement for the OpenAI Python client.

## 🚀 Get Started

- **[Installation]({% link installation.md %})** - Install OneLLM in seconds
- **[Quick Start]({% link quickstart.md %})** - Your first OneLLM script
- **[Configuration]({% link configuration.md %})** - Set up API keys and options
- **[Provider Setup]({% link providers/setup.md %})** - Configure your providers

## 🌟 Key Features

- **Drop-in OpenAI Replacement**: Use the same code with 300+ models
- **Unified Interface**: One API for all providers (OpenAI, Anthropic, Google, etc.)
- **Smart Fallbacks**: Automatic failover between providers
- **Type Safety**: Full type hints and IDE support
- **Async Support**: Native async/await capabilities
- **Provider Agnostic**: Switch models with just a string change

## 📖 Documentation

### Core Concepts
- [Architecture]({% link architecture.md %}) - How OneLLM works under the hood
- [Provider System]({% link providers/README.md %}) - Understanding providers and models
- [Error Handling]({% link error-handling.md %}) - Handling errors gracefully
- [Advanced Features]({% link advanced-features.md %}) - Fallbacks, retries, and more

### API Reference
- [Client API]({% link api/client.md %}) - OpenAI-compatible client interface
- [Chat Completions]({% link api/chat-completions.md %}) - Chat completion methods

### Providers
- [Available Providers]({% link providers/README.md %}) - List of supported providers
- [Provider Capabilities]({% link providers/capabilities.md %}) - Feature support matrix
- [Azure OpenAI]({% link providers/azure.md %}) - Azure-specific configuration
- [AWS Bedrock]({% link providers/bedrock.md %}) - Bedrock setup guide
- [Local Models]({% link providers/ollama.md %}) - Run models locally

## 💡 Example

```python
from onellm import ChatCompletion

# Basic usage (identical to OpenAI's client)
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message["content"])
```

## 🤝 Support

- [GitHub Issues](https://github.com/muxi-ai/onellm/issues) - Report bugs or request features
- [Discussions](https://github.com/muxi-ai/onellm/discussions) - Ask questions and share ideas

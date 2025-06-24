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

Welcome to the OneLLM documentation! OneLLM is a unified interface for 300+ LLMs across 18+ providers, designed as a drop-in replacement for the OpenAI Python client.

## 📚 Documentation Overview

### Getting Started

- [Installation](installation.md) - How to install OneLLM
- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Configuration](configuration.md) - Setting up API keys and options

### Core Concepts

- [Architecture](architecture.md) - How OneLLM works under the hood
- [Provider System](providers/README.md) - Understanding providers and models
- [Error Handling](error-handling.md) - Handling errors gracefully

### API Reference

- [Client API](api/client.md) - OpenAI-compatible client interface
- [Chat Completions](api/chat-completions.md) - Chat completion methods
- [Completions](api/completions.md) - Text completion methods
- [Embeddings](api/embeddings.md) - Embedding generation
- [Files](api/files.md) - File operations
- [Audio](api/audio.md) - Speech-to-text and text-to-speech
- [Images](api/images.md) - Image generation

### Providers

- [Provider List](providers/README.md) - All 18 supported providers
- [Provider Capabilities](providers/capabilities.md) - Feature comparison
- [Provider Setup](providers/setup.md) - Setting up each provider

### Guides

- [Migration Guide](guides/migration.md) - Migrating from OpenAI
- [Best Practices](guides/best-practices.md) - Tips and recommendations
- [Advanced Usage](guides/advanced.md) - Advanced features
- [Troubleshooting](guides/troubleshooting.md) - Common issues

### Examples

- [Basic Examples](examples/basic.md) - Simple usage examples
- [Provider Examples](examples/providers.md) - Provider-specific examples
- [Advanced Examples](examples/advanced.md) - Complex use cases

## 🚀 Quick Links

- **Installation**: `pip install onellm`
- **GitHub**: https://github.com/muxi-ai/onellm
- **PyPI**: https://pypi.org/project/onellm/

## 💡 Key Features

- **Drop-in Replacement**: Works exactly like the OpenAI client
- **18+ Providers**: OpenAI, Anthropic, Google, Mistral, and more
- **300+ Models**: Access to a vast ecosystem of LLMs
- **Unified Interface**: Same code works with all providers
- **Type Safety**: Full type hints and IDE support
- **Async Support**: Both sync and async operations
- **Local Models**: Support for Ollama and llama.cpp

## 📖 How to Use This Documentation

1. **New Users**: Start with [Installation](installation.md) and [Quick Start](quickstart.md)
2. **Migrating**: Check the [Migration Guide](guides/migration.md)
3. **API Reference**: Use the [API docs](api/client.md) for detailed method information
4. **Provider Setup**: See [Provider Setup](providers/setup.md) for configuration
5. **Examples**: Browse [Examples](examples/basic.md) for practical usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## 📝 License

OneLLM is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

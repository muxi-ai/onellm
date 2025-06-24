---
layout: default
title: Providers
nav_order: 7
has_children: true
---

# Providers

OneLLM supports 18 providers, giving you access to 300+ language models through a unified interface.

## Provider List

### 🚀 Major Providers

#### OpenAI
- **Models**: GPT-4o, GPT-4, GPT-3.5-Turbo
- **Features**: Function calling, JSON mode, vision, DALL-E, embeddings
- **Pricing**: Pay per token
- **Best for**: General purpose, production applications
- **Setup**: [OpenAI Setup Guide](setup.md#openai)

#### Anthropic
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- **Features**: 200K+ context, vision support
- **Pricing**: Pay per token
- **Best for**: Long context, careful reasoning
- **Setup**: [Anthropic Setup Guide](setup.md#anthropic)

#### Google AI Studio
- **Models**: Gemini 1.5 Pro/Flash, Gemini Pro
- **Features**: Multimodal, 1M+ context, JSON mode
- **Pricing**: Free tier available
- **Best for**: Multimodal tasks, long context
- **Setup**: [Google Setup Guide](setup.md#google)

#### Mistral
- **Models**: Mistral Large/Medium/Small, Mixtral
- **Features**: European hosting, function calling
- **Pricing**: Pay per token
- **Best for**: EU compliance, multilingual
- **Setup**: [Mistral Setup Guide](setup.md#mistral)

### ⚡ Fast Inference Providers

#### Groq
- **Models**: Llama 3, Mixtral, Gemma
- **Features**: Ultra-fast LPU inference, 10x faster
- **Pricing**: Pay per token
- **Best for**: Real-time applications, low latency
- **Setup**: [Groq Setup Guide](setup.md#groq)

#### Together AI
- **Models**: Llama, Mistral, CodeLlama, 50+ models
- **Features**: Open source models, custom fine-tunes
- **Pricing**: Simple per-token pricing
- **Best for**: Open source models, research
- **Setup**: [Together Setup Guide](setup.md#together)

#### Fireworks
- **Models**: Llama, Mixtral, Starcoder
- **Features**: Optimized inference, function calling
- **Pricing**: Competitive per-token
- **Best for**: Fast open model serving
- **Setup**: [Fireworks Setup Guide](setup.md#fireworks)

#### Anyscale
- **Models**: Llama, Mistral, CodeLlama
- **Features**: Ray integration, schema-based JSON
- **Pricing**: $1/million tokens flat rate
- **Best for**: Scale-out workloads
- **Setup**: [Anyscale Setup Guide](setup.md#anyscale)

### 🌐 Specialized Providers

#### X.AI (Grok)
- **Models**: Grok-2, Grok-1
- **Features**: 128K context window
- **Pricing**: Premium
- **Best for**: Large context, reasoning
- **Setup**: [X.AI Setup Guide](setup.md#xai)

#### Perplexity
- **Models**: Sonar models with web search
- **Features**: Real-time web access, citations
- **Pricing**: Pay per request
- **Best for**: Current information, research
- **Setup**: [Perplexity Setup Guide](setup.md#perplexity)

#### DeepSeek
- **Models**: DeepSeek Chat, DeepSeek Coder
- **Features**: Chinese/English bilingual
- **Pricing**: Competitive
- **Best for**: Chinese language, coding
- **Setup**: [DeepSeek Setup Guide](setup.md#deepseek)

#### Cohere
- **Models**: Command R/R+, Embed
- **Features**: RAG optimization, embeddings
- **Pricing**: Enterprise/startup plans
- **Best for**: Enterprise NLP, search
- **Setup**: [Cohere Setup Guide](setup.md#cohere)

### 🌍 Multi-Provider Gateways

#### OpenRouter
- **Models**: 100+ models from all providers
- **Features**: Unified billing, free models
- **Pricing**: Small markup on provider prices
- **Best for**: Model exploration, fallbacks
- **Setup**: [OpenRouter Setup Guide](setup.md#openrouter)

### ☁️ Enterprise Cloud

#### Azure OpenAI
- **Models**: GPT-4, GPT-3.5, DALL-E, Embeddings
- **Features**: Enterprise SLA, VNet integration
- **Pricing**: Same as OpenAI
- **Best for**: Enterprise, compliance
- **Setup**: [Azure Setup Guide](setup.md#azure)

#### AWS Bedrock
- **Models**: Claude, Llama, Titan, Stable Diffusion
- **Features**: AWS integration, multiple providers
- **Pricing**: Pay per use
- **Best for**: AWS ecosystem
- **Setup**: [Bedrock Setup Guide](setup.md#bedrock)

#### Google Vertex AI
- **Models**: Gemini, PaLM, Codey
- **Features**: MLOps platform, enterprise
- **Pricing**: Enterprise pricing
- **Best for**: GCP ecosystem
- **Setup**: [Vertex AI Setup Guide](setup.md#vertex)

### 💻 Local Providers

#### Ollama
- **Models**: Any GGUF model
- **Features**: Local hosting, model management
- **Pricing**: Free (self-hosted)
- **Best for**: Privacy, offline use
- **Setup**: [Ollama Setup Guide](setup.md#ollama)

#### llama.cpp
- **Models**: Any GGUF model
- **Features**: Direct inference, GPU support
- **Pricing**: Free (self-hosted)
- **Best for**: Maximum control, embedded
- **Setup**: [llama.cpp Setup Guide](setup.md#llama-cpp)

## Provider Comparison

### By Speed
1. **Groq** - Ultra-fast LPU (100+ tokens/sec)
2. **Fireworks** - Optimized inference
3. **Together** - Fast parallel inference
4. **OpenAI** - Reliable performance
5. **Local** - Depends on hardware

### By Context Length
1. **Google Gemini 1.5** - 1M+ tokens
2. **Anthropic Claude** - 200K tokens
3. **X.AI Grok** - 128K tokens
4. **Perplexity** - 128K tokens
5. **OpenAI GPT-4** - 128K tokens

### By Price (Lowest to Highest)
1. **Local** (Ollama/llama.cpp) - Free
2. **Anyscale** - $1/M tokens flat
3. **Together/Fireworks** - Competitive
4. **OpenRouter** - Various options
5. **OpenAI/Anthropic** - Premium

### By Features
- **Function Calling**: OpenAI, Mistral, Groq, Anyscale
- **Vision**: OpenAI, Anthropic, Google, Vertex AI
- **Web Search**: Perplexity
- **JSON Mode**: OpenAI, Google, Mistral, Groq
- **Embeddings**: OpenAI, Cohere, Google, Bedrock

## Model Naming

All models use the format `provider/model-name`:

```python
# Examples
"openai/gpt-4o-mini"
"anthropic/claude-3-5-sonnet-20241022"
"google/gemini-1.5-flash"
"groq/llama3-70b-8192"
"together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
```

## Quick Start

```python
from onellm import OpenAI

# Client works with all providers
client = OpenAI()

# Use any provider by changing model name
response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20241022",  # Just change this
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Choosing a Provider

### For Production
- **OpenAI**: Most reliable, best ecosystem
- **Anthropic**: Best for complex reasoning
- **Azure OpenAI**: Enterprise requirements

### For Speed
- **Groq**: Ultra-fast responses
- **Fireworks**: Fast and affordable
- **Local**: No network latency

### For Cost
- **Local**: Free (your hardware)
- **Anyscale**: Predictable pricing
- **OpenRouter**: Access to free models

### For Privacy
- **Ollama**: Fully local
- **llama.cpp**: Complete control
- **Azure/Vertex**: Enterprise privacy

## Next Steps

- [Provider Setup](setup.md) - Detailed setup instructions
- [Provider Capabilities](capabilities.md) - Feature comparison matrix
- [Examples](../examples/providers.md) - Provider-specific examples
- [Best Practices](../guides/best-practices.md) - Choosing providers
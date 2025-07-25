---
layout: default
title: Providers
nav_order: 7
has_children: true
---

# Providers

OneLLM supports 19 providers, giving you access to 300+ language models through a unified interface.

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

#### Moonshot
- **Models**: Kimi K2, moonshot-v1-8k/32k/128k
- **Features**: Long-context (200K+ tokens), Chinese/English bilingual, vision support
- **Pricing**: Cost-effective (~5x cheaper than Claude/Gemini)
- **Best for**: Long-context processing, Chinese language, document analysis
- **Setup**: [Moonshot Setup Guide](setup.md#moonshot)

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
2. **Moonshot Kimi** - 200K tokens
3. **Anthropic Claude** - 200K tokens
4. **X.AI Grok** - 128K tokens
5. **Perplexity** - 128K tokens
6. **OpenAI GPT-4** - 128K tokens

### By Price (Lowest to Highest)
1. **Local** (Ollama/llama.cpp) - Free
2. **Anyscale** - $1/M tokens flat
3. **Together/Fireworks** - Competitive
4. **OpenRouter** - Various options
5. **OpenAI/Anthropic** - Premium

### By Features
- **Function Calling**: OpenAI, Mistral, Groq, Anyscale, Moonshot
- **Vision**: OpenAI, Anthropic, Google, Vertex AI, Moonshot
- **Web Search**: Perplexity
- **JSON Mode**: OpenAI, Google, Mistral, Groq, Moonshot
- **Embeddings**: OpenAI, Cohere, Google, Bedrock

## Model Naming Convention

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
  <tr>
    <td>Moonshot</td>
    <td><code>moonshot/{model}</code></td>
    <td><code>moonshot/moonshot-v1-8k</code></td>
  </tr>
</table>

### Additional Examples

```python
# Standard models
"openai/gpt-4o-mini"
"anthropic/claude-3-5-sonnet-20241022"
"google/gemini-1.5-flash"
"groq/llama3-70b-8192"
"moonshot/moonshot-v1-8k"

# Models with organization prefixes
"together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
"fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct"

# Local models
"ollama/llama3:latest"
"llama_cpp/models/llama-3-8b-instruct.Q4_K_M.gguf"
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
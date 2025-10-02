---
layout: default
title: Providers
nav_order: 7
has_children: true
---

# Providers

OneLLM supports 21 providers, giving you access to 300+ language models through a unified interface.

## Provider List

### 🚀 Major Providers

#### OpenAI
- **Models**:
  - GPT-5 family: `gpt-5`, `gpt-5-pro`, `gpt-5-mini`, `gpt-5-nano`
  - GPT-4 family: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-4-turbo-preview`
  - GPT-3.5: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
  - O-series (reasoning): `o1`, `o1-preview`, `o1-mini`, `o3`, `o3-mini`
  - Embeddings: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Features**: Function calling, JSON mode, vision, DALL-E, embeddings
- **Pricing**: Pay per token
- **Best for**: General purpose, production applications
- **Setup**: [OpenAI Setup Guide](setup.md#openai)

#### Anthropic
- **Models**:
  - Claude 4 family: `claude-sonnet-4.5`, `claude-opus-4.1`, `claude-sonnet-4`, `claude-opus-4`
  - Claude 3.5: `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-20240620`
  - Claude 3: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
  - Legacy: `claude-2.1`, `claude-2.0`, `claude-instant-1.2`
- **Features**: 200K+ context, vision support
- **Pricing**: Pay per token
- **Best for**: Long context, careful reasoning
- **Setup**: [Anthropic Setup Guide](setup.md#anthropic)

#### Google AI Studio
- **Models**:
  - Gemini 2.5: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-flash-image`
  - Gemini 1.5: `gemini-1.5-pro`, `gemini-1.5-pro-latest`, `gemini-1.5-flash`, `gemini-1.5-flash-latest`
  - Gemini 1.0: `gemini-pro`, `gemini-pro-vision`
  - Embeddings: `text-embedding-004`, `embedding-001`
- **Features**: Multimodal, 1M+ context, JSON mode
- **Pricing**: Free tier available
- **Best for**: Multimodal tasks, long context
- **Setup**: [Google Setup Guide](setup.md#google)

#### Mistral
- **Models**:
  - Latest: `mistral-large-latest`, `mistral-medium-latest`, `mistral-small-latest`
  - Specialized: `codestral` (code), `pixtral` (vision), `devstral` (development), `voxtral` (voice), `ministral` (lightweight)
  - Mixtral: `mixtral-8x7b`, `mixtral-8x22b`
  - Legacy: `mistral-tiny`, `open-mistral-7b`
- **Features**: European hosting, function calling
- **Pricing**: Pay per token
- **Best for**: EU compliance, multilingual
- **Setup**: [Mistral Setup Guide](setup.md#mistral)

### ⚡ Fast Inference Providers

#### Groq
- **Models**:
  - Llama 3: `llama3-70b-8192`, `llama3-8b-8192`, `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`
  - Mixtral: `mixtral-8x7b-32768`
  - Gemma: `gemma-7b-it`, `gemma2-9b-it`
  - Llama Guard: `llama-guard-3-8b` (content moderation)
- **Features**: Ultra-fast LPU inference, 10x faster
- **Pricing**: Pay per token
- **Best for**: Real-time applications, low latency
- **Setup**: [Groq Setup Guide](setup.md#groq)

#### Together AI
- **Models**:
  - Llama: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
  - Mixtral: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-Instruct-v0.1`
  - Qwen: `Qwen/Qwen2.5-72B-Instruct-Turbo`, `Qwen/Qwen2.5-7B-Instruct-Turbo`
  - DeepSeek: `deepseek-ai/deepseek-llm-67b-chat`
  - CodeLlama: `codellama/CodeLlama-34b-Instruct-hf`
  - 50+ other open-source models
- **Features**: Open source models, custom fine-tunes
- **Pricing**: Simple per-token pricing
- **Best for**: Open source models, research
- **Setup**: [Together Setup Guide](setup.md#together)

#### Fireworks
- **Models**:
  - Llama: `accounts/fireworks/models/llama-v3p1-70b-instruct`, `accounts/fireworks/models/llama-v3p1-8b-instruct`
  - Mixtral: `accounts/fireworks/models/mixtral-8x7b-instruct`, `accounts/fireworks/models/mixtral-8x22b-instruct`
  - Qwen: `accounts/fireworks/models/qwen2p5-72b-instruct`
  - Deepseek: `accounts/fireworks/models/deepseek-v3`
  - StarCoder: `accounts/fireworks/models/starcoder-16b`
- **Features**: Optimized inference, function calling
- **Pricing**: Competitive per-token
- **Best for**: Fast open model serving
- **Setup**: [Fireworks Setup Guide](setup.md#fireworks)

#### Anyscale
- **Models**:
  - Llama: `meta-llama/Meta-Llama-3.1-70B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - Mixtral: `mistralai/Mixtral-8x7B-Instruct-v0.1`
  - Qwen: `Qwen/Qwen2.5-72B-Instruct`
  - Gemma: `google/gemma-2-9b-it`
- **Features**: Ray integration, schema-based JSON
- **Pricing**: $1/million tokens flat rate
- **Best for**: Scale-out workloads
- **Setup**: [Anyscale Setup Guide](setup.md#anyscale)

### 🌐 Specialized Providers

#### X.AI (Grok)
- **Models**:
  - Latest: `grok-2-latest`, `grok-2-1212`, `grok-2-vision-1212`
  - Grok 2: `grok-2-public`, `grok-2-mini`
  - Legacy: `grok-1`, `grok-beta`
- **Features**: 128K context window
- **Pricing**: Premium
- **Best for**: Large context, reasoning
- **Setup**: [X.AI Setup Guide](setup.md#xai)

#### Perplexity
- **Models**:
  - Sonar (online): `llama-3.1-sonar-small-128k-online`, `llama-3.1-sonar-large-128k-online`, `llama-3.1-sonar-huge-128k-online`
  - Sonar (chat): `llama-3.1-sonar-small-128k-chat`, `llama-3.1-sonar-large-128k-chat`
  - Sonar Pro: `sonar-pro` (advanced search)
- **Features**: Real-time web access, citations
- **Pricing**: Pay per request
- **Best for**: Current information, research
- **Setup**: [Perplexity Setup Guide](setup.md#perplexity)

#### DeepSeek
- **Models**:
  - Latest: `deepseek-chat`, `deepseek-reasoner`
  - Specialized: `deepseek-coder` (coding tasks)
  - Legacy: `deepseek-llm-67b-chat`
- **Features**: Chinese/English bilingual
- **Pricing**: Competitive
- **Best for**: Chinese language, coding
- **Setup**: [DeepSeek Setup Guide](setup.md#deepseek)

#### Moonshot
- **Models**:
  - Kimi: `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
  - Latest: `kimi-k2-0711-preview` (preview)
  - Vision: `kimi-vl` (multimodal)
  - Audio: `kimi-audio` (voice input)
- **Features**: Long-context (200K+ tokens), Chinese/English bilingual, vision support
- **Pricing**: Cost-effective (~5x cheaper than Claude/Gemini)
- **Best for**: Long-context processing, Chinese language, document analysis
- **Setup**: [Moonshot Setup Guide](setup.md#moonshot)

#### GLM (Zhipu AI)
- **Models**:
  - GLM-4: `glm-4`, `glm-4-plus`, `glm-4-air`, `glm-4-flash`
  - GLM-4V: `glm-4v` (vision support)
  - Legacy: `glm-3-turbo`
- **Features**: Chinese/English bilingual, streaming, function calling, vision
- **Pricing**: Competitive
- **Best for**: Chinese language tasks, cost-effective inference
- **Setup**: [GLM Setup Guide](setup.md#glm)

#### Cohere
- **Models**:
  - Command: `command-r-plus`, `command-r`, `command`, `command-light`
  - Embeddings: `embed-english-v3.0`, `embed-multilingual-v3.0`, `embed-english-light-v3.0`
- **Features**: RAG optimization, embeddings
- **Pricing**: Enterprise/startup plans
- **Best for**: Enterprise NLP, search
- **Setup**: [Cohere Setup Guide](setup.md#cohere)

### 🌍 Multi-Provider Gateways

#### OpenRouter
- **Models**:
  - Access 100+ models using `openrouter/{provider}/{model}` format
  - Free models: `meta-llama/llama-3.2-3b-instruct:free`, `google/gemma-2-9b-it:free`
  - Premium: `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `google/gemini-2.5-pro-exp`
- **Features**: Unified billing, free models
- **Pricing**: Small markup on provider prices
- **Best for**: Model exploration, fallbacks
- **Setup**: [OpenRouter Setup Guide](setup.md#openrouter)

#### Vercel AI Gateway
- **Models**:
  - Access 100+ models using `vercel/{provider}/{model}` format
  - OpenAI: `vercel/openai/gpt-4o-mini`, `vercel/openai/gpt-4o`
  - Anthropic: `vercel/anthropic/claude-sonnet-4`, `vercel/anthropic/claude-opus-4`
  - Google: `vercel/google/gemini-2.5-pro`, `vercel/google/gemini-2.5-flash`
  - Meta: `vercel/meta/llama-3.1-70b-instruct`
  - Many more providers and models
- **Features**: Unified billing, streaming, function calling, vision
- **Pricing**: Provider passthrough with optional markup
- **Best for**: Production deployments, unified billing
- **Setup**: [Vercel Setup Guide](setup.md#vercel)

### ☁️ Enterprise Cloud

#### Azure OpenAI
- **Models**:
  - GPT-4: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
  - GPT-3.5: `gpt-35-turbo`, `gpt-35-turbo-16k`
  - Embeddings: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
  - DALL-E: `dall-e-3`, `dall-e-2`
- **Features**: Enterprise SLA, VNet integration
- **Pricing**: Same as OpenAI
- **Best for**: Enterprise, compliance
- **Setup**: [Azure Setup Guide](setup.md#azure)

#### AWS Bedrock
- **Models**:
  - Anthropic: `anthropic.claude-3-5-sonnet-20241022-v2:0`, `anthropic.claude-3-opus-20240229-v1:0`
  - Meta: `meta.llama3-1-70b-instruct-v1:0`, `meta.llama3-1-8b-instruct-v1:0`
  - Amazon: `amazon.titan-text-premier-v1:0`, `amazon.titan-embed-text-v2:0`
  - Cohere: `cohere.command-r-plus-v1:0`, `cohere.embed-english-v3`
  - Mistral: `mistral.mistral-large-2407-v1:0`
- **Features**: AWS integration, multiple providers
- **Pricing**: Pay per use
- **Best for**: AWS ecosystem
- **Setup**: [Bedrock Setup Guide](setup.md#bedrock)

#### Google Vertex AI
- **Models**:
  - Gemini: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`
  - Legacy: `gemini-pro`, `gemini-pro-vision`
  - Embeddings: `text-embedding-004`, `textembedding-gecko@003`
- **Features**: MLOps platform, enterprise
- **Pricing**: Enterprise pricing
- **Best for**: GCP ecosystem
- **Setup**: [Vertex AI Setup Guide](setup.md#vertex)

### 💻 Local Providers

#### Ollama
- **Models**:
  - Popular: `llama3`, `llama3.1`, `mistral`, `mixtral`, `gemma2`, `qwen2.5`
  - Code: `codellama`, `deepseek-coder-v2`, `starcoder2`
  - Vision: `llava`, `llava-phi3`, `bakllava`
  - Specialized: `dolphin-mixtral`, `wizardlm2`, `phi3`
  - Any model from [ollama.com/library](https://ollama.com/library)
- **Features**: Local hosting, model management
- **Pricing**: Free (self-hosted)
- **Best for**: Privacy, offline use
- **Setup**: [Ollama Setup Guide](setup.md#ollama)

#### llama.cpp
- **Models**:
  - Any GGUF model from HuggingFace
  - Llama: `llama-3-8b-instruct.Q4_K_M.gguf`, `llama-3.1-70b-instruct.Q4_K_M.gguf`
  - Mistral: `mistral-7b-instruct.Q4_K_M.gguf`
  - Quantization levels: Q4_K_M (recommended), Q5_K_M, Q8_0, etc.
  - Use `onellm download <model>` to fetch models
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
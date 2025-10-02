---
layout: default
title: Provider Setup
parent: Providers
nav_order: 1
---

# Provider Setup Guide

Detailed setup instructions for each provider supported by OneLLM.

## Table of Contents

- [OpenAI](#openai)
- [Anthropic](#anthropic)
- [Google AI Studio](#google)
- [Mistral](#mistral)
- [Groq](#groq)
- [Together AI](#together)
- [Fireworks](#fireworks)
- [Anyscale](#anyscale)
- [X.AI](#xai)
- [Perplexity](#perplexity)
- [DeepSeek](#deepseek)
- [Moonshot](#moonshot)
- [GLM (Zhipu AI)](#glm)
- [Cohere](#cohere)
- [OpenRouter](#openrouter)
- [Vercel AI Gateway](#vercel)
- [Azure OpenAI](#azure)
- [AWS Bedrock](#bedrock)
- [Google Vertex AI](#vertex)
- [Ollama](#ollama)
- [llama.cpp](#llama-cpp)

---

## OpenAI

### 1. Get API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create new secret key

### 2. Set Environment Variable
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Test Connection
```python
from onellm import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message['content'])
```

### Available Models
- `openai/gpt-4o` - Most capable
- `openai/gpt-4o-mini` - Faster, cheaper
- `openai/gpt-4-turbo` - Previous generation
- `openai/gpt-3.5-turbo` - Legacy, fast

---

## Anthropic

### 1. Get API Key
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up for access
3. Navigate to API Keys
4. Generate key

### 2. Set Environment Variable
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models
- `anthropic/claude-3-5-sonnet-20241022` - Latest, most capable
- `anthropic/claude-3-opus-20240229` - Powerful, expensive
- `anthropic/claude-3-sonnet-20240229` - Balanced
- `anthropic/claude-3-haiku-20240307` - Fast, cheap

---

## Google AI Studio

### 1. Get API Key
1. Go to [makersuite.google.com](https://makersuite.google.com)
2. Sign in with Google account
3. Get API key from settings

### 2. Set Environment Variable
```bash
export GOOGLE_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="google/gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models
- `google/gemini-1.5-pro` - Most capable, 1M context
- `google/gemini-1.5-flash` - Fast, efficient
- `google/gemini-pro` - Previous generation

---

## Mistral

### 1. Get API Key
1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Create account
3. Navigate to API Keys
4. Create new key

### 2. Set Environment Variable
```bash
export MISTRAL_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="mistral/mistral-small-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models
- `mistral/mistral-large-latest` - Most capable
- `mistral/mistral-medium-latest` - Balanced
- `mistral/mistral-small-latest` - Fast
- `mistral/mistral-tiny` - Fastest

---

## Groq

### 1. Get API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for access
3. Get API key from dashboard

### 2. Set Environment Variable
```bash
export GROQ_API_KEY="gsk_..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="groq/llama3-70b-8192",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models
- `groq/llama3-70b-8192` - Llama 3 70B
- `groq/llama3-8b-8192` - Llama 3 8B
- `groq/mixtral-8x7b-32768` - Mixtral MoE
- `groq/gemma-7b-it` - Google Gemma

---

## Together AI

### 1. Get API Key
1. Go to [api.together.xyz](https://api.together.xyz)
2. Sign up
3. Get API key from account

### 2. Set Environment Variable
```bash
export TOGETHER_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models
- Various Llama models
- Mistral models
- CodeLlama models
- 50+ open source models

---

## Fireworks

### 1. Get API Key
1. Go to [app.fireworks.ai](https://app.fireworks.ai)
2. Create account
3. Get API key

### 2. Set Environment Variable
```bash
export FIREWORKS_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Anyscale

### 1. Get API Key
1. Go to [anyscale.com](https://www.anyscale.com)
2. Sign up for Endpoints
3. Get API key (starts with `esecret_`)

### 2. Set Environment Variable
```bash
export ANYSCALE_API_KEY="esecret_..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="anyscale/llama3-70b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## X.AI

### 1. Get API Key
1. Go to [x.ai](https://x.ai)
2. Request access
3. Get API key when approved

### 2. Set Environment Variable
```bash
export XAI_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="xai/grok-2-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Perplexity

### 1. Get API Key
1. Go to [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
2. Sign up
3. Generate API key

### 2. Set Environment Variable
```bash
export PERPLEXITY_API_KEY="pplx-..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="perplexity/llama-3.1-sonar-small-128k-online",
    messages=[{"role": "user", "content": "What's happening today?"}]
)
```

---

## DeepSeek

### 1. Get API Key
1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Register account
3. Get API key

### 2. Set Environment Variable
```bash
export DEEPSEEK_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="deepseek/deepseek-chat",
    messages=[{"role": "user", "content": "你好!"}]
)
```

---

## Moonshot

### 1. Get API Key
1. Go to [platform.moonshot.ai](https://platform.moonshot.ai)
2. Register account  
3. Get API key from console

### 2. Set Environment Variable
```bash
export MOONSHOT_API_KEY="sk-..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="moonshot/moonshot-v1-8k",
    messages=[{"role": "user", "content": "你好!"}]
)
```

### 4. Available Models
- `moonshot-v1-8k` - 8K context window
- `moonshot-v1-32k` - 32K context window  
- `moonshot-v1-128k` - 128K context window
- `kimi-k2-0711-preview` - Latest K2 model (preview)

### 5. Features
- **Long Context**: Up to 200K tokens
- **Multilingual**: Strong Chinese/English support
- **Vision**: Kimi-VL model supports images
- **Audio**: Kimi-Audio model supports audio input
- **Cost-effective**: ~5x cheaper than Claude/Gemini

---

## GLM (Zhipu AI) {#glm}

### 1. Get API Key
1. Go to [open.bigmodel.cn](https://open.bigmodel.cn)
2. Register account
3. Navigate to API Keys section
4. Create new API key

### 2. Set Environment Variable
```bash
export GLM_API_KEY="..."
# Or alternatively:
export ZAI_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="glm/glm-4",
    messages=[{"role": "user", "content": "你好!"}]
)
```

### 4. Available Models
- `glm-4` - Latest GLM-4 model
- `glm-4-plus` - Enhanced version
- `glm-4-air` - Lightweight version
- `glm-4-flash` - Fastest version
- `glm-4v` - Vision support

### 5. Features
- **Bilingual**: Strong Chinese and English support
- **Vision**: GLM-4V supports image understanding
- **Function Calling**: Tool use capabilities
- **Streaming**: Real-time response streaming
- **Cost-effective**: Competitive pricing for Chinese market

---

## Cohere {#cohere}

### 1. Get API Key
1. Go to [dashboard.cohere.com](https://dashboard.cohere.com)
2. Sign up
3. Get API key

### 2. Set Environment Variable
```bash
export COHERE_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="cohere/command-r-plus",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## OpenRouter

### 1. Get API Key
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up
3. Get API key from dashboard

### 2. Set Environment Variable
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Test Connection
```python
# Access 100+ models
response = client.chat.completions.create(
    model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Vercel AI Gateway {#vercel}

### 1. Get API Key
1. Go to [vercel.com/ai-gateway](https://vercel.com/ai-gateway)
2. Sign up or log in to Vercel
3. Navigate to AI Gateway dashboard
4. Create new API key

### 2. Set Environment Variable
```bash
export VERCEL_AI_API_KEY="..."
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="vercel/openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 4. Available Models
Use the format `vercel/{vendor}/{model}`:
- OpenAI: `vercel/openai/gpt-4o-mini`, `vercel/openai/gpt-4o`
- Anthropic: `vercel/anthropic/claude-sonnet-4`, `vercel/anthropic/claude-opus-4`
- Google: `vercel/google/gemini-2.5-pro`, `vercel/google/gemini-2.5-flash`
- Meta: `vercel/meta/llama-3.1-70b-instruct`, `vercel/meta/llama-3.1-8b-instruct`
- xAI: `vercel/xai/grok-2-latest`
- Mistral: `vercel/mistral/mistral-large-latest`
- DeepSeek: `vercel/deepseek/deepseek-chat`
- Many more providers and models

### 5. Features
- **Multi-Provider Gateway**: Access 100+ models from multiple providers
- **Unified Billing**: Single bill for all model usage
- **Streaming**: Real-time response streaming
- **Function Calling**: Tool use support for compatible models
- **Vision**: Multimodal capabilities for supported models
- **Production Ready**: Built for scale with Vercel's infrastructure

---

## Azure OpenAI {#azure}

### 1. Setup Azure Resources
1. Create Azure account
2. Create OpenAI resource
3. Deploy models
4. Get endpoint and key

### 2. Create Configuration File
Create `azure.json`:
```json
{
    "endpoint": "https://your-resource.openai.azure.com",
    "api_key": "your-key",
    "api_version": "2024-02-01",
    "deployments": {
        "gpt-4": "your-gpt4-deployment-name",
        "gpt-35-turbo": "your-gpt35-deployment-name"
    }
}
```

### 3. Test Connection
```python
client = OpenAI(azure_config_path="azure.json")
response = client.chat.completions.create(
    model="azure/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## AWS Bedrock

### 1. Setup AWS
1. Create AWS account
2. Enable Bedrock models
3. Configure IAM permissions

### 2. Create Configuration File
Create `bedrock.json`:
```json
{
    "region": "us-east-1",
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "..."
}
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="bedrock/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Google Vertex AI

### 1. Setup GCP
1. Create GCP project
2. Enable Vertex AI API
3. Create service account
4. Download credentials JSON

### 2. Set Environment Variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### 3. Test Connection
```python
response = client.chat.completions.create(
    model="vertexai/gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Ollama

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from ollama.com
```

### 2. Pull Models
```bash
ollama pull llama3
ollama pull mistral
```

### 3. Test Connection
```python
# Default localhost
response = client.chat.completions.create(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Custom endpoint
response = client.chat.completions.create(
    model="ollama/llama3@192.168.1.100:11434",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## llama.cpp

### 1. Install Dependencies
```bash
pip install llama-cpp-python
```

### 2. Download Models
```bash
# Using OneLLM's utility
onellm download llama-3-8b

# Or manually download GGUF files
wget https://huggingface.co/model.gguf
```

### 3. Set Model Directory
```bash
export LLAMA_CPP_MODEL_DIR="/path/to/models"
```

### 4. Test Connection
```python
response = client.chat.completions.create(
    model="llama_cpp/llama-3-8b-instruct.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### GPU Acceleration
```python
# Configure GPU layers
client = OpenAI(
    llama_cpp_config={
        "n_gpu_layers": 35,  # Number of layers on GPU
        "n_ctx": 4096        # Context window
    }
)
```

---

## Troubleshooting

### API Key Issues
- Ensure key is set correctly
- Check for extra spaces/quotes
- Verify key hasn't expired
- Try regenerating key

### Connection Issues
- Check internet connection
- Verify firewall settings
- Try different regions (if applicable)
- Check service status pages

### Model Access
- Ensure model is available in your region
- Check if model requires special access
- Verify billing is set up
- Check quota limits

## Next Steps

- [Provider Capabilities]({% link providers/capabilities.md %}) - Compare features
- [Examples]({% link examples/providers.md %}) - Provider examples
- [Configuration]({% link configuration.md %}) - Advanced config

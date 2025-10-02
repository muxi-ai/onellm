---
layout: default
title: Capabilities
parent: Providers
nav_order: 2
---

# Provider Capabilities

Comprehensive comparison of features and capabilities across all OneLLM providers.

## Feature Matrix

| Provider | Chat | Stream | Functions | JSON | Vision | Audio | Embed | Search | Local |
|----------|------|--------|-----------|------|--------|-------|--------|---------|-------|
| **OpenAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Anthropic** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Google** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Mistral** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Groq** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Together** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fireworks** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Anyscale** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **X.AI** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Perplexity** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **DeepSeek** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Cohere** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **OpenRouter** | ✅ | ✅ | Varies | Varies | Varies | ❌ | Varies | ❌ | ❌ |
| **Azure** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Bedrock** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Vertex AI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Ollama** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **llama.cpp** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Legend
- ✅ Supported
- ❌ Not Supported
- Varies: Depends on underlying model

## Notable Models by Category

Through these providers, you gain access to hundreds of models across different categories:

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

## Detailed Capabilities

### Chat Completions
All providers support basic chat completions with:
- System messages
- Multi-turn conversations
- Temperature control
- Max tokens limit

### Streaming
All providers support streaming responses for better UX.

### Function Calling
Providers with function calling support:
- **OpenAI**: Full support with parallel calls
- **Mistral**: Full support
- **Groq**: Basic support
- **Together**: Basic support
- **Fireworks**: Basic support
- **Anyscale**: Single function calls only
- **Azure**: Full support (same as OpenAI)
- **Vertex AI**: Full support

### JSON Mode
Force structured JSON output:
- **OpenAI**: `response_format={"type": "json_object"}`
- **Anthropic**: Via prompting
- **Google**: Native support
- **Mistral**: Native support
- **Groq**: Native support
- **Anyscale**: With schema specification

### Vision/Multimodal
Process images alongside text:
- **OpenAI**: GPT-4V models
- **Anthropic**: Claude 3 models
- **Google**: All Gemini models
- **Azure**: GPT-4V deployments
- **Bedrock**: Claude 3, select others
- **Vertex AI**: Gemini models

### Audio Processing
- **OpenAI**: Whisper (transcription), TTS
- **Google**: Speech services
- **Azure**: Full audio support
- **Vertex AI**: Speech services

### Embeddings
Generate text embeddings:
- **OpenAI**: text-embedding-3-small/large
- **Google**: text-embedding models
- **Mistral**: mistral-embed
- **Cohere**: embed-v3 models
- **Azure**: OpenAI embeddings
- **Bedrock**: Titan, Cohere embeddings
- **Vertex AI**: text-embedding models

### Web Search
Real-time internet access:
- **Perplexity**: All Sonar models with "online" suffix

### Local Execution
Run models on your hardware:
- **Ollama**: Model management included
- **llama.cpp**: Direct GGUF execution

## Context Windows

| Provider | Model | Max Context |
|----------|-------|-------------|
| Google | Gemini 1.5 Pro | 2,000,000 tokens |
| Google | Gemini 1.5 Flash | 1,000,000 tokens |
| Anthropic | Claude 3 | 200,000 tokens |
| X.AI | Grok-2 | 128,000 tokens |
| Perplexity | Sonar models | 128,000 tokens |
| OpenAI | GPT-4 Turbo | 128,000 tokens |
| Mistral | Large | 32,000 tokens |
| Groq | Mixtral | 32,768 tokens |
| Most others | - | 4,096-16,384 tokens |

## Performance Characteristics

### Response Speed (First Token)
1. **Groq**: <100ms (LPU acceleration)
2. **Fireworks**: ~200ms
3. **Together**: ~300ms
4. **OpenAI**: ~500ms
5. **Anthropic**: ~800ms
6. **Local**: Varies by hardware

### Throughput (Tokens/Second)
1. **Groq**: 300+ tokens/sec
2. **Local** (with GPU): 50-200 tokens/sec
3. **Fireworks**: 100+ tokens/sec
4. **Together**: 80+ tokens/sec
5. **OpenAI**: 50-80 tokens/sec
6. **Anthropic**: 40-60 tokens/sec

## Pricing Comparison

### Input Tokens (per 1M)
- **Local** (Ollama/llama.cpp): $0 (your hardware)
- **Anyscale**: $1 flat rate
- **Groq**: $0.10-0.80
- **Together**: $0.20-4.00
- **Fireworks**: $0.20-0.90
- **OpenRouter**: Varies by model
- **Mistral**: $2-8
- **OpenAI**: $0.50-30.00
- **Anthropic**: $3-15

### Output Tokens (per 1M)
- **Local**: $0
- **Anyscale**: $1 flat rate (same as input)
- **Groq**: $0.10-0.80
- **Together**: $0.20-4.00
- **OpenAI**: $1.50-60.00
- **Anthropic**: $15-75

## Special Features

### OpenAI
- DALL-E image generation
- GPT-4 with vision
- Whisper transcription
- Text-to-speech
- Fine-tuning API

### Anthropic
- Constitutional AI
- 200K context window
- Careful reasoning
- XML tag support

### Google
- 1M+ context window
- Native multimodal
- Multiple response candidates
- Safety settings

### Perplexity
- Real-time web search
- Source citations
- Current information
- Fact checking

### Groq
- Ultra-fast LPU inference
- Consistent low latency
- High throughput
- Deterministic performance

### Anyscale
- Ray integration
- Schema-based JSON
- Simple flat pricing
- 30 concurrent request limit

### Local Providers
- Complete privacy
- No internet required
- Custom models
- Hardware acceleration

## Model Recommendations

### For General Use
- **OpenAI GPT-4o-mini**: Best balance
- **Anthropic Claude 3.5 Sonnet**: Complex reasoning
- **Google Gemini 1.5 Flash**: Fast and capable

### For Speed
- **Groq Llama 3**: Ultra-fast
- **Fireworks**: Optimized inference
- **Local**: No network latency

### For Long Context
- **Google Gemini 1.5**: Up to 2M tokens
- **Anthropic Claude**: 200K tokens
- **X.AI Grok**: 128K tokens

### For Cost
- **Local models**: Free
- **Anyscale**: Predictable pricing
- **Groq**: Competitive rates

### For Privacy
- **Ollama**: Fully local
- **llama.cpp**: Complete control
- **Azure/Vertex**: Enterprise privacy

## Provider Limitations

### OpenAI
- Rate limits on popular models
- Higher pricing
- US-based data processing

### Anthropic
- No function calling
- Limited availability
- Higher pricing

### Google
- Limited function calling
- Beta features
- Region restrictions

### Groq
- Limited model selection
- Context window limits
- No vision support

### Local
- Requires capable hardware
- Setup complexity
- No built-in scaling

## Choosing the Right Provider

Consider these factors:

1. **Features Needed**
   - Function calling → OpenAI, Mistral
   - Vision → OpenAI, Anthropic, Google
   - Search → Perplexity
   - Local → Ollama, llama.cpp

2. **Performance Requirements**
   - Ultra-low latency → Groq
   - High throughput → Groq, Local GPU
   - Consistent performance → Major providers

3. **Budget**
   - Unlimited budget → OpenAI, Anthropic
   - Cost-conscious → Anyscale, Groq
   - Zero cost → Local providers

4. **Compliance**
   - GDPR → Mistral (EU), Azure
   - HIPAA → Azure, Vertex AI
   - Data residency → Local, Azure regions

## Next Steps

- [Provider Setup]({{ site.baseurl }}/setup.md) - Set up providers
- [Examples]({{ site.baseurl }}/examples/providers.md) - Provider-specific code
- [Best Practices]({{ site.baseurl }}/guides/best-practices.md) - Optimization tips
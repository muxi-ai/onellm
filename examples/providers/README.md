# OneLLM Provider Examples

This directory contains comprehensive examples for all 21 providers supported by OneLLM. Each example demonstrates the unique features and capabilities of that provider.

## 📚 Provider Examples

### Cloud Providers

1. **[OpenAI](openai_example.py)** - Industry standard with GPT models
   - Chat completions, streaming, function calling
   - Embeddings, JSON mode
   - Required: `OPENAI_API_KEY`

2. **[Anthropic](anthropic_example.py)** - Claude models with extended context
   - Claude 3.5 Sonnet, Claude 3 Haiku
   - Multi-turn conversations, code generation
   - Required: `ANTHROPIC_API_KEY`

3. **[Google AI Studio](google_example.py)** - Gemini models with multimodal support
   - Vision capabilities, long context
   - JSON mode, safety settings
   - Required: `GOOGLE_API_KEY`

4. **[Mistral](mistral_example.py)** - European AI with multilingual support
   - Multiple model sizes, function calling
   - Code generation, multilingual capabilities
   - Required: `MISTRAL_API_KEY`

5. **[Cohere](cohere_example.py)** - Enterprise NLP with RAG capabilities
   - Command models, structured generation
   - Summarization, code explanation
   - Required: `COHERE_API_KEY`

### Fast Inference Providers

6. **[Groq](groq_example.py)** - Ultra-fast LPU inference
   - Llama 3, Mixtral models
   - Performance benchmarks, streaming
   - Required: `GROQ_API_KEY`

7. **[Together AI](together_example.py)** - Fast open model inference
   - Llama, Mixtral, CodeLlama models
   - Quantized models, long context
   - Required: `TOGETHER_API_KEY`

8. **[Fireworks](fireworks_example.py)** - Optimized model serving
   - Speed tests, batch processing
   - Multilingual support
   - Required: `FIREWORKS_API_KEY`

9. **[Anyscale](anyscale_example.py)** - Scalable inference with Ray
   - Open-source models, JSON with schema
   - Simple pricing model
   - Required: `ANYSCALE_API_KEY`

### Specialized Providers

10. **[X.AI (Grok)](xai_example.py)** - Grok models with 128K context
    - Long context handling, reasoning
    - Creative writing, analysis
    - Required: `XAI_API_KEY`

11. **[Perplexity](perplexity_example.py)** - Search-augmented AI
    - Real-time web access, fact checking
    - Research assistant capabilities
    - Required: `PERPLEXITY_API_KEY`

12. **[DeepSeek](deepseek_example.py)** - Chinese AI with multilingual support
    - Strong Chinese language capabilities
    - Mathematical reasoning, code generation
    - Required: `DEEPSEEK_API_KEY`

13. **[OpenRouter](openrouter_example.py)** - Gateway to 100+ models
    - Free tier models, premium models
    - Model routing, variety showcase
    - Required: `OPENROUTER_API_KEY`

14. **[Moonshot](moonshot_example.py)** - Kimi models with long-context capabilities
    - Up to 200K token context
    - Multilingual Chinese/English support
    - Required: `MOONSHOT_API_KEY`

15. **[GLM (Zhipu AI)](glm_example.py)** - Chinese AI with GLM-4 models
    - Strong Chinese language capabilities
    - Vision and function calling support
    - Required: `GLM_API_KEY` or `ZAI_API_KEY`

### Multi-Provider Gateways

16. **[Vercel AI Gateway](vercel_example.py)** - Gateway to 100+ models
    - Access to OpenAI, Anthropic, Google, Meta models
    - Unified billing and streaming
    - Required: `VERCEL_AI_API_KEY`

### Enterprise Cloud

17. **[Azure OpenAI](azure_example.py)** - Microsoft-hosted OpenAI
    - Enterprise deployments, content filtering
    - Custom endpoints, DALL-E 3
    - Required: Azure configuration file

18. **[AWS Bedrock](bedrock_example.py)** - Multi-provider on AWS
    - Claude, Llama, Titan models
    - Streaming, embeddings
    - Required: AWS credentials in `bedrock.json`

19. **[Vertex AI](vertexai_example.py)** - Google Cloud enterprise AI
    - Gemini models, multimodal support
    - Enterprise features, batch processing
    - Required: Service account JSON

### Local Providers

20. **[Ollama](ollama_example.py)** - Local model management
    - Dynamic endpoint routing
    - Model management, custom endpoints
    - Required: Ollama running locally

21. **[llama.cpp](llama_cpp_example.py)** - Direct GGUF execution
    - Model loading, GPU acceleration
    - Token counting, performance tuning
    - Required: GGUF model files

## 🚀 Getting Started

1. **Install OneLLM**:
   ```bash
   pip install onellm
   ```

2. **Set up API keys** for the providers you want to use:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   # ... etc
   ```

3. **Run an example**:
   ```bash
   python providers/openai_example.py
   ```

## 📋 Feature Matrix

| Provider | Chat | Stream | Functions | JSON | Vision | Embeddings | Search |
|----------|------|--------|-----------|------|--------|------------|---------|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Anthropic | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Google | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Groq | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Perplexity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 💡 Common Patterns

All providers use the same unified interface:

```python
from onellm import OpenAI

# Initialize client (works for all providers)
client = OpenAI()

# Basic chat completion
response = client.chat.completions.create(
    model="provider/model-name",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message['content'])
```

## 🔧 Provider-Specific Features

Some providers have unique capabilities:

- **Perplexity**: Real-time web search
- **Groq**: Ultra-fast inference speeds
- **Vertex AI**: Enterprise Google Cloud integration
- **OpenRouter**: Access to 100+ models
- **llama.cpp**: Local GGUF model execution

## 📝 Notes

- Examples are designed to be self-contained and runnable
- Each example includes error handling and helpful setup instructions
- Some features may require specific model support or API tier
- Check provider documentation for latest model availability

## 🤝 Contributing

To add a new provider example:

1. Create `provider_name_example.py`
2. Include all major features of that provider
3. Add clear setup instructions
4. Test with actual API credentials
5. Update this README

## 📚 More Resources

- [OneLLM Documentation](https://github.com/muxi-ai/onellm)
- [Provider Setup Guides](../../README.md#providers)
- [General Examples](../) - Cross-provider examples
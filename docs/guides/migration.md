---
layout: default
title: Migration Guide
parent: Guides
nav_order: 1
---

# Migration Guide

This guide helps you migrate from the OpenAI Python client to OneLLM with minimal code changes.

## Why Migrate to OneLLM?

- **Access 300+ models** with the same code
- **Built-in fallbacks** for reliability
- **Cost optimization** by switching providers
- **Local model support** for privacy
- **100% OpenAI compatible** - no breaking changes

## Migration Options

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

## Step-by-Step Migration

### 1. Install OneLLM

```bash
pip install onellm
```

### 2. Update Imports

Choose your preferred approach:

```python
# Option A: Drop-in replacement
from onellm import OpenAI as OpenAI
client = OpenAI()

# Option B: Direct API
from onellm import ChatCompletion, Completion, Embedding

# Option C: Both (recommended)
from onellm import OpenAI, ChatCompletion
```

### 3. Update Model Names (Optional)

OneLLM automatically handles OpenAI model names, but you can be explicit:

```python
# These are equivalent when using OneLLM
model="gpt-4"              # Auto-prefixed to "openai/gpt-4"
model="openai/gpt-4"       # Explicit provider prefix
```

### 4. Add API Keys

```bash
# Same as before - OneLLM uses the same environment variable
export OPENAI_API_KEY="sk-..."

# Add keys for other providers you want to use
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Common Migration Patterns

### Chat Completions

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)

# After (OneLLM - Option 1: Client)
from onellm import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",  # or "openai/gpt-4"
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)

# After (OneLLM - Option 2: Direct)
from onellm import ChatCompletion
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

### Streaming

```python
# Before (OpenAI)
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

# After (OneLLM - identical!)
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

### Embeddings

```python
# Before (OpenAI)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)

# After (OneLLM - identical!)
response = client.embeddings.create(
    model="text-embedding-3-small",  # or "openai/text-embedding-3-small"
    input="Hello world"
)
```

### Function Calling

```python
# Before (OpenAI)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {...}
        }
    }]
)

# After (OneLLM - identical!)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {...}
        }
    }]
)
```

## Advanced Migration: Multi-Provider

Once migrated, you can easily use other providers:

```python
from onellm import OpenAI

client = OpenAI()

# Use different providers by changing the model string
models = [
    "openai/gpt-4",
    "anthropic/claude-3-opus",
    "google/gemini-pro",
    "groq/llama3-70b",
    "mistral/mistral-large"
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"{model}: {response.choices[0].message.content}")
```

## Async Migration

```python
# Before (OpenAI)
from openai import AsyncOpenAI
client = AsyncOpenAI()
response = await client.chat.completions.create(...)

# After (OneLLM - identical!)
from onellm import AsyncOpenAI
client = AsyncOpenAI()
response = await client.chat.completions.create(...)

# Or use the direct async API
from onellm import ChatCompletion
response = await ChatCompletion.acreate(...)
```

## Error Handling

OneLLM uses the same error types as OpenAI:

```python
from onellm import OpenAI
from openai import AuthenticationError, RateLimitError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
```

## Testing Your Migration

1. **Run existing tests** - They should pass without changes
2. **Add fallback tests** - Test the new reliability features
3. **Try other providers** - Experiment with different models

```python
# Test script
from onellm import OpenAI

client = OpenAI()

# Test 1: Basic compatibility
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say 'test passed'"}]
)
assert "test passed" in response.choices[0].message.content.lower()

# Test 2: Fallback
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say 'fallback works'"}],
    fallback_models=["gpt-3.5-turbo"]
)
assert "fallback works" in response.choices[0].message.content.lower()

print("All tests passed!")
```

## Migration Checklist

- [ ] Install OneLLM: `pip install onellm`
- [ ] Update imports (choose your approach)
- [ ] Set up API keys for providers you'll use
- [ ] Run existing tests to ensure compatibility
- [ ] Add fallback models for critical paths
- [ ] Test with different providers
- [ ] Monitor costs across providers

## Common Questions

### Do I need to change my existing code?

No! OneLLM is designed as a drop-in replacement. Your existing code will work without changes.

### Will my API keys still work?

Yes, OneLLM uses the same environment variables as OpenAI (OPENAI_API_KEY, etc.).

### Can I use both libraries together?

Yes, but it's not recommended. OneLLM includes all OpenAI functionality.

### How do I know which provider was used?

The response includes the full model name (e.g., "openai/gpt-4") showing which provider served the request.

## Next Steps

- [Provider Setup]({{ site.baseurl }}/providers/setup) - Configure additional providers
- [Advanced Features]({{ site.baseurl }}/advanced-features) - Learn about fallbacks and retries
- [Best Practices]({{ site.baseurl }}/best-practices) - Optimize your usage
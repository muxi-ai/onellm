---
layout: default
title: Client API
nav_order: 8
has_children: true
---

# Client API Reference

The main client interface for OneLLM, providing OpenAI-compatible methods for all providers.

## Client Classes

### OpenAI

The main client class that provides access to all API endpoints.

```python
from onellm import OpenAI

client = OpenAI(
    api_key=None,  # Optional: Override environment variable
    timeout=60,    # Optional: Request timeout in seconds
    max_retries=3  # Optional: Maximum retry attempts
)
```

**Parameters:**
- `api_key` (str, optional): API key for the default provider
- `timeout` (float, optional): Default timeout for requests
- `max_retries` (int, optional): Maximum number of retry attempts
- `**kwargs`: Additional provider-specific configuration

**Attributes:**
- `chat`: Chat completions interface
- `completions`: Text completions interface (legacy)
- `embeddings`: Embeddings interface
- `files`: File operations interface
- `audio`: Audio transcription and TTS interface
- `images`: Image generation interface

### AsyncOpenAI

Async version of the client for better performance.

```python
from onellm import AsyncOpenAI
import asyncio

async def main():
    client = AsyncOpenAI()
    response = await client.chat.completions.create(...)

asyncio.run(main())
```

**Parameters:** Same as `OpenAI`

## Common Parameters

### Model Naming

All models use the format `provider/model-name`:

```python
# Examples
"openai/gpt-4o-mini"
"anthropic/claude-3-5-sonnet-20241022"
"google/gemini-1.5-flash"
"groq/llama3-8b-8192"
```

### Message Format

Messages follow the OpenAI format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"},
    {"role": "user", "content": "What's the weather like?"}
]
```

**Roles:**
- `system`: System prompt (optional)
- `user`: User messages
- `assistant`: Assistant responses
- `function`: Function call results (when using functions)

### Multimodal Messages

For providers that support images:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]
```

## Response Format

### ChatCompletionResponse

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "openai/gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}
```

### Streaming Response

When `stream=True`, returns an iterator of chunks:

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1677652288,
    "model": "openai/gpt-4o-mini",
    "choices": [{
        "index": 0,
        "delta": {
            "content": "Hello"
        },
        "finish_reason": None
    }]
}
```

## Error Handling

All methods can raise these exceptions:

```python
from onellm.errors import (
    AuthenticationError,    # Invalid API key
    RateLimitError,        # Rate limit exceeded
    InvalidRequestError,   # Invalid parameters
    ServiceUnavailableError # Service down
)
```

## Provider Detection

The client automatically detects the provider from the model name:

```python
# Automatically uses OpenAI provider
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[...]
)

# Automatically uses Anthropic provider
response = client.chat.completions.create(
    model="anthropic/claude-3",
    messages=[...]
)
```

## Configuration

### Environment Variables

The client reads these environment variables:

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
# ... etc

# OneLLM settings
ONELLM_TIMEOUT=60
ONELLM_MAX_RETRIES=3
```

### Runtime Configuration

Override settings per request:

```python
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[...],
    timeout=30,  # Override default timeout
    max_retries=5  # Override default retries
)
```

## Advanced Usage

### Custom Headers

Add custom headers to requests:

```python
client = OpenAI(
    default_headers={
        "X-Custom-Header": "value"
    }
)
```

### Base URL Override

Use custom endpoints:

```python
client = OpenAI(
    base_url="https://custom-endpoint.com/v1"
)
```

### Proxy Configuration

Use HTTP proxy:

```python
import os
os.environ["HTTP_PROXY"] = "http://proxy.example.com:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.example.com:8080"
```

## Type Hints

OneLLM provides full type hints:

```python
from typing import List, Dict, Any
from onellm.types import Message, ChatCompletionResponse

def create_message(content: str) -> Message:
    return {"role": "user", "content": content}

def get_response_text(response: ChatCompletionResponse) -> str:
    return response.choices[0].message["content"]
```

## Backwards Compatibility

OneLLM maintains compatibility with OpenAI client:

```python
# Works with existing OpenAI code
from onellm import OpenAI  # Instead of: from openai import OpenAI

# All OpenAI methods work the same
client = OpenAI()
response = client.chat.completions.create(...)
```

## Performance Tips

1. **Use Async**: For multiple requests, use `AsyncOpenAI`
2. **Stream Responses**: Use `stream=True` for better UX
3. **Set Timeouts**: Prevent hanging requests
4. **Handle Errors**: Implement retry logic
5. **Cache Responses**: Cache when appropriate

## Next Steps

- [Chat Completions](chat-completions.md) - Detailed chat API
- [Embeddings](embeddings.md) - Generate embeddings
- [Error Handling](../error-handling.md) - Handle errors properly
- [Examples](../examples/basic.md) - See more examples

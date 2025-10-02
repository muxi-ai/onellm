---
layout: default
title: Quick Start
nav_order: 3
---

# Quick Start Guide

Get up and running with OneLLM in 5 minutes!

## Installation

```bash
pip install onellm
```

## Basic Usage

### 1. Set Your API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

### 2. Your First Chat

```python
from onellm import OpenAI

# Create a client
client = OpenAI()

# Send a message
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message['content'])
```

## Using Different Providers

OneLLM supports 18+ providers. Simply change the model name:

### OpenAI
```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic
```python
# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-key"

response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google
```python
# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "your-key"

response = client.chat.completions.create(
    model="google/gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Groq (Ultra-fast)
```python
# Set your Groq API key
os.environ["GROQ_API_KEY"] = "your-key"

response = client.chat.completions.create(
    model="groq/llama3-8b-8192",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Streaming Responses

Get responses token by token:

```python
stream = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.get("content"):
        print(chunk.choices[0].delta["content"], end="", flush=True)
```

## System Messages

Add context with system messages:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I read a file in Python?"}
    ]
)
```

## Temperature Control

Adjust creativity with temperature:

```python
# More creative (temperature = 1.0)
creative = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem"}],
    temperature=1.0
)

# More focused (temperature = 0.2)
focused = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.2
)
```

## Token Limits

Control response length:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=100  # Limit response to ~100 tokens
)
```

## Error Handling

Handle errors gracefully:

```python
from onellm.errors import APIError, RateLimitError

try:
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except RateLimitError:
    print("Rate limit reached. Please wait.")
except APIError as e:
    print(f"API error: {e}")
```

## Async Support

Use async for better performance:

```python
import asyncio
from onellm import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(response.choices[0].message['content'])

asyncio.run(main())
```

## Common Patterns

### Multi-turn Conversations

```python
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"}
]

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages
)
# Output: "Your name is Alice."
```

### JSON Output

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Output valid JSON only"},
        {"role": "user", "content": "List 3 colors with hex codes"}
    ],
    response_format={"type": "json_object"}
)
```

### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

## Quick Tips

1. **Model Names**: Always use format `provider/model-name`
2. **API Keys**: Set via environment variables
3. **Streaming**: Use for better UX with long responses
4. **Temperature**: 0.0 = deterministic, 1.0 = creative
5. **Max Tokens**: Control cost and response length

## Next Steps

- [Provider List]({{ site.baseurl }}/providers/README.md) - Explore all 21 providers
- [API Reference]({{ site.baseurl }}/api/client.md) - Detailed API documentation
- [Examples]({{ site.baseurl }}/examples/basic.md) - More code examples
- [Best Practices]({{ site.baseurl }}/guides/best-practices.md) - Tips for production use

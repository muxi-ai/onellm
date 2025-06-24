---
layout: default
title: Chat Completions
parent: Client API
nav_order: 1
---

# Chat Completions API

The chat completions API is the primary interface for conversational AI interactions.

## Basic Usage

```python
from onellm import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message['content'])
```

## Method Signature

```python
def create(
    self,
    messages: List[Dict[str, Any]],
    model: str,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    user: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]
```

## Parameters

### Required Parameters

- **messages** (List[Dict]): List of messages in the conversation
  ```python
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a joke."}
  ]
  ```

- **model** (str): Model identifier in format `provider/model`
  ```python
  model = "openai/gpt-4o-mini"
  model = "anthropic/claude-3-5-sonnet-20241022"
  model = "google/gemini-1.5-flash"
  ```

### Optional Parameters

- **temperature** (float, 0-2): Controls randomness. 0 = deterministic, 2 = very random
  ```python
  temperature=0.7  # Balanced creativity
  ```

- **max_tokens** (int): Maximum tokens in response
  ```python
  max_tokens=150  # Limit response length
  ```

- **stream** (bool): Stream response chunks
  ```python
  stream=True  # Returns iterator
  ```

- **top_p** (float, 0-1): Nucleus sampling. Alternative to temperature
  ```python
  top_p=0.9  # Consider top 90% probability mass
  ```

- **frequency_penalty** (float, -2 to 2): Reduce repetition of tokens
  ```python
  frequency_penalty=0.5  # Discourage repetition
  ```

- **presence_penalty** (float, -2 to 2): Encourage new topics
  ```python
  presence_penalty=0.5  # Encourage diversity
  ```

- **stop** (str or List[str]): Stop sequences
  ```python
  stop=["\n", "END"]  # Stop at newline or "END"
  ```

- **n** (int): Number of completions to generate
  ```python
  n=3  # Generate 3 different responses
  ```

- **response_format** (Dict): Control output format
  ```python
  response_format={"type": "json_object"}  # JSON mode
  ```

- **tools** (List[Dict]): Functions the model can call
  ```python
  tools=[{
      "type": "function",
      "function": {
          "name": "get_weather",
          "description": "Get weather for a location",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string"}
              }
          }
      }
  }]
  ```

- **tool_choice** (str or Dict): Control function calling
  ```python
  tool_choice="auto"  # Let model decide
  tool_choice="none"  # Don't use tools
  tool_choice={"type": "function", "function": {"name": "get_weather"}}
  ```

- **seed** (int): For reproducible outputs (provider-dependent)
  ```python
  seed=42  # Deterministic generation
  ```

- **user** (str): Unique user identifier
  ```python
  user="user-123"  # Track user
  ```

- **timeout** (float): Request timeout in seconds
  ```python
  timeout=30.0  # 30 second timeout
  ```

## Response Format

### Standard Response

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# Access the response
print(response.id)  # Unique ID
print(response.model)  # Model used
print(response.created)  # Timestamp
print(response.choices[0].message['content'])  # Message content
print(response.choices[0].finish_reason)  # Why generation stopped
print(response.usage['total_tokens'])  # Token usage
```

### Streaming Response

```python
stream = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.get('content'):
        print(chunk.choices[0].delta['content'], end='')
```

## Message Types

### Text Messages

```python
messages = [
    {"role": "system", "content": "You are a pirate."},
    {"role": "user", "content": "Tell me about treasure."}
]
```

### Multimodal Messages

For providers that support images:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"  # Optional: high, low, auto
                }
            }
        ]
    }
]
```

### Function Results

When using function calling:

```python
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}'
            }
        }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": '{"temperature": 22, "condition": "sunny"}'
    }
]
```

## Advanced Features

### JSON Mode

Force JSON output:

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

Let the model call functions:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)

# Check if model wants to call a function
if response.choices[0].message.get('tool_calls'):
    tool_call = response.choices[0].message['tool_calls'][0]
    print(f"Function: {tool_call['function']['name']}")
    print(f"Arguments: {tool_call['function']['arguments']}")
```

### Multi-turn Conversations

Maintain context across messages:

```python
messages = []

# First turn
messages.append({"role": "user", "content": "My name is Alice"})
response1 = client.chat.completions.create(model="openai/gpt-4o-mini", messages=messages)
messages.append(response1.choices[0].message)

# Second turn
messages.append({"role": "user", "content": "What's my name?"})
response2 = client.chat.completions.create(model="openai/gpt-4o-mini", messages=messages)
# Response: "Your name is Alice"
```

### Multiple Completions

Generate multiple responses:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a creative name"}],
    n=3,  # Generate 3 options
    temperature=0.9
)

for i, choice in enumerate(response.choices):
    print(f"Option {i+1}: {choice.message['content']}")
```

## Provider-Specific Features

### OpenAI
- Supports all parameters
- Function calling
- JSON mode
- Vision (GPT-4V)

### Anthropic
- No function calling
- Supports vision
- Extended context (100K+)

### Google (Gemini)
- Multimodal support
- JSON mode
- Long context

### Groq
- Ultra-fast inference
- Limited context
- Basic features

## Best Practices

1. **System Messages**: Use clear, specific system prompts
2. **Temperature**: Use 0.2-0.3 for factual tasks, 0.7-0.9 for creative
3. **Max Tokens**: Set reasonable limits to control costs
4. **Streaming**: Use for better user experience
5. **Error Handling**: Always handle API errors

## Examples

### Creative Writing
```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a haiku about programming"}
    ],
    temperature=0.9,
    max_tokens=50
)
```

### Code Generation
```python
response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    temperature=0.2,
    max_tokens=200
)
```

### Data Extraction
```python
response = client.chat.completions.create(
    model="google/gemini-1.5-pro",
    messages=[
        {"role": "system", "content": "Extract structured data as JSON."},
        {"role": "user", "content": "John Doe, 30 years old, john@example.com"}
    ],
    response_format={"type": "json_object"},
    temperature=0
)
```

### Multi-modal with Audio
```python
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Transcribe and analyze this audio clip"},
                {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
            ]
        }
    ]
)
```

### With Fallback Models
```python
# Use multiple models with automatic fallback
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    fallback_models=["anthropic/claude-3-opus", "mistral/mistral-large"]
)
```

### With Retries and Fallbacks
```python
# Enhanced reliability with retries before fallback
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    retries=3,  # Try gpt-4 up to 3 additional times before failing or using fallbacks
    fallback_models=["anthropic/claude-3-opus", "mistral/mistral-large"]
)
```

## Next Steps

- [Streaming](../guides/streaming.md) - Detailed streaming guide
- [Function Calling](../guides/function-calling.md) - Advanced function calling
- [Error Handling](../error-handling.md) - Handle errors properly
- [Provider Capabilities](../providers/capabilities.md) - Provider-specific features
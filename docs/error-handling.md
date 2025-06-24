---
layout: default
title: Error Handling
nav_order: 6
---

# Error Handling

This guide covers error handling in OneLLM, including error types, handling strategies, and best practices.

## Error Hierarchy

OneLLM uses a structured error hierarchy for consistent error handling across all providers:

```
OneLLMError (base)
├── APIError
│   ├── AuthenticationError
│   ├── RateLimitError
│   ├── InvalidRequestError
│   ├── ResourceNotFoundError
│   └── ServiceUnavailableError
├── ConfigurationError
│   └── InvalidConfigurationError
└── ProviderError
    └── ProviderNotFoundError
```

## Common Error Types

### AuthenticationError

Raised when API authentication fails:

```python
from onellm.errors import AuthenticationError

try:
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print(f"Provider: {e.provider}")
    print(f"Status code: {e.status_code}")
```

**Common Causes:**
- Invalid API key
- Expired API key
- Wrong API key for provider
- Missing API key

### RateLimitError

Raised when rate limits are exceeded:

```python
from onellm.errors import RateLimitError
import time

try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    print(f"Rate limit hit: {e}")
    if hasattr(e, 'retry_after'):
        print(f"Retry after: {e.retry_after} seconds")
        time.sleep(e.retry_after)
```

**Common Causes:**
- Too many requests per minute
- Token limits exceeded
- Concurrent request limits

### InvalidRequestError

Raised for invalid requests:

```python
from onellm.errors import InvalidRequestError

try:
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[],  # Empty messages
        temperature=2.5  # Invalid temperature
    )
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
    print(f"Details: {e.details}")
```

**Common Causes:**
- Invalid parameters
- Missing required fields
- Unsupported features for model
- Invalid model name

### ServiceUnavailableError

Raised when service is temporarily unavailable:

```python
from onellm.errors import ServiceUnavailableError
import asyncio

async def retry_with_backoff():
    for attempt in range(3):
        try:
            return await client.chat.completions.acreate(...)
        except ServiceUnavailableError:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
```

## Handling Strategies

### 1. Basic Error Handling

```python
from onellm import OpenAI
from onellm.errors import APIError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 2. Specific Error Handling

```python
from onellm.errors import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ServiceUnavailableError
)

try:
    response = client.chat.completions.create(...)
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {getattr(e, 'retry_after', 60)}s")
except InvalidRequestError as e:
    print(f"Fix your request: {e}")
except ServiceUnavailableError:
    print("Service down, try again later")
except Exception as e:
    print(f"Unexpected: {e}")
```

### 3. Retry Logic

```python
from onellm.utils.retry import retry_with_exponential_backoff

@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1,
    exponential_base=2,
    errors=(RateLimitError, ServiceUnavailableError)
)
def make_request():
    return client.chat.completions.create(...)

# Or manually:
def manual_retry(max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(...)
        except (RateLimitError, ServiceUnavailableError) as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)
```

### 4. Fallback Providers

```python
providers = [
    "openai/gpt-4",
    "anthropic/claude-3-5-sonnet-20241022",
    "google/gemini-1.5-pro"
]

for model in providers:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}]
        )
        break
    except APIError as e:
        print(f"{model} failed: {e}")
        if model == providers[-1]:
            raise
```

## Error Context

OneLLM errors include helpful context:

```python
try:
    response = client.chat.completions.create(...)
except APIError as e:
    # Basic error info
    print(f"Message: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Status code: {e.status_code}")
    
    # Additional context if available
    if hasattr(e, 'request_id'):
        print(f"Request ID: {e.request_id}")
    if hasattr(e, 'details'):
        print(f"Details: {e.details}")
```

## Async Error Handling

```python
import asyncio
from onellm import AsyncOpenAI

async def handle_async_errors():
    client = AsyncOpenAI()
    
    try:
        response = await client.chat.completions.create(...)
    except APIError as e:
        print(f"Async error: {e}")

# With multiple requests
async def handle_multiple():
    tasks = [
        client.chat.completions.create(...),
        client.chat.completions.create(...),
        client.chat.completions.create(...)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
        else:
            print(f"Request {i} succeeded")
```

## Logging Errors

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    response = client.chat.completions.create(...)
except APIError as e:
    logger.error(f"API error: {e}", exc_info=True)
    logger.error(f"Provider: {e.provider}, Status: {e.status_code}")
```

## Custom Error Handling

### Create Custom Errors

```python
from onellm.errors import OneLLMError

class CustomError(OneLLMError):
    """Custom error for specific use case."""
    pass

class ModelNotSupportedError(InvalidRequestError):
    """Model doesn't support requested feature."""
    pass
```

### Error Middleware

```python
class ErrorHandlingClient:
    def __init__(self, client):
        self.client = client
    
    def create_with_fallback(self, primary_model, fallback_model, **kwargs):
        try:
            return self.client.chat.completions.create(
                model=primary_model, **kwargs
            )
        except (RateLimitError, ServiceUnavailableError):
            return self.client.chat.completions.create(
                model=fallback_model, **kwargs
            )
```

## Provider-Specific Errors

Different providers may have unique errors:

### OpenAI
```python
try:
    response = client.chat.completions.create(
        model="openai/gpt-4-vision",
        messages=[...]
    )
except InvalidRequestError as e:
    if "image" in str(e):
        print("Image format not supported")
```

### Anthropic
```python
try:
    response = client.chat.completions.create(
        model="anthropic/claude-3",
        messages=[...],
        max_tokens=100000  # Too high
    )
except InvalidRequestError as e:
    if "max_tokens" in str(e):
        print("Reduce max_tokens for this model")
```

## Best Practices

### 1. Always Handle Authentication
```python
if not os.environ.get("OPENAI_API_KEY"):
    raise ConfigurationError("OPENAI_API_KEY not set")
```

### 2. Implement Graceful Degradation
```python
def get_response_with_fallback(message):
    models = ["openai/gpt-4", "openai/gpt-3.5-turbo"]
    
    for model in models:
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            )
        except APIError:
            continue
    
    return {"error": "All models failed"}
```

### 3. Log Errors for Debugging
```python
import json

try:
    response = client.chat.completions.create(...)
except APIError as e:
    error_log = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(e).__name__,
        "message": str(e),
        "provider": getattr(e, 'provider', 'unknown'),
        "status_code": getattr(e, 'status_code', None)
    }
    logger.error(json.dumps(error_log))
```

### 4. Handle Timeouts
```python
try:
    response = client.chat.completions.create(
        model="openai/gpt-4",
        messages=[...],
        timeout=30  # 30 second timeout
    )
except TimeoutError:
    print("Request timed out, try a faster model")
```

## Testing Error Handling

```python
import pytest
from unittest.mock import patch
from onellm.errors import RateLimitError

def test_rate_limit_handling():
    with patch.object(client, 'chat') as mock_chat:
        mock_chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            provider="openai",
            status_code=429
        )
        
        with pytest.raises(RateLimitError) as exc_info:
            client.chat.completions.create(...)
        
        assert exc_info.value.status_code == 429
```

## Next Steps

- [Best Practices](guides/best-practices.md) - Error handling best practices
- [Troubleshooting](guides/troubleshooting.md) - Common error solutions
- [API Reference](api/client.md) - Complete API documentation
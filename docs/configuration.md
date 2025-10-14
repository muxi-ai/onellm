---
layout: default
title: Configuration
nav_order: 4
---

# Configuration

This guide covers how to configure OneLLM for different providers and use cases.

## Environment Variables

OneLLM uses environment variables for API keys and configuration.

### Provider API Keys

Set API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google AI Studio
export GOOGLE_API_KEY="..."

# Mistral
export MISTRAL_API_KEY="..."

# Groq
export GROQ_API_KEY="..."

# X.AI
export XAI_API_KEY="..."

# And more...
```

### OneLLM-Specific Variables

Configure OneLLM behavior:

```bash
# Set default timeout (seconds)
export ONELLM_TIMEOUT=60

# Set default max retries
export ONELLM_MAX_RETRIES=3

# Set logging level
export ONELLM_LOG_LEVEL=INFO
```

## Using .env Files

Create a `.env` file in your project:

```
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...

# OneLLM Configuration
ONELLM_TIMEOUT=60
ONELLM_MAX_RETRIES=3
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()

from onellm import OpenAI
client = OpenAI()
```

## Programmatic Configuration

### Client Configuration

Configure the client at initialization:

```python
from onellm import OpenAI

client = OpenAI(
    api_key="sk-...",  # Override environment variable
    timeout=120,       # Custom timeout
    max_retries=5      # Custom retry count
)
```

### Provider-Specific Configuration

Some providers need special configuration:

#### Azure OpenAI

Create `azure.json`:

```json
{
    "endpoint": "https://your-name.openai.azure.com",
    "api_key": "your-azure-key",
    "api_version": "2024-02-01",
    "deployments": {
        "gpt-4": "your-gpt4-deployment",
        "gpt-35-turbo": "your-gpt35-deployment"
    }
}
```

Use in code:

```python
client = OpenAI(azure_config_path="azure.json")
```

#### AWS Bedrock

Create `bedrock.json`:

```json
{
    "region": "us-east-1",
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "..."
}
```

#### Vertex AI

Set service account:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

## Model Configuration

### Model Aliases

Create custom model aliases:

```python
from onellm import OpenAI

# Create client with model mappings
client = OpenAI()

# Use provider/model format
response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Default Parameters

Set default parameters for all requests:

```python
from functools import partial

# Create a configured create method
create_chat = partial(
    client.chat.completions.create,
    temperature=0.7,
    max_tokens=500,
    top_p=0.9
)

# Use with defaults
response = create_chat(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Runtime Configuration

Configure OneLLM behavior at runtime:

```python
import onellm

# Set API keys programmatically
onellm.openai_api_key = "sk-..."  # OpenAI API key
onellm.anthropic_api_key = "sk-..."  # Anthropic API key

# Configure fallback behavior
onellm.config.fallback = {
    "enabled": True,
    "default_chains": {
        "chat": ["openai/gpt-4", "anthropic/claude-3-opus", "groq/llama3-70b"],
        "embedding": ["openai/text-embedding-3-small", "cohere/embed-english"]
    },
    "retry_delay": 1.0,
    "max_retries": 3
}
```

## Advanced Configuration

### Retry Configuration

Configure retry behavior:

```python
from onellm import OpenAI
from onellm.utils.retry import RetryConfig

client = OpenAI(
    retry_config=RetryConfig(
        max_retries=5,
        initial_backoff=1.0,
        max_backoff=60.0,
        exponential_base=2.0
    )
)
```

### Timeout Configuration

Set different timeouts:

```python
# Global timeout
client = OpenAI(timeout=120)

# Per-request timeout
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    timeout=30  # Override for this request
)
```

### Logging Configuration

Configure logging:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Or configure OneLLM logger specifically
logger = logging.getLogger("onellm")
logger.setLevel(logging.INFO)
```

## Local Model Configuration

### Ollama Configuration

```python
# Default Ollama endpoint
client = OpenAI()  # Uses http://localhost:11434

# Custom Ollama endpoint
response = client.chat.completions.create(
    model="ollama/llama3:8b@192.168.1.100:11434",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### llama.cpp Configuration

```python
# Set model directory
import os
os.environ["LLAMA_CPP_MODEL_DIR"] = "/path/to/models"

# Configure GPU layers
client = OpenAI(
    llama_cpp_config={
        "n_gpu_layers": 35,  # GPU acceleration
        "n_ctx": 4096,       # Context window
        "n_threads": 8       # CPU threads
    }
)
```

## Security Best Practices

### 1. Never Hardcode API Keys

❌ Bad:
```python
client = OpenAI(api_key="sk-1234567890")
```

✅ Good:
```python
client = OpenAI()  # Uses environment variable
```

### 2. Use Separate Keys for Different Environments

```bash
# Development
export OPENAI_API_KEY="sk-dev-..."

# Production
export OPENAI_API_KEY="sk-prod-..."
```

### 3. Rotate Keys Regularly

Keep track of key usage and rotate periodically.

### 4. Use Key Restrictions

Many providers allow restricting keys by:
- IP address
- Usage limits
- Specific models

## Configuration Files

### CLAUDE.md

For Claude.ai Code assistant, create `CLAUDE.md`:

```markdown
# CLAUDE.md

This project uses OneLLM for LLM interactions.

## Configuration
- API keys are in .env file
- Default model: openai/gpt-4o-mini
- Timeout: 60 seconds

## Common Commands
- Run tests: pytest
- Format: black .
- Lint: ruff check .
```

### pyproject.toml

Configure OneLLM in `pyproject.toml`:

```toml
[tool.onellm]
default_provider = "openai"
default_model = "gpt-4o-mini"
timeout = 60
max_retries = 3
```

## Troubleshooting Configuration

### Check Current Configuration

```python
from onellm.config import config

# Print current configuration
print(config)

# Check specific provider
print(config["providers"]["openai"])
```

### Validate API Keys

```python
import os

providers = ["OPENAI", "ANTHROPIC", "GOOGLE", "MISTRAL"]

for provider in providers:
    key_name = f"{provider}_API_KEY"
    if os.environ.get(key_name):
        print(f"✅ {key_name} is set")
    else:
        print(f"❌ {key_name} is not set")
```

## Next Steps

- [Provider Setup]({{ site.baseurl }}/providers/setup) - Detailed provider configuration
- [Best Practices]({{ site.baseurl }}/guides/best-practices) - Configuration best practices
- [Troubleshooting]({{ site.baseurl }}/guides/troubleshooting) - Common configuration issues

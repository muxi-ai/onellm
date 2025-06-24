---
layout: default
title: Ollama
parent: Providers
nav_order: 5
---

# Ollama Provider

The Ollama provider enables OneLLM to work with locally running Ollama servers, supporting dynamic endpoint routing for using multiple Ollama instances.

## Installation

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull models you want to use:
   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ollama pull llava:latest  # For vision support
   ```

## Configuration

### Environment Variables
- `OLLAMA_API_BASE` - Default Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_TIMEOUT` - Request timeout in seconds (default: 120)

### Programmatic Configuration
```python
import onellm

# Set default Ollama server
onellm.update_provider_config("ollama", api_base="http://gpu-server:11434")
```

## Model Naming Format

Ollama supports dynamic endpoint routing using the format:
```
ollama/model:tag@host:port
```

Examples:
- `ollama/llama3:8b` - Uses default server (localhost:11434)
- `ollama/llama3:8b@gpu-server:11434` - Uses specific server
- `ollama/mixtral:8x7b-instruct-q4_K_M@10.0.0.5:11434` - Uses IP address
- `ollama/llava:latest@https://secure-server:11434` - Uses HTTPS

## Usage Examples

### Basic Usage
```python
from onellm import Client

client = Client()

# Use default localhost server
response = await client.chat.completions.create(
    model="ollama/llama3:8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Multiple Servers
```python
# Use different servers for different models
client = Client()

# Fast local model
local_response = await client.chat.completions.create(
    model="ollama/llama3:8b",
    messages=[{"role": "user", "content": "Quick question"}]
)

# Powerful remote model
remote_response = await client.chat.completions.create(
    model="ollama/mixtral:8x7b@gpu-server:11434",
    messages=[{"role": "user", "content": "Complex analysis"}]
)

# Specialized model on different server
special_response = await client.chat.completions.create(
    model="ollama/codellama:34b@code-server:11434",
    messages=[{"role": "user", "content": "Write a Python function"}]
)
```

### Streaming
```python
stream = await client.chat.completions.create(
    model="ollama/llama3:8b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Vision Models
```python
# Using LLaVA for image analysis
response = await client.chat.completions.create(
    model="ollama/llava:latest",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,..."}
            }
        ]
    }]
)
```

### Ollama-Specific Parameters
```python
response = await client.chat.completions.create(
    model="ollama/llama3:8b",
    messages=[{"role": "user", "content": "Hello"}],
    # Ollama-specific parameters
    num_gpu=1,        # GPU layers to use
    num_thread=8,     # CPU threads
    num_ctx=4096,     # Context window size
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
```

### List Available Models
```python
from onellm.providers import get_provider

ollama = get_provider("ollama")

# List models on default server
models = await ollama.list_models()
print("Local models:", models)

# List models on remote server
remote_models = await ollama.list_models("http://gpu-server:11434")
print("Remote models:", remote_models)
```

## Supported Models

### Text Generation Models
- Llama 3 family: `llama3:8b`, `llama3:70b`
- Mistral family: `mistral:7b`, `mixtral:8x7b`, `mixtral:8x22b`
- CodeLlama: `codellama:7b`, `codellama:13b`, `codellama:34b`
- Phi-3: `phi3:mini`, `phi3:medium`
- Gemma: `gemma:2b`, `gemma:7b`
- And many more...

### Vision Models
- LLaVA: `llava:latest`, `llava:34b`
- BakLLaVA: `bakllava:latest`
- LLaVA-Llama3: `llava-llama3:latest`
- LLaVA-Phi3: `llava-phi3:latest`
- Moondream: `moondream:latest`
- MiniCPM-V: `minicpm-v:latest`
- Llama 3.2 Vision: `llama3.2-vision:11b`

## Features

### Supported
- ✅ Chat completions
- ✅ Streaming responses
- ✅ Vision/multimodal (model-dependent)
- ✅ Multiple server endpoints
- ✅ Model listing
- ✅ Custom parameters

### Not Supported
- ❌ Function calling (model-dependent)
- ❌ Embeddings (use specialized models)
- ❌ Audio processing
- ❌ File uploads

## Performance Tips

1. **Local vs Remote**: Use local models for low latency, remote for GPU power
2. **Model Selection**: Choose appropriate model sizes for your hardware
3. **Quantization**: Use quantized models (e.g., `q4_K_M`) for better performance
4. **GPU Acceleration**: Configure `num_gpu` for GPU-enabled systems
5. **Context Size**: Adjust `num_ctx` based on your needs and memory

## Common Issues

### Ollama Server Not Running
```
Error: Cannot connect to Ollama server at http://localhost:11434
```
**Solution**: Start Ollama with `ollama serve`

### Model Not Found
```
Error: Model 'llama3:8b' not found on http://localhost:11434
```
**Solution**: Pull the model with `ollama pull llama3:8b`

### Timeout Errors
Large models may take time to load. Increase timeout:
```python
import onellm
onellm.update_provider_config("ollama", timeout=300)
```

### Memory Issues
Reduce GPU layers or use smaller/quantized models:
```python
response = await client.chat.completions.create(
    model="ollama/llama3:8b-instruct-q4_K_M",  # Quantized model
    messages=[{"role": "user", "content": "Hello"}],
    num_gpu=0  # CPU only
)
```

## Advanced Usage

### Load Balancing
```python
import random

# Define server pool
servers = [
    "server1:11434",
    "server2:11434",
    "server3:11434"
]

# Random server selection
server = random.choice(servers)
model = f"ollama/llama3:8b@{server}"

response = await client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Model Routing by Task
```python
# Route models based on task type
task_models = {
    "code": "ollama/codellama:34b@code-server:11434",
    "chat": "ollama/llama3:8b@localhost:11434",
    "analysis": "ollama/mixtral:8x7b@gpu-server:11434",
    "vision": "ollama/llava:34b@vision-server:11434"
}

model = task_models.get(task_type, task_models["chat"])
response = await client.chat.completions.create(
    model=model,
    messages=messages
)
```
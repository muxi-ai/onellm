---
layout: default
title: Implementation Plan
parent: Llama.cpp
grand_parent: Providers
nav_order: 3
---

# llama.cpp Provider Implementation Plan

Based on the tutorial, here's a simple approach for the llama.cpp provider:

## Model Naming Convention

Support two formats:
1. **Full path**: `llama-cpp//Users/ran/models/llama-3-8b-q4_K_M.gguf`
2. **Model name**: `llama-cpp/llama-3-8b-q4_K_M.gguf` (searches in configured directory)

## Default Configuration

```python
# In config.py
"llama_cpp": {
    "model_dir": None,  # Defaults to ~/llama_models or LLAMA_CPP_MODEL_DIR
    "n_ctx": 2048,      # Context window
    "n_gpu_layers": 0,  # GPU layers (0 = CPU only)
    "n_threads": None,  # Auto-detect CPU cores
    "temperature": 0.7, # Default temperature
    "timeout": 300,     # 5 minutes for model loading
}
```

## Environment Variables

```bash
LLAMA_CPP_MODEL_DIR=/path/to/models  # Default model directory
LLAMA_CPP_N_GPU_LAYERS=32            # GPU acceleration
LLAMA_CPP_N_CTX=2048                 # Context window
LLAMA_CPP_N_THREADS=8                # CPU threads
```

## Simple Usage

```python
from onellm import Client

client = Client()

# Use model from default directory
response = await client.chat.completions.create(
    model="llama-cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use full path
response = await client.chat.completions.create(
    model="llama-cpp//home/user/models/mixtral-8x7b-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)

# With custom settings
response = await client.chat.completions.create(
    model="llama-cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}],
    n_gpu_layers=32,  # Use GPU
    temperature=0.3,  # More focused
    max_tokens=500
)
```

## Implementation Strategy

1. **Model Loading**: Cache loaded models to avoid reloading
2. **Path Resolution**: Check both full paths and model directory
3. **Auto-detection**: Detect CPU cores if n_threads not set
4. **Error Messages**: Clear instructions when llama-cpp-python not installed
5. **Memory Management**: Unload models after inactivity timeout

## Installation Message

When llama-cpp-python is not installed:
```
llama.cpp provider requires llama-cpp-python. Install it with:

# For CPU only:
pip install llama-cpp-python

# For GPU acceleration (Mac M1/M2/M3):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# For NVIDIA GPUs:
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

See docs/llama_cpp_tutorial.md for detailed setup instructions.
```

Would you like me to implement the provider with this simple approach?
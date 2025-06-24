# llama.cpp Provider

The llama.cpp provider enables OneLLM to run Large Language Models locally using GGUF format files, with optional GPU acceleration.

## Prerequisites

### 1. Install llama-cpp-python

**For CPU only:**
```bash
pip install llama-cpp-python
```

**For Mac (M1/M2/M3) with Metal GPU:**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

**For NVIDIA GPUs:**
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

**For AMD GPUs:**
```bash
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

### 2. Download GGUF Models

OneLLM includes a built-in utility for downloading GGUF models:

```bash
# Download a model (saves to ~/llama_models by default)
onellm download --repo-id "repo/name" --filename "model.gguf"

# Download to custom location
onellm download -r "repo/name" -f "model.gguf" -o /path/to/models
```

#### Examples:

```bash
# Download Llama 3 8B
onellm download -r shinkeonkim/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF \
                -f meta-llama-3-8b-instruct-q4_k_m.gguf

# Download Mistral 7B
onellm download -r TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
                -f mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Download Phi-3 Mini
onellm download -r microsoft/Phi-3-mini-4k-instruct-gguf \
                -f Phi-3-mini-4k-instruct-q4.gguf
```

#### Manual Download:

Alternatively, you can manually download models from [Hugging Face](https://huggingface.co/models?search=gguf):
- [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct-GGUF)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2-GGUF)
- [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)

Choose appropriate quantization (Q4_K_M recommended for balance)

## Configuration

### Environment Variables
- `LLAMA_CPP_MODEL_DIR` - Directory containing GGUF models (default: `~/llama_models`)
- `LLAMA_CPP_N_GPU_LAYERS` - Number of layers to offload to GPU (default: 0)
- `LLAMA_CPP_N_CTX` - Context window size (default: 2048)
- `LLAMA_CPP_N_THREADS` - CPU threads (default: auto-detect)

### Programmatic Configuration
```python
import onellm

# Set model directory
onellm.update_provider_config("llama_cpp", 
    model_dir="/path/to/models",
    n_gpu_layers=32,
    n_ctx=4096
)
```

## Model Naming Format

Two formats are supported:

1. **Model name** (searches in configured directory):
   ```
   llama_cpp/model-name.gguf
   ```

2. **Full path**:
   ```
   llama_cpp//absolute/path/to/model.gguf
   ```

## Usage Examples

### Basic Usage
```python
from onellm import Client

client = Client()

# Use model from default directory
response = await client.chat.completions.create(
    model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### GPU Acceleration
```python
# Enable GPU acceleration per request
response = await client.chat.completions.create(
    model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}],
    n_gpu_layers=32  # Offload 32 layers to GPU
)
```

### Custom Parameters
```python
response = await client.chat.completions.create(
    model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500,
    n_ctx=4096,       # Larger context window
    n_gpu_layers=32,  # GPU acceleration
    n_threads=8,      # CPU threads
    temperature=0.3,  # Lower = more focused
    top_k=40,
    top_p=0.95
)
```

### Streaming
```python
stream = await client.chat.completions.create(
    model="llama_cpp/llama-3-8b-instruct-q4_K_M.gguf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### List Available Models
```python
from onellm.providers import get_provider

provider = get_provider("llama_cpp")
models = provider.list_available_models()
print("Available models:", models)
```

## Supported Parameters

### Generation Parameters
- `max_tokens` - Maximum tokens to generate
- `temperature` - Randomness (0.0-1.0)
- `top_p` - Nucleus sampling
- `top_k` - Top-k sampling
- `stop` - Stop sequences

### Hardware Parameters
- `n_ctx` - Context window size (default: 2048)
- `n_gpu_layers` - GPU layers (0 = CPU only)
- `n_threads` - CPU threads
- `n_batch` - Batch size for processing

## Performance Tips

### 1. Choose the Right Model Size
- **8GB RAM**: 3B-7B models
- **16GB RAM**: 7B-13B models
- **32GB RAM**: 13B-30B models
- **64GB+ RAM**: 30B+ models

### 2. Quantization Selection
- **Q8_0**: Best quality, largest size
- **Q5_K_M**: Very good quality
- **Q4_K_M**: Good balance (recommended)
- **Q3_K_M**: Smaller, some quality loss
- **Q2_K**: Smallest, noticeable quality loss

### 3. GPU Acceleration
```python
# Find optimal n_gpu_layers:
# Start with 32, increase until you hit memory limits
n_gpu_layers=32  # Try 16, 32, 64, etc.
```

### 4. Context Window
- Larger context = more memory usage
- Start with 2048, increase as needed
- Maximum depends on model training

## Model Recommendations

### General Purpose
- **Llama 3 8B**: Best overall performance
- **Mistral 7B**: Fast and capable
- **Phi-3 Mini**: Tiny but powerful

### Coding
- **CodeLlama 13B**: Specialized for code
- **DeepSeek Coder**: Good for multiple languages

### Long Context
- **Yarn Llama**: Extended context models
- **Mixtral 8x7B**: Large context window

## Common Issues

### Out of Memory
```
Error: not enough memory
```
**Solutions:**
- Use smaller model or more aggressive quantization
- Reduce `n_gpu_layers` or set to 0
- Reduce `n_ctx` (context window)

### Slow Performance
**Solutions:**
- Enable GPU: `n_gpu_layers=32`
- Use quantized models (Q4_K_M)
- Ensure sufficient CPU threads
- Close other applications

### Model Not Found
```
Error: Model 'model.gguf' not found in ~/llama_models
```
**Solutions:**
- Check model exists in directory
- Use full path: `llama_cpp//full/path/to/model.gguf`
- Verify file has .gguf extension

### Installation Failed
**Solutions:**
- Update pip: `pip install --upgrade pip`
- Install build tools:
  - Mac: `xcode-select --install`
  - Windows: Visual Studio Build Tools
  - Linux: `sudo apt-get install build-essential`

## Advanced Usage

### Model Caching
Models are cached in memory for 5 minutes after use:
```python
# First call loads model (slower)
response1 = await client.chat.completions.create(...)

# Subsequent calls use cached model (faster)
response2 = await client.chat.completions.create(...)
```

### Custom Chat Format
The provider uses a simple chat format by default. For model-specific formats, you may need to customize the prompt:
```python
# Manual prompt formatting if needed
prompt = "### Human: Hello\n### Assistant:"
response = await client.completions.create(
    model="llama_cpp/model.gguf",
    prompt=prompt
)
```

## Features

### Supported
- ✅ Chat completions
- ✅ Text completions
- ✅ Streaming responses
- ✅ GPU acceleration
- ✅ Custom parameters
- ✅ Model caching

### Not Supported
- ❌ Embeddings (use specialized models)
- ❌ Vision/multimodal
- ❌ Function calling
- ❌ Audio processing
- ❌ File uploads

## See Also
- [llama.cpp Tutorial](../llama_cpp_tutorial.md) - Detailed setup guide
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Model Downloads](https://huggingface.co/models?search=gguf)
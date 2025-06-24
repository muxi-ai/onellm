---
layout: default
title: Tutorial
parent: Llama.cpp
grand_parent: Providers
nav_order: 2
---

# llama.cpp Tutorial for OneLLM

## What is llama.cpp?

llama.cpp is a lightweight, efficient way to run Large Language Models (LLMs) locally on your computer. It's written in C++ for speed and has Python bindings that OneLLM can use. Think of it as a way to run AI models similar to ChatGPT, but completely offline on your own machine!

## Key Concepts Explained

### 1. **GGUF Model Files**
- **What they are**: GGUF (GPT-Generated Unified Format) files are compressed versions of AI models
- **Why they matter**: They're much smaller than original models (e.g., 4GB instead of 20GB)
- **Where to get them**: Download from [Hugging Face](https://huggingface.co/models?search=gguf)
- **Naming convention**: `modelname-size-quantization.gguf`
  - Example: `llama-3-8b-instruct-q4_K_M.gguf`
  - `8b` = 8 billion parameters (model size)
  - `q4_K_M` = quantization level (compression type)

### 2. **Quantization Levels**
Think of quantization like image compression - you trade some quality for much smaller file size:

| Level | Size | Quality | Use Case |
|-------|------|---------|----------|
| Q8_0 | Largest | Best | When quality matters most |
| Q5_K_M | Large | Very Good | Good balance |
| Q4_K_M | Medium | Good | **Recommended for most users** |
| Q3_K_M | Small | Okay | When space/RAM is limited |
| Q2_K | Tiny | Lower | Emergency only |

**Recommendation**: Start with Q4_K_M - it's the sweet spot!

### 3. **Hardware Considerations**

#### CPU vs GPU
- **CPU Only**: Works on any computer, but slower
- **GPU Acceleration**: Much faster, but requires compatible GPU
  - NVIDIA GPUs: Use CUDA
  - Mac M1/M2/M3: Use Metal
  - AMD GPUs: Use ROCm (less common)

#### RAM Requirements
- **8B models**: Need ~6-8GB RAM (Q4_K_M)
- **13B models**: Need ~10-12GB RAM
- **70B models**: Need ~40-50GB RAM

### 4. **Context Window**
- **What it is**: How much text the model can "remember" in a conversation
- **Default**: Usually 2048 tokens (~1500 words)
- **Larger context**: Uses more RAM but remembers more
- **Recommendation**: Start with 2048, increase if needed

## Step-by-Step Setup Guide

### Step 1: Install llama-cpp-python

**For Mac Users (M1/M2/M3):**
```bash
# Install with Metal support for GPU acceleration
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

**For Windows/Linux with NVIDIA GPU:**
```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

**For CPU only (any system):**
```bash
# Basic installation
pip install llama-cpp-python
```

### Step 2: Download a Model

#### Using OneLLM's Download Utility (Recommended)

OneLLM includes a built-in command to download models:

```bash
# Download Llama 3 8B (recommended starter model)
onellm download -r shinkeonkim/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF \
                -f meta-llama-3-8b-instruct-q4_k_m.gguf

# Download Phi-3 Mini (smaller, faster)
onellm download -r microsoft/Phi-3-mini-4k-instruct-gguf \
                -f Phi-3-mini-4k-instruct-q4.gguf
```

The models will be saved to `~/llama_models` by default.

#### Manual Download

Alternatively, you can manually download from Hugging Face:
1. Go to Hugging Face and search for GGUF models
2. Recommended starter models:
   - Small & Fast: [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
   - Balanced: [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
   - Powerful: [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
3. Download the Q4_K_M version (good balance of size/quality)

### Step 3: Organize Your Models

Create a folder for your models:
```bash
# Create a models directory in your home folder
mkdir ~/llama_models

# Move your downloaded model there
mv ~/Downloads/llama-3-8b-instruct-q4_K_M.gguf ~/llama_models/
```

### Step 4: Test Your Setup

```python
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="/Users/yourname/llama_models/llama-3-8b-instruct-q4_K_M.gguf",
    n_ctx=2048,  # Context window
    n_gpu_layers=1  # Use GPU if available (set to 0 for CPU only)
)

# Test it
response = llm("Hello! Can you explain what you are?", max_tokens=100)
print(response["choices"][0]["text"])
```

## Configuration Options Explained

### Basic Settings

```python
llm = Llama(
    model_path="path/to/model.gguf",  # Path to your model file
    n_ctx=2048,        # Context window (how much text to remember)
    n_threads=8,       # CPU threads (set to your CPU core count)
    n_gpu_layers=32,   # GPU layers (0 = CPU only, higher = more GPU)
    temperature=0.7,   # Creativity (0 = focused, 1 = creative)
)
```

### What Each Setting Does

1. **n_ctx (Context Window)**
   - Default: 2048
   - Higher = remembers more conversation
   - But uses more RAM
   - Try: 2048, 4096, or 8192

2. **n_threads (CPU Threads)**
   - Default: 4
   - Set to number of CPU cores
   - Mac: `sysctl -n hw.ncpu`
   - Windows/Linux: Check Task Manager/System Monitor

3. **n_gpu_layers (GPU Acceleration)**
   - 0 = CPU only (slower but works everywhere)
   - 1-100 = How many model layers on GPU
   - Start with 32, adjust based on GPU memory
   - If you get memory errors, reduce this number

4. **temperature (Creativity)**
   - 0.1 = Very focused, factual
   - 0.7 = Balanced (recommended)
   - 1.0 = Very creative, more random

## Recommended Setup for OneLLM

### 1. Directory Structure
```
~/llama_models/
├── general/
│   └── llama-3-8b-instruct-q4_K_M.gguf
├── code/
│   └── codellama-13b-instruct-q4_K_M.gguf
└── creative/
    └── mixtral-8x7b-instruct-q4_K_M.gguf
```

### 2. Environment Variables
Add to your `.bashrc`/`.zshrc`:
```bash
# Default model directory
export LLAMA_CPP_MODEL_DIR="$HOME/llama_models"

# Hardware settings (adjust based on your system)
export LLAMA_CPP_N_GPU_LAYERS=32  # 0 for CPU only
export LLAMA_CPP_N_CTX=2048        # Context window
export LLAMA_CPP_N_THREADS=8       # Your CPU core count
```

### 3. Model Naming for OneLLM

When llama.cpp is integrated with OneLLM, you'll use models like:
```python
# Full path approach
model="llama-cpp//Users/yourname/llama_models/llama-3-8b-instruct-q4_K_M.gguf"

# Or with configured model directory
model="llama-cpp/llama-3-8b-instruct-q4_K_M.gguf"
```

## Performance Tips

### 1. **Choosing the Right Model Size**
- **4-8GB RAM**: Use 3B-7B models
- **16GB RAM**: Use 8B-13B models  
- **32GB+ RAM**: Can handle 30B+ models

### 2. **Speed Optimization**
- Use GPU layers (`n_gpu_layers`) if you have a GPU
- Use quantized models (Q4_K_M or Q5_K_M)
- Reduce context window if you don't need long conversations
- Close other applications to free up RAM

### 3. **Quality vs Speed Trade-offs**
- **Need Speed?** Use smaller models (3B-7B) with Q4_K_M
- **Need Quality?** Use larger models (13B+) with Q5_K_M or higher
- **Best of Both?** Use 8B models with Q4_K_M

## Troubleshooting

### "Out of Memory" Error
**Solution**: 
- Use a smaller model
- Reduce `n_gpu_layers`
- Use more aggressive quantization (Q3_K_M)

### Slow Performance
**Solution**:
- Enable GPU acceleration
- Use a smaller model
- Reduce context window
- Check CPU usage

### Installation Fails
**Solution**:
- Update pip: `pip install --upgrade pip`
- Install build tools:
  - Mac: `xcode-select --install`
  - Windows: Install Visual Studio Build Tools
  - Linux: `sudo apt-get install build-essential`

## Next Steps

1. **Start Small**: Begin with a 7B parameter model
2. **Experiment**: Try different models for different tasks
3. **Join Community**: [llama.cpp Discord](https://discord.gg/llama-cpp) for help
4. **Explore Models**: Browse [Hugging Face](https://huggingface.co/models?search=gguf) for more models

## Quick Reference

### Recommended Starter Setup
```python
# For most users
model_path = "~/llama_models/llama-3-8b-instruct-q4_K_M.gguf"
n_gpu_layers = 32  # or 0 for CPU only
n_ctx = 2048
temperature = 0.7
```

### Model Selection Guide
- **General Chat**: Llama 3 8B
- **Coding**: CodeLlama 13B
- **Creative Writing**: Mixtral 8x7B
- **Fast Responses**: Phi-3 Mini
- **Best Quality**: Llama 3 70B (needs lots of RAM!)

That's it! You're ready to run LLMs locally with llama.cpp! 🚀
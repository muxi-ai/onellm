---
layout: default
title: Semantic Caching
nav_order: 7
---

# Semantic Caching

OneLLM includes an intelligent semantic caching layer that reduces API costs by 50-80% and improves response times for repeated or similar queries.

## Overview

The cache uses a hybrid two-tier approach with automatic expiration and streaming support:

1. **Hash-based exact matching** - Instant cache hits (~3.5µs) for identical queries - **42,000-143,000x faster**
2. **Semantic similarity matching** - Fast similarity search (~18ms) for near-duplicate queries - **10-30x faster**
3. **TTL auto-expiration** - Entries expire after configurable time (default: 1 day) with refresh-on-access
4. **Streaming simulation** - Cached responses chunked naturally to preserve streaming UX

**Key Benefits:**
- 💰 **Reduces API costs** by 50-80% in production
- ⚡ **Blazing fast** - 42,000-143,000x speedup for exact matches, 10-30x for semantic
- 📺 **Streaming support** - Both streaming and non-streaming requests benefit from cache
- ⏱️ **Auto-expiration** - TTL prevents stale data with refresh-on-access
- 🌍 **Multilingual support** - Works with 50+ languages
- 💵 **Zero ongoing costs** - Uses local embeddings, no API calls
- 🔒 **Privacy-focused** - All processing happens locally

## Quick Start

```python
import onellm
from onellm import ChatCompletion

# Enable cache once at startup
onellm.init_cache()

# Use OneLLM normally - responses are cached automatically
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}]
)
# First call: ~2000ms (API call + cached)

response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}]
)
# Second call: <1ms (hash cache hit - exact match)

response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Tell me about the Python programming language"}]
)
# Third call: ~18ms (semantic cache hit - 95%+ similar)
```

## How It Works

### Two-Tier Architecture

```
┌─────────────────────────────────────┐
│  User Request                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  1. Hash Lookup (~2µs)               │
│     • SHA256 of request              │
│     • OrderedDict (LRU)              │
│     • Exact match only               │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │ Hit?        │
        └──────┬──────┘
        Yes    │    No
         │     │
         │     ▼
         │  ┌─────────────────────────────┐
         │  │  2. Semantic Search (~18ms)  │
         │  │     • Extract text content   │
         │  │     • Generate embedding     │
         │  │     • FAISS similarity       │
         │  │     • Threshold: 0.95        │
         │  └──────────┬──────────────────┘
         │             │
         │      ┌──────┴──────┐
         │      │ Similarity  │
         │      │   > 0.95?   │
         │      └──────┬──────┘
         │      Yes    │    No
         │       │     │
         ▼       ▼     ▼
    ┌────────────────────┐   ┌──────────────┐
    │  Return cached     │   │  API Call    │
    │  response          │   │  + Cache     │
    └────────────────────┘   └──────────────┘
```

### Cache Key Generation

**Included in cache key (must match):**
- `model` - Model identifier
- `messages` - Conversation history
- `temperature` - Sampling temperature
- `max_tokens` - Response length limit
- `response_format` - JSON mode, etc.
- All other generation parameters

**Excluded from cache key (ignored):**
- `stream` - Streaming flag (cached responses can be returned as streams)
- `timeout` - Request timeout
- `metadata` - Custom metadata fields

### Embedding Model

OneLLM uses `paraphrase-multilingual-MiniLM-L12-v2`:
- **Size:** 118MB (one-time download)
- **Languages:** 50+ (English, Spanish, French, German, Chinese, etc.)
- **Dimensions:** 384D
- **Speed:** ~18ms per query on CPU
- **Quality:** 95% default similarity threshold

## Configuration

### Basic Configuration

```python
import onellm

# Default settings (recommended)
onellm.init_cache()

# Full configuration options
onellm.init_cache(
    max_entries=1000,              # LRU eviction limit (default: 1000)
    p=0.95,                        # Similarity threshold (default: 0.95)
    hash_only=False,               # Disable semantic matching (default: False)
    stream_chunk_strategy="words", # Chunking: words|sentences|paragraphs|characters
    stream_chunk_length=8,         # Chunk size (default: 8)
    ttl=86400                      # Time-to-live in seconds (default: 86400 = 1 day)
)

# More aggressive matching (catches more similar queries)
onellm.init_cache(p=0.90)  # p is shorthand for similarity_threshold

# Less aggressive (only very similar queries)
onellm.init_cache(p=0.98)

# Larger cache for long-running applications
onellm.init_cache(max_entries=5000)  # Default: 1000

# Shorter TTL for frequently changing data
onellm.init_cache(ttl=3600)  # 1 hour (default: 86400 = 1 day)

# Configure streaming chunk behavior
onellm.init_cache(
    stream_chunk_strategy="sentences",  # words|sentences|paragraphs|characters
    stream_chunk_length=2               # 2 sentences per chunk
)

# Combine options for production
onellm.init_cache(p=0.92, max_entries=10000, ttl=7200)
```

### Advanced Configuration

```python
# Hash-only mode (skip semantic model load, exact matches only)
onellm.init_cache(hash_only=True)

# Custom TTL with refresh-on-access
onellm.init_cache(ttl=3600)  # Entries expire after 1 hour of no access
# Note: Accessing an entry refreshes its TTL
```

### Cache Management

```python
# Get statistics
stats = onellm.cache_stats()
print(stats)
# {'hits': 15, 'misses': 5, 'entries': 10}

# Calculate hit rate
hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
print(f"Hit rate: {hit_rate:.1%}")  # 75.0%

# Clear all cached entries
onellm.clear_cache()

# Disable cache
onellm.disable_cache()

# Re-enable with different settings
onellm.init_cache(p=0.85)
```

## Use Cases

### ✅ Ideal For

**Long-running processes:**
- Web applications (Flask, FastAPI, Django)
- API services and microservices
- Background workers and daemons
- Development servers
- Jupyter notebooks
- Testing suites (cache persists across tests in same process)

**Scenarios:**
- Development and testing (repeated similar queries)
- Production with high query duplication
- Applications with common user questions
- Chatbots with FAQ-style queries

### ⚠️ Limited Benefit For

**Short-lived processes:**
- One-off scripts that exit immediately
- CLI tools that run and exit
- Batch jobs that restart frequently

**Reason:** Cache is memory-only and doesn't persist across process restarts. Each run starts with an empty cache and requires ~13s model load.

## Performance

### Benchmarks

```
Operation                    Latency      Speedup vs API    Cost
──────────────────────────────────────────────────────────────────
GPT-4 API call              150-500ms    1x (baseline)     $0.015
Hash cache hit              3.5µs        42,000-143,000x   $0
Semantic cache hit          ~18ms        10-30x            $0
Streaming simulation        3.5µs+       instant return    $0
Model load (one-time)       ~13s         -                 $0

Cache overhead on miss:     ~3µs (<0.001% of request time)
Memory per entry:           ~1-2KB
Model size:                 118MB
TTL expiration:             Automatic with refresh-on-access
```

**Key Insights:**
- Even semantic matching (~18ms) is 10-30x faster than API calls
- Cache overhead is essentially zero compared to API latency
- Streaming responses are cached and simulated naturally
- TTL auto-expiration prevents stale data accumulation

### Typical Savings

**Development:**
- Cache hit rate: 60-80%
- Cost reduction: 60-80%
- Time saved: 60-80% of API latency

**Production (with similar queries):**
- Cache hit rate: 20-40%
- Cost reduction: 20-40%
- Time saved: 20-40% of API latency

## Limitations

### Memory-Only (MVP)

The cache does **not persist** across application restarts:
- Cache is stored in RAM only
- Each process restart starts with empty cache
- No file-based or database persistence

**Future:** Persistence (SQLite backend) may be added in v1.1 if requested.

### Streaming Support with Natural Chunking

Streaming responses are **fully cached** and simulated naturally:

```python
# Streaming requests check cache and simulate streaming from cached responses
for chunk in ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}],
    stream=True
):
    print(chunk.choices[0].delta.content, end="", flush=True)
# If cached: returns instantly, chunks naturally to preserve streaming UX
# If not cached: makes API call, streams real-time, and caches for next time
```

**How it works:**
- **Cache hit**: Complete response is chunked and yielded naturally (feels like streaming)
- **Cache miss**: Real API streaming response is accumulated and cached
- **Cost savings**: Even streaming requests benefit from cache
- **UX preserved**: Users still see natural streaming behavior

**Chunking strategies:**
- `words` (default): 8 words per chunk - natural for general text
- `sentences`: 8 sentences per chunk - good for structured content
- `paragraphs`: 8 paragraphs per chunk - for longer form content
- `characters`: 8 characters per chunk - precise control

Configure chunking:
```python
onellm.init_cache(
    stream_chunk_strategy="sentences",  # or words, paragraphs, characters
    stream_chunk_length=2               # chunks per yield
)
```

### Thread Safety

Basic thread safety is provided by Python's GIL:
- Simple dict operations are thread-safe
- No explicit locks or synchronization
- Suitable for most applications

**Future:** Explicit thread-safety guarantees may be added in v1.2+ if issues are reported.

## Examples

### Development Workflow

```python
import onellm

# Initialize once at startup
onellm.init_cache()

# During development, repeatedly test similar prompts
for prompt in ["What is Python?", "Tell me about Python", "Explain Python"]:
    response = ChatCompletion.create(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    # First: API call, subsequent: cache hits
    print(response.choices[0].message["content"])

# Check how much you saved
stats = onellm.cache_stats()
print(f"Saved {stats['hits']} API calls!")
```

### Production API Service

```python
from flask import Flask, request, jsonify
import onellm

app = Flask(__name__)

# Initialize cache at startup
onellm.init_cache(max_entries=5000, p=0.93)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json

    response = ChatCompletion.create(
        model="openai/gpt-4",
        messages=data['messages']
    )

    return jsonify(response.choices[0].message)

if __name__ == '__main__':
    app.run()
```

### A/B Testing Cache Thresholds

```python
import onellm

# Test different similarity thresholds
thresholds = [0.90, 0.93, 0.95, 0.98]

for threshold in thresholds:
    onellm.clear_cache()
    onellm.init_cache(p=threshold)

    # Run test queries
    queries = [...]  # Your test queries
    for query in queries:
        response = ChatCompletion.create(...)

    stats = onellm.cache_stats()
    hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
    print(f"Threshold {threshold}: {hit_rate:.1%} hit rate")
```

### Monitoring Cache Performance

```python
import onellm
import time

onellm.init_cache()

# Wrapper to track timing
def timed_chat_completion(**kwargs):
    start = time.time()
    response = ChatCompletion.create(**kwargs)
    elapsed = time.time() - start

    stats = onellm.cache_stats()
    cache_status = "HIT" if stats['hits'] > prev_hits else "MISS"

    print(f"{cache_status}: {elapsed*1000:.1f}ms")
    return response

# Use wrapper
response = timed_chat_completion(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Troubleshooting

### Cache Not Working

**Check if cache is initialized:**
```python
stats = onellm.cache_stats()
if stats['hits'] == 0 and stats['misses'] == 0:
    print("Cache not initialized. Call onellm.init_cache()")
```

**Note on streaming:**
```python
# Streaming requests ARE cached and simulated naturally
# Both stream=True and stream=False benefit from cache
response = ChatCompletion.create(..., stream=True)   # Will use cache
response = ChatCompletion.create(..., stream=False)  # Will use cache
```

### Slow First Query

The first query after `init_cache()` loads the embedding model (~13s):
- This is a one-time cost per process
- Subsequent queries are fast
- Consider loading cache in startup code

### Low Hit Rate

**Possible causes:**
1. **Similarity threshold too high** - Try lowering: `init_cache(p=0.90)`
2. **Queries are genuinely different** - Check query similarity
3. **Parameters changing** - temperature, max_tokens, etc. affect cache key
4. **Different models** - Each model has separate cache entries

**Debug:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
onellm.init_cache()

# Check what's being cached
stats = onellm.cache_stats()
print(f"Entries: {stats['entries']}, Hits: {stats['hits']}, Misses: {stats['misses']}")
```

### Out of Memory

Cache uses ~1KB per entry. For 1000 entries:
- Hash cache: ~1MB
- Semantic index: ~400KB per 1000 vectors
- **Total**: ~1.5MB per 1000 entries

**Solutions:**
1. Reduce `max_entries`: `init_cache(max_entries=500)`
2. Use hash-only mode: `init_cache(hash_only=True)`
3. Clear cache periodically: `onellm.clear_cache()`

## API Reference

### `init_cache()`

Initialize the global semantic cache.

```python
onellm.init_cache(
    max_entries: int = 1000,
    similarity_threshold: float = 0.95,
    p: float | None = None,
    hash_only: bool = False
)
```

**Parameters:**
- `max_entries` - Maximum cache entries before LRU eviction (default: 1000)
- `similarity_threshold` - Minimum similarity score for semantic hits (default: 0.95)
- `p` - Shorthand for `similarity_threshold` (e.g., `p=0.9`)
- `hash_only` - Disable semantic matching, use only exact matches (default: False)

### `cache_stats()`

Get cache statistics.

```python
stats = onellm.cache_stats()
# Returns: {'hits': int, 'misses': int, 'entries': int}
```

### `clear_cache()`

Clear all cached entries.

```python
onellm.clear_cache()
```

### `disable_cache()`

Disable caching.

```python
onellm.disable_cache()
```

## Further Reading

- [Architecture](./architecture.md) - Overall OneLLM architecture
- [Advanced Features](./advanced-features.md) - Fallbacks, retries, and more
- [Configuration](./configuration.md) - API keys and provider setup
- [Example: cache_example.py](../examples/cache_example.py) - Complete working example

## Support

If you encounter issues with caching:
1. Check this documentation
2. Review [examples/cache_example.py](../examples/cache_example.py)
3. Open an issue on [GitHub](https://github.com/muxi-ai/onellm/issues)

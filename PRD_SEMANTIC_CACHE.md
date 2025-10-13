# Product Requirements Document: Semantic Cache Layer

## Overview

Add an optional, lightweight semantic caching layer to OneLLM that reduces API costs and improves response times by intelligently caching LLM responses. The cache will use a hybrid approach combining instant hash-based exact matching with semantic similarity search for near-duplicate queries.

## Problem Statement

**Current State:**
- Every LLM API call incurs cost and latency (~$0.015 for GPT-4, 2000ms response time)
- Developers make repeated or similar queries during development and testing
- No built-in mechanism to reduce redundant API calls
- Production applications often implement custom caching solutions

**Impact:**
- Unnecessary API costs during development (repeated identical queries)
- Slow iteration cycles when testing prompts
- Users must implement caching themselves or use heavy dependencies like GPTCache

## Goals

### Primary Goals
1. **Reduce API costs** by caching responses for identical and semantically similar queries
2. **Improve response times** from ~2000ms to <10ms for cached queries
3. **Zero ongoing cost** - use local embeddings instead of API-based embeddings
4. **Multilingual support** - work with queries in 50+ languages out of the box
5. **Simple opt-in** - easy to enable with sensible defaults

### Non-Goals (Deferred to v1.1+)
- **Cache persistence across application restarts** - Memory-only in MVP
- TTL-based expiration
- Complex eviction policies beyond LRU
- Cache warming or pre-population features
- Integration with external cache stores (Redis, Memcached, etc.)
- Thread-safety guarantees beyond Python GIL
- Detailed memory profiling in statistics
- Cache key customization hooks

### Important Limitations (MVP)
- **Cache is memory-only**: Cache does not persist across process restarts
- **Best for long-running processes**: Web servers, daemons, Jupyter notebooks
- **Limited value for short scripts**: Each `python script.py` starts with empty cache (13s model load + empty cache)

## Success Metrics

- **Cache hit rate**: >80% for typical development workflows
- **Cache overhead on miss**: <1% of total request time (<20ms)
- **Memory footprint**: <200MB for 1000 cached entries
- **Model load time**: <15 seconds (one-time startup cost)
- **API cost reduction**: >50% in development, >20% in production with similar queries

## Technical Design

### Architecture

```
User Request
    ↓
┌─────────────────────────────────────┐
│  Cache Layer (onellm.cache)         │
│                                      │
│  1. Hash Lookup (2µs)                │
│     ├─ Hit? → Return cached response │
│     └─ Miss ↓                        │
│                                      │
│  2. Semantic Search (18ms)           │
│     ├─ Hit? → Return cached response │
│     └─ Miss ↓                        │
└─────────────────────────────────────┘
    ↓
Provider API Call (2000ms)
    ↓
Cache Response & Return
```

### Cache Key Strategy

**Hash-based (Exact Match):**
```python
cache_key = sha256(json.dumps({
    "model": model,
    "messages": messages,
    "kwargs": filtered_kwargs  # Exclude stream, metadata
}, sort_keys=True))
```

**Semantic Similarity:**
```python
# Extract text content from messages
text = " ".join([msg["content"] for msg in messages])

# Generate embedding (local model, no API cost)
embedding = embedder.encode(text)  # 384D vector

# FAISS similarity search
similar_entries = faiss_index.search(embedding, k=5)

# Return if similarity > threshold (default 0.95)
if similar_entries[0].score > 0.95:
    return similar_entries[0].response
```

### Components

#### 1. Cache Module (`onellm/cache.py`)

```python
class CacheConfig:
    """Configuration for cache behavior"""
    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.95,
        hash_only: bool = False
    ):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.hash_only = hash_only

class SimpleCache:
    """
    Lightweight semantic cache for LLM responses.

    Uses hybrid approach:
    1. Hash-based exact matching (instant)
    2. Semantic similarity with local embeddings (fast, zero API cost)
    """

    def __init__(self, config: CacheConfig):
        """Initialize cache with local embedding model"""
        self.config = config
        self.hash_cache = OrderedDict()  # LRU via OrderedDict
        self.hits = 0
        self.misses = 0

        if not config.hash_only:
            self._init_semantic()

    def _init_semantic(self):
        """Lazy load semantic components"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.semantic_index = faiss.IndexFlatIP(384)
            self.semantic_data = []
        except ImportError as e:
            warnings.warn(f"Semantic cache disabled: {e}", UserWarning)
            self.config.hash_only = True

    def get(
        self,
        model: str,
        messages: list[dict],
        **kwargs
    ) -> dict | None:
        """Retrieve cached response if available"""

    def set(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        **kwargs
    ):
        """Cache a response"""

    def clear(self):
        """Clear all cached entries"""

    def stats(self) -> dict:
        """Return cache statistics"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.hash_cache)
        }
```

#### 2. Integration Points

**Chat Completion** (`onellm/chat_completion.py`):
```python
class ChatCompletion:
    @classmethod
    def create(cls, model, messages, **kwargs):
        # Check cache first
        if _cache.enabled:
            cached = _cache.get(model, messages, **kwargs)
            if cached:
                return cached

        # Make API call
        response = provider.create_chat_completion(...)

        # Cache response
        if _cache.enabled:
            _cache.set(model, messages, response, **kwargs)

        return response
```

**Global Cache Instance** (`onellm/__init__.py`):
```python
# Global cache instance (None until initialized)
_cache = None

def init_cache(
    max_entries: int = 1000,
    similarity_threshold: float = 0.95,
    p: float | None = None,  # Shorthand for similarity_threshold
    hash_only: bool = False
):
    """
    Initialize the global cache.

    Args:
        max_entries: Maximum cache entries before LRU eviction (default: 1000)
        similarity_threshold: Minimum similarity score for cache hit (default: 0.95)
        p: Shorthand for similarity_threshold (e.g., p=0.9 instead of similarity_threshold=0.9)
        hash_only: Disable semantic matching, use only hash-based exact matches (default: False)

    Example:
        >>> import onellm
        >>> onellm.init_cache()  # Enable with defaults
        >>> onellm.init_cache(p=0.9)  # More aggressive matching
        >>> onellm.init_cache(hash_only=True)  # Fast, exact matches only
        >>> response = ChatCompletion.create(...)  # Uses cache
    """
    global _cache

    # Allow p as shorthand for similarity_threshold
    if p is not None:
        similarity_threshold = p

    config = CacheConfig(max_entries, similarity_threshold, hash_only)
    _cache = SimpleCache(config)

def disable_cache():
    """Disable caching"""
    global _cache
    _cache = None

def clear_cache():
    """Clear all cached entries"""
    if _cache:
        _cache.clear()

def cache_stats() -> dict:
    """Get cache statistics"""
    if _cache:
        return _cache.stats()
    return {"hits": 0, "misses": 0, "entries": 0}
```

### Dependencies

**New Dependencies:**
```toml
[project.dependencies]
dependencies = [
    # ... existing dependencies ...
    "sentence-transformers>=2.0.0",
    "faiss-cpu>=1.7.0",
]
```

**Size Impact:**
- `sentence-transformers`: ~50MB
- `faiss-cpu`: ~20MB
- Embedding model (first run): ~118MB download
- **Total**: ~190MB additional disk space

### Embedding Model Selection

**Default: `paraphrase-multilingual-MiniLM-L12-v2`**

| Characteristic | Value | Rationale |
|----------------|-------|-----------|
| Languages | 50+ | Global audience support |
| Size | 118MB | Acceptable one-time download |
| Speed | ~18ms/query | <1% overhead vs 2000ms LLM calls |
| Dimensions | 384D | Good balance of quality and speed |
| Parameters | 118M | Lightweight enough for any machine |

**Alternative Models (Advanced Users):**
- `all-MiniLM-L6-v2`: English-only, 22MB, 9ms (faster but English-only)
- `all-mpnet-base-v2`: English-only, larger, better quality, slower

### Cache Behavior

**What Gets Cached:**
- ✅ Chat completions (non-streaming)
- ✅ Text completions (non-streaming)
- ✅ Embeddings
- ❌ Streaming responses (not cached)
- ❌ Image generation (non-deterministic)
- ❌ File operations

**Cache Key Exclusions:**
The following parameters are excluded from cache keys (don't affect matching):
- `stream`: Streaming flag
- `timeout`: Request timeout
- `metadata`: Custom metadata fields

The following ARE included (must match for cache hit):
- `model`: Must use same model
- `messages`: Core content
- `temperature`: Different temps = different responses
- `max_tokens`: Affects response length
- `response_format`: JSON mode, etc.
- All other generation parameters

**Eviction:**
- LRU eviction when `max_entries` reached (using OrderedDict)
- No TTL support in MVP (users can call `clear_cache()` manually)

### Error Handling

**Model Loading Failures:**
```python
try:
    embedder = SentenceTransformer(model)
except Exception as e:
    warnings.warn(
        f"Failed to load cache embedding model: {e}. "
        f"Cache will use hash-only mode (exact matches only).",
        UserWarning
    )
    embedder = None  # Fallback to hash-only
```

**Cache Miss is Not an Error:**
- Cache miss simply proceeds to API call
- No exceptions thrown for cache operations
- Silent degradation if cache fails

## User Experience

### Target Use Cases

**✅ Ideal for:**
- Long-running web applications (Flask, FastAPI, Django)
- Development servers that stay running
- Jupyter notebooks and interactive sessions
- Testing suites (cache persists across tests in same process)
- API services and microservices
- Background workers and daemons

**⚠️ Limited benefit for:**
- One-off scripts that exit immediately
- CLI tools that run and exit
- Batch jobs that restart frequently

### Default Behavior (Opt-in, Simple)

```python
from onellm import ChatCompletion
import onellm

# Enable cache once (at startup)
onellm.init_cache()
# 🔄 Loading cache model... (13s one-time, 118MB download on first run)
# ✅ Cache ready with multilingual support (50+ languages)

# Use OneLLM normally - responses are cached automatically
response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# First call: 2000ms (API call) + cached

response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# Second call: <1ms (hash hit)

response = ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hi there"}]  # Similar!
)
# Third call: ~18ms (semantic hit, 95%+ similar to "Hello")
```

### First-Time User Experience

```bash
$ pip install onellm
...
Successfully installed onellm-0.20251008.0

$ python
>>> import onellm
>>> from onellm import ChatCompletion
>>>
>>> onellm.init_cache()
🔄 Downloading cache embedding model (118MB, one-time)...
✅ Cache initialized with multilingual support (50+ languages)
>>>
>>> response = ChatCompletion.create(
...     model="openai/gpt-4",
...     messages=[{"role": "user", "content": "Hello"}]
... )
# Responses now cached automatically
```

### Advanced Configuration

```python
import onellm

# More aggressive matching (catches more similar queries)
onellm.init_cache(p=0.90)

# Larger cache for long-running applications
onellm.init_cache(max_entries=5000)

# Hash-only mode (fastest, exact matches only)
onellm.init_cache(hash_only=True)

# Combine options
onellm.init_cache(p=0.85, max_entries=10000)

# Check cache stats
print(onellm.cache_stats())
# {"hits": 145, "misses": 23, "entries": 89}

# Calculate hit rate
stats = onellm.cache_stats()
hit_rate = stats["hits"] / (stats["hits"] + stats["misses"])
print(f"Hit rate: {hit_rate:.1%}")  # 86.3%

# Clear cache
onellm.clear_cache()

# Disable cache
onellm.disable_cache()
```

### OpenAI Client Interface

```python
import onellm
from onellm import OpenAI

# Enable cache
onellm.init_cache()

client = OpenAI()

# Cache works automatically with client interface too
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Implementation Plan

### Phase 1: Core Cache Implementation (2-3 hours)

**Tasks:**
1. Create `onellm/cache.py` (~120 lines)
   - `CacheConfig` class
   - `SimpleCache` class with hash-based + semantic matching
   - LRU eviction via OrderedDict
   - Basic statistics (hits, misses, entries)

2. Integrate with `ChatCompletion.create()` and `ChatCompletion.acreate()`
   - Check cache before API call
   - Store response after API call
   - Handle streaming gracefully (bypass cache)

3. Integrate with `Completion.create()` and `Embedding.create()`

4. Add cache functions to `onellm/__init__.py`
   - `init_cache()`
   - `disable_cache()`
   - `clear_cache()`
   - `cache_stats()`

5. Update dependencies in `pyproject.toml`

### Phase 2: Testing (1-2 hours)

**Test Coverage:**
1. Unit tests for `SimpleCache` class
   - Hash matching accuracy
   - Semantic similarity matching
   - LRU eviction behavior

2. Integration tests
   - Cache hit/miss scenarios
   - Multiple models
   - Different message formats
   - Async operations

3. Error handling tests
   - Model loading failures
   - Graceful degradation to hash-only mode

### Phase 3: Documentation (0.5-1 hour)

1. Add concise cache section to `README.md`
2. Create `examples/cache_example.py`
3. Add docstrings to all cache functions

### Phase 4: Future Enhancements

**Not in MVP, prioritized for future releases:**

**v1.1 (High Priority if Requested):**
- **Persistent cache (SQLite backend)** - for script use cases and cross-session caching
  - Simple API: `onellm.init_cache(persist=True, cache_dir="~/.onellm/cache")`
  - Auto-cleanup of old entries
  - Corruption recovery
  - ~2 hours implementation

**v1.2+ (Lower Priority):**
- TTL support - if users request time-based expiry
- Thread safety guarantees - if users report concurrency issues
- Detailed memory profiling - if users need resource monitoring
- Cache key customization hooks - if users have specific use cases
- Alternative embedding models - if users need faster/better models
- Cache warming utilities - if users want pre-population
- Distributed caching - if enterprise users need multi-instance support

## Testing Strategy

### Unit Tests (`tests/unit/test_cache.py`)

```python
def test_hash_exact_match():
    """Test exact hash matching returns cached response"""

def test_semantic_similarity_match():
    """Test similar queries match with threshold"""

def test_lru_eviction():
    """Test LRU eviction when max_entries exceeded"""

def test_cache_disabled():
    """Test cache can be disabled"""

def test_different_models_separate_cache():
    """Test different models don't share cache entries"""

def test_streaming_bypasses_cache():
    """Test streaming requests bypass cache"""
```

### Integration Tests (`tests/integration/test_cache_integration.py`)

```python
@pytest.mark.integration
def test_cache_reduces_api_calls():
    """Test repeated queries only hit API once"""

@pytest.mark.integration
def test_cache_with_fallbacks():
    """Test cache works with fallback models"""

@pytest.mark.integration
def test_multilingual_cache():
    """Test cache works with non-English queries"""
```

### Performance Tests (`tests/performance/test_cache_performance.py`)

```python
def test_cache_overhead_on_miss():
    """Verify cache overhead <20ms on miss"""

def test_cache_hit_latency():
    """Verify cache hits <10ms"""

def test_memory_usage():
    """Verify memory usage scales linearly"""
```

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Embedding model download fails | Cache disabled | Low | Graceful fallback to hash-only mode with warning |
| High memory usage | Application slowdown | Medium | Default max_entries=1000, LRU eviction |
| Semantic matches are incorrect | Wrong cached response | Low | High default threshold (0.95), user can adjust via p parameter |
| Cache causes non-deterministic behavior | Unexpected results | Medium | Opt-in design, clear documentation |
| Dependency conflicts | Installation fails | Low | Well-maintained popular packages (sentence-transformers, faiss) |
| Thread safety issues | Race conditions | Low | Python GIL handles basic operations, document limitations |

## Launch Checklist

**Before Merge:**
- [ ] All unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Performance benchmarks meet targets (<20ms overhead)
- [ ] Documentation complete (README, example, docstrings)
- [ ] Dependency licenses verified (all compatible with Apache 2.0)
- [ ] Error handling covers edge cases (model load failure, import errors)

**After Merge:**
- [ ] Monitor GitHub issues for cache-related problems
- [ ] Collect user feedback on default threshold
- [ ] Track cache hit rates in real-world usage (if telemetry added)
- [ ] Consider persistent cache in future release

## Open Questions

1. **Should cache be enabled by default (auto-init)?**
   - Pro: Zero-config benefits, reduces costs immediately
   - Con: 13s startup penalty, surprising behavior, harder to debug
   - **Decision**: No, require explicit `init_cache()` call. Opt-in is clearer.

2. **What should the default similarity threshold be?**
   - Too high (0.99): Fewer cache hits
   - Too low (0.85): Risk of incorrect matches
   - **Decision**: 0.95 as safe default, easily adjustable via `p` parameter

3. **Should streaming responses be cached?**
   - Pro: Could cache final accumulated response
   - Con: Complicates implementation, unclear UX (user expects streaming)
   - **Decision**: No, bypass cache for streaming requests in MVP

4. **Should we include persistent cache (SQLite) in MVP?**
   - Pro: Survives application restarts, useful for short-lived scripts
   - Con: Adds ~80-100 lines of code, file management, corruption handling, ~2 more hours
   - **Decision**: No, memory-only in MVP. Add in v1.1 if users request it. This keeps implementation simple and fast to ship. The primary use case (long-running servers) doesn't need persistence.

5. **What about thread safety?**
   - Pro: Guaranteed correct behavior in multi-threaded apps
   - Con: Adds complexity (locks, performance overhead)
   - **Decision**: Rely on Python GIL for basic safety, document limitations, improve in v2 if issues reported

## Success Criteria

**MVP is successful if:**
1. ✅ Cache reduces API costs by >50% in typical development workflows
2. ✅ Cache overhead on miss is <1% of total request time (<20ms)
3. ✅ Zero reported bugs related to incorrect cache hits (threshold is safe)
4. ✅ <5 GitHub issues about cache confusion (API is intuitive, docs are clear)
5. ✅ Memory usage scales linearly and stays reasonable (<200MB for 1000 entries)
6. ✅ Model loads in <15 seconds on reasonable hardware (CPU)
7. ✅ Implementation takes <6 hours (simplified design pays off)

## Appendix

### Benchmark Results

```
╔════════════════════════════════════════════════════════════╗
║         Embedding Cache Performance Benchmark              ║
╚════════════════════════════════════════════════════════════╝

📊 Hash-Based Cache Key Benchmarks:
────────────────────────────────────────────────────────────
Prompt 1 (  4 tokens):   2.66µs (avg of 1000 runs)
Prompt 2 ( 11 tokens):   2.56µs (avg of 1000 runs)
Prompt 3 ( 31 tokens):   2.96µs (avg of 1000 runs)

📊 Local Embedding Benchmarks (CPU):
────────────────────────────────────────────────────────────
Model: paraphrase-multilingual-MiniLM-L12-v2
Load time: ~13s (one-time cost)
Prompt 1 (  4 tokens):  ~18ms (avg of 10 runs)
Prompt 2 ( 11 tokens):  ~18ms (avg of 10 runs)
Prompt 3 ( 31 tokens):  ~18ms (avg of 10 runs)

📊 Comparison:
────────────────────────────────────────────────────────────
Hash lookup:        2.66µs  (0.0001% overhead vs 2000ms LLM call)
Semantic search:      18ms  (0.9% overhead vs 2000ms LLM call)
OpenAI embedding:    150ms  (7.5% overhead + API cost)
```

### Related Documents

- [CLAUDE.md](./CLAUDE.md) - Development guide
- [README.md](./README.md) - User-facing documentation
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-13 | 1.0 | Initial draft | Claude |
| 2025-10-13 | 1.1 | Simplified design based on review:<br>- Removed TTL support<br>- Removed persistence<br>- Simplified statistics<br>- Removed thread safety over-engineering<br>- Added `p` shorthand parameter<br>- Changed to opt-in initialization<br>- Reduced implementation time 40% | Claude |

---

**Document Status:** Ready for Implementation
**Last Updated:** 2025-10-13
**Author:** Claude (via Conductor)
**Reviewers:** TBD

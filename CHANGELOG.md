# CHANGELOG

## 0.20251222.0 - Semantic Cache Improvements

**Status**: Development Status :: 5 - Production/Stable

### Bug Fixes

#### Semantic Cache False Positives

Fixed critical issues with the semantic cache that caused incorrect cache matches:

1. **System Prompt Hash Matching**: The semantic cache now includes a hash of the system prompt when matching cached responses. Previously, different LLM operations with similar user messages but different system prompts could incorrectly return cached responses from unrelated operations.

2. **Short Text Exclusion**: Messages shorter than 128 characters are now excluded from semantic matching (configurable via `min_text_length`). Short questions have misleadingly high semantic similarity scores which caused false cache hits. These short messages still benefit from exact hash matching.

3. **Stricter Default Threshold**: Default similarity threshold increased from 0.95 to 0.98 for more reliable matching.

### Changes

- Added `_extract_system_hash()` method to compute SHA256 hash of system prompt content
- Modified `_semantic_search()` to require both semantic similarity AND system hash match
- Added configurable `min_text_length` parameter (default: 128 chars) before semantic cache operations
- Changed default `similarity_threshold` from 0.95 to 0.98
- Added `caching` parameter to `ChatCompletion.create/acreate` for per-call cache bypass

---

## 0.20251218.0 - Exception Naming Improvements

**Status**: Development Status :: 5 - Production/Stable

### Breaking Changes

This release contains breaking changes to exception class names. These changes improve Python compatibility and brand consistency.

#### Exception Renames (PR #9 - Python Builtin Shadowing Fix)

The following exceptions were renamed to avoid shadowing Python's built-in exception names:

| Old Name | New Name |
|----------|----------|
| `TimeoutError` | `RequestTimeoutError` |
| `PermissionError` | `PermissionDeniedError` |

**Migration:**
```python
# Before
from onellm.exceptions import TimeoutError, PermissionError

try:
    response = client.chat.completions.create(...)
except TimeoutError:
    print("Request timed out")
except PermissionError:
    print("Permission denied")

# After
from onellm.exceptions import RequestTimeoutError, PermissionDeniedError

try:
    response = client.chat.completions.create(...)
except RequestTimeoutError:
    print("Request timed out")
except PermissionDeniedError:
    print("Permission denied")
```

#### Base Exception Rename (PR #10 - Brand Consistency)

The base exception class was renamed for brand consistency:

| Old Name | New Name |
|----------|----------|
| `MuxiLLMError` | `OneLLMError` |

**Migration:**
```python
# Before
from onellm.exceptions import MuxiLLMError

try:
    response = client.chat.completions.create(...)
except MuxiLLMError as e:
    print(f"OneLLM error: {e}")

# After
from onellm.exceptions import OneLLMError

try:
    response = client.chat.completions.create(...)
except OneLLMError as e:
    print(f"OneLLM error: {e}")
```

### Improvements

- **Exception Chaining**: All exceptions now use proper exception chaining (`raise ... from e`) for better debugging and stack traces
- **Test Suite Fixes**: Fixed test suite issues including state pollution between tests and improved mocking patterns

### Technical Details

- Exceptions no longer shadow Python builtins, preventing subtle bugs when catching exceptions
- All 373 unit tests passing with improved test isolation
- Exception hierarchy remains unchanged - only class names were updated

## 0.20251121.0 - MiniMax Provider Support

**Status**: Development Status :: 5 - Production/Stable

### New Provider

- **MiniMax**: Added support for MiniMax's M2 model series through their Anthropic-compatible API
  - **Provider name**: `minimax`
  - **API endpoint**: `https://api.minimax.io/anthropic` (international) or `https://api.minimaxi.com/anthropic` (China)
  - **Environment variable**: `MINMAX_API_KEY`
  - **Supported models**: 
    - `minimax/MiniMax-M2` - Agentic capabilities with advanced reasoning
    - `minimax/MiniMax-M2-Stable` - Optimized for high concurrency and commercial use
  - **Key features**:
    - Interleaved thinking for complex reasoning tasks
    - Tool/function calling support
    - Streaming responses
    - Anthropic-compatible API format
  - **Documentation**: Added comprehensive example at `examples/providers/minimax_example.py`

### Architecture Improvements

- **Anthropic-Compatible Provider Base**: Created `AnthropicCompatibleProvider` base class
  - Enables easy integration of providers that implement Anthropic's API format
  - Similar architecture to `OpenAICompatibleProvider` for consistency
  - MiniMax is the first provider to use this new base class
  - Zero changes required to existing Anthropic provider (fully backward compatible)

### Configuration

```python
# Using MiniMax
from onellm import OpenAI

client = OpenAI()

# Basic chat completion
response = client.chat.completions.create(
    model="minimax/MiniMax-M2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1000
)

# With interleaved thinking for reasoning tasks
response = client.chat.completions.create(
    model="minimax/MiniMax-M2",
    messages=[{"role": "user", "content": "Solve this problem: ..."}],
    max_tokens=500,
    thinking={"enabled": True, "budget_tokens": 20000}
)
```

### Testing

- Added 15 comprehensive unit tests for MiniMax provider
- All tests passing with 100% coverage for new code
- Verified Anthropic provider remains unaffected (17/17 tests passing)

### Documentation Updates

- Updated `examples/providers/README.md` to include MiniMax as provider #3
- Added `examples/providers/minimax_example.py` with usage examples
- Updated main README.md with MiniMax in provider list
- Total supported providers now: **22**

## 0.20251013.0 - Semantic Caching

**Status**: Development Status :: 5 - Production/Stable

### New Features

- **Semantic Caching**: Blazing-fast in-memory cache with intelligent semantic matching
  - **42,000-143,000x faster responses**: Cache hits return in ~7µs vs 300-1000ms for API calls
  - **50-80% cost savings**: Dramatically reduces API costs through intelligent caching
  - **Zero ongoing API costs**: Uses local multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`)
  - **Two-tier matching**: Hash-based exact matching (~2µs) with semantic similarity fallback (~18ms)
  - **Streaming support**: Artificial streaming for cached responses preserves natural UX
  - **TTL with refresh-on-access**: Configurable time-to-live (default: 86400s / 1 day)
  - **50+ language support**: Multilingual semantic matching out of the box
  - **LRU eviction**: Memory-bounded with configurable max entries (default: 1000)

### Cache Configuration

```python
import onellm

# Initialize semantic cache
onellm.init_cache(
    max_entries=1000,           # Maximum cache entries
    p=0.95,                     # Similarity threshold (0-1)
    hash_only=False,            # Enable semantic matching
    stream_chunk_strategy="words",  # Streaming chunking: words/sentences/paragraphs/characters
    stream_chunk_length=8,      # Chunks per yield
    ttl=86400                   # Time-to-live in seconds (1 day)
)

# Use cache with any provider
response = onellm.ChatCompletion.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Cache management
stats = onellm.cache_stats()    # Get hit/miss/entries stats
onellm.clear_cache()            # Clear all entries
onellm.disable_cache()          # Disable caching
```

### Performance Benchmarks

- **Hash exact match**: ~2µs (2,000,000% faster than API)
- **Semantic match**: ~18ms (1,500-5,000% faster than API)
- **Typical API call**: 300-1000ms
- **Streaming simulation**: Instant cached response with natural chunked delivery
- **Model download**: One-time 118MB download (~13s on first init)

### Technical Details

- **Dependencies**: Added `sentence-transformers>=2.0.0` and `faiss-cpu>=1.7.0` to core dependencies
- **Memory-only**: In-memory cache for long-running processes (no persistence)
- **Thread-safe**: OrderedDict-based LRU with atomic operations
- **Streaming chunking**: Four strategies (words, sentences, paragraphs, characters) for natural streaming UX
- **TTL refresh**: Cache hits refresh TTL, keeping frequently-used entries alive
- **Hash key filtering**: Excludes non-semantic parameters (`stream`, `timeout`, `metadata`) from cache key

### Documentation

- **New docs**: Comprehensive `docs/caching.md` with architecture, usage, and best practices
- **Updated README**: Highlighted semantic caching in Key Features and Advanced Features
- **Updated docs**: Added caching to `docs/README.md`, `docs/advanced-features.md`, and `docs/quickstart.md`
- **Examples**: Added `examples/cache_example.py` demonstrating all cache features

### Use Cases

**Ideal for:**
- High-traffic web applications with repeated queries
- Interactive demos and chatbots
- Development and testing environments
- API cost optimization
- Latency-sensitive applications

**Limited for:**
- Stateless serverless functions (short-lived processes)
- Highly unique, non-repetitive queries
- Contexts requiring strict data freshness

## 0.20251008.0 - ScalVer Adoption

**Status**: Development Status :: 5 - Production/Stable

### Versioning Change

- **ScalVer Adoption**: OneLLM now uses [ScalVer (Scalable Calendar Versioning)](https://scalver.org) instead of Semantic Versioning
  - Version format: `MAJOR.YYYYMMDD.PATCH` (daily cadence)
  - Current version: `0.20251008.0` (October 8, 2025)
  - MAJOR = 0 indicates alpha/experimental status per ScalVer convention
  - DATE segment uses daily format (YYYYMMDD) for maximum release flexibility
  - PATCH increments for multiple releases on the same day
  - ScalVer is SemVer-compatible, so existing tooling continues to work
  - Provides clear calendar-based release tracking while maintaining compatibility guarantees

### Why ScalVer?

ScalVer offers the best of both worlds:
- **Time-based clarity**: Know exactly when a release was made from the version number
- **SemVer compatibility**: All existing package managers and tooling work unchanged
- **Flexible cadence**: Daily format allows for rapid iteration and hotfixes
- **Breaking change tracking**: MAJOR version still signals breaking changes
- **Tool support**: Every ScalVer tag is syntactically valid SemVer 2.0

For more information about ScalVer, visit [scalver.org](https://scalver.org).

### Async Reliability

- Replaced manual event loop creation with `utils.run_async`, letting synchronous APIs safely reuse running loops in notebooks and web frameworks.
- Added Jupyter-aware fallbacks (`nest_asyncio`) and clearer guidance when sync methods are invoked from async contexts.
- Published `utils.maybe_await` to normalize sync/async callables across helpers.

### Input Validation Guardrails

- Introduced `onellm.validators` to enforce safe ranges for temperature, token limits, penalties, stop sequences, and related parameters.
- Added provider-aware model validation so invalid OpenAI, Anthropic, Mistral, and other model names fail fast with actionable messages.

### Secure File Handling & Streaming

- Hardened `File.upload`/`File.aupload` by sanitizing filenames, enforcing extension and MIME allowlists, and streaming-safe size limits for files, bytes, and file-like objects.
- Propagated validated filenames through every provider while closing directory traversal, TOCTOU, and race-condition gaps surfaced in review.
- Stabilized Amazon Bedrock streaming with aligned async usage, higher timeouts, and queue handling fixes.

### Testing & Security Evidence

- Added dedicated unit and integration coverage for async helper behavior, file validation, and provider upload regressions.
- Captured proactive security scan results and remediation reports documenting the hardening work.

## 0.1.4 - Vercel AI Gateway & 2025 Model Updates

**Status**: Development Status :: 5 - Production/Stable

### New Providers

- **Vercel AI Gateway**: Added OpenAI-compatible provider for Vercel AI Gateway
  - Access 100+ models from OpenAI, Anthropic, Google, Meta, xAI, Mistral, DeepSeek, and more
  - API Base: `https://ai-gateway.vercel.sh/v1`
  - Model naming: `vercel/vendor/model` (e.g., `vercel/openai/gpt-4o-mini`, `vercel/anthropic/claude-sonnet-4`)
  - Supports streaming, JSON mode, function calling, and vision capabilities
  - Authentication via `VERCEL_AI_API_KEY` environment variable

### Model Updates (2025 Releases)

- **OpenAI**: Added GPT-5 family (gpt-5, gpt-5-pro, gpt-5-mini, gpt-5-nano)
- **Anthropic**: Added Claude 4 family (claude-sonnet-4.5, claude-opus-4.1, claude-sonnet-4, claude-opus-4)
- **Google**: Added Gemini 2.5 family (gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-flash-image)
- **Mistral**: Added specialized models (codestral, pixtral, devstral, voxtral, ministral)

### Documentation

- Updated provider count from 20 to 21 across all documentation
- Added comprehensive provider documentation with model lists
- Added Vercel setup guide and examples

## 0.1.3 - Cache Metrics & GLM Provider

**Status**: Development Status :: 5 - Production/Stable

### Enhancements

- **Cache-Aware Usage Metrics**: Extended `UsageInfo` with `*_cached`/`*_uncached` counts while keeping totals intact for billing parity.
  - OpenAI adapter now surfaces cache hits via `prompt_tokens_details.cached_tokens`.
  - Anthropic adapter maps `cache_read_input_tokens` / `cache_creation_input_tokens` into the unified schema.
  - All consumers continue to receive `total_tokens` plus new fields defaulting to 0 when providers omit cache data.

### New Providers

- **GLM (Zhipu AI)**: Added OpenAI-compatible provider targeting `https://api.z.ai/api/paas/v4`.
  - Enables access to GLM-4 model family with streaming, JSON mode, tool calling, and vision support.
  - Reads credentials from `GLM_API_KEY` or the `ZAI_API_KEY` environment variable.

### Maintenance

- Adjusted configuration loader to accept multiple environment variable aliases per provider.
- Added focused unit tests covering cache usage normalization and GLM provider initialization.

## 0.1.2 - OpenAI Provider Compatibility Updates

**Status**: Development Status :: 5 - Production/Stable

### Bug Fixes

- **OpenAI Provider Parameter Updates**: Fixed compatibility issues with newer OpenAI models
  - Automatically converts `max_tokens` to `max_completion_tokens` for all OpenAI models
  - Removes `temperature` parameter for GPT-5 and o-series models that only support default temperature
  - Ensures compatibility with GPT-5, o1, o3, and future OpenAI model releases
  - Backward compatible - existing code using `max_tokens` continues to work without changes

### Technical Details

- Models starting with `gpt-5` or `o` now have temperature parameter automatically removed
- All OpenAI API calls now use `max_completion_tokens` instead of deprecated `max_tokens`
- Changes are transparent to users - no code modifications required

## 0.1.1 - Moonshot Provider Addition

**Status**: Development Status :: 5 - Production/Stable

### New Features

- **Moonshot Provider**: Added support for Moonshot AI's Kimi models with long-context capabilities
  - Support for `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k` models
  - Support for experimental `kimi-k2-0711-preview` model
  - Long-context processing up to 200,000 tokens
  - Multilingual support with strong Chinese language capabilities
  - Vision and audio input support via Kimi-VL and Kimi-Audio models
  - Streaming, function calling, and JSON mode support
  - Uses `MOONSHOT_API_KEY` environment variable for authentication

### Provider Count Update

- **19 Total Providers**: OneLLM now supports 19 providers (up from 18)
- **300+ Models**: Access to Moonshot's model family adds to the comprehensive model coverage

### Documentation Updates

- Updated README.md to include Moonshot in the supported providers list
- Added comprehensive examples in `examples/providers/moonshot_example.py`
- Full test coverage with real API integration tests

## 0.1.0 - Production Release

**Status**: Development Status :: 5 - Production/Stable

This release marks OneLLM's transition from beta to production-ready status, with a mature and stable codebase suitable for production deployments.

### Major Achievements

- **18 Implemented Providers**: Full support for OpenAI, Anthropic, Google, Azure, Bedrock, Mistral, Groq, Together, Anyscale, Fireworks, DeepSeek, Perplexity, OpenRouter, X.AI, Cohere, Vertex AI, Ollama, and llama.cpp
- **300+ Accessible Models**: Comprehensive model coverage across all major LLM families
- **96% Test Coverage**: Extensive test suite with 357 passing tests ensuring reliability
- **Complete Documentation**: Full Jekyll-based documentation site with guides, API reference, and examples

### New Features

- **Enhanced Fallback System**: Improved fallback mechanism with configurable retry strategies
- **Auto-Retry Support**: Automatic retries with exponential backoff for transient failures
- **JSON Mode**: Structured output support for compatible providers
- **Multi-Modal Enhancements**: Better support for vision and audio across providers
- **Advanced Configuration**: Fine-grained control over timeouts, retries, and fallback behavior
- **Local Model Support**: Seamless integration with Ollama and llama.cpp for local inference
- **CLI Model Download**: Built-in `onellm download` command to fetch GGUF models from HuggingFace

### Provider Additions

- OpenAI
- Anthropic
- Google
- Mistral
- Groq
- Together
- Fireworks
- Anyscale
- X.AI
- Perplexity
- DeepSeek
- Cohere
- OpenRouter
- Azure
- Bedrock
- Vertex AI
- Ollama
- llama.cpp

### Improvements

- **API Refinements**: Cleaner separation between `ChatCompletion`, `Completion`, and `Embedding` interfaces
- **Error Handling**: More descriptive error messages with provider-specific context
- **Performance**: Optimized provider routing and response processing
- **Type Safety**: Enhanced type hints for better IDE support
- **Documentation**: Comprehensive guides for migration, advanced features, and best practices

### Development Experience

- **Clean Test Organization**: Tests reorganized into logical unit/integration structure
- **Provider Templates**: Simplified process for adding new providers
- **CLI Tools**: Built-in model download utility for GGUF files
- **Example Library**: Extensive examples covering all major use cases

### Breaking Changes

None - This release maintains full backward compatibility with previous versions.

### Migration Notes

For users upgrading from earlier versions:
- No code changes required
- New features are opt-in through configuration
- Fallback syntax now uses `fallback_models` parameter instead of list-based model specification
- Import pattern remains the same: `from onellm import ChatCompletion`

## 0.0.7

Added an automatic unicode artifact cleaning to prevent AI detection.

- Universal Coverage: All response types (chat, streaming, completion) automatically cleaned
- Invisible Character Removal: Eliminates zero-width spaces, directional marks, and other AI-injected artifacts
- Multi-Modal Support: Handles both text and mixed-content responses
- Zero Configuration: Applied by default to all providers without code changes
- Preserves Legitimate Content: Maintains Hebrew, Arabic, Chinese, and other international text

Impact: Prevents plagiarism detectors and AI detection tools from flagging OneLLM responses based on invisible Unicode patterns commonly inserted by AI models.

## 0.0.6

- PyPi stuff

## 0.0.3

- Changed license from AGPL-3.0 to Apache-2.0 for broader adoption and easier integration
- Updated license classifiers in pyproject.toml
- Updated license badge in README.md
- Updated license section in documentation

## 0.0.2

- Initial public release
- Added support for OpenAI provider
- Implemented core API interfaces
- Added comprehensive error handling
- Added streaming support
- Added retry mechanisms
- Added model fallback chains
- Added multi-modal capabilities

## 0.0.1 (Initial Release)

The initial release of OneLLM includes a comprehensive set of features providing a complete provider-agnostic interface for large language models.

### Core Architecture

- Provider-agnostic design with consistent interface across all providers
- Consistent provider/model naming convention (`provider/model` format)
- Provider interface with factory pattern for easy extensibility
- OpenAI-compatible public API design
- Flexible configuration system supporting environment variables and runtime settings
- Full type annotation support throughout the codebase

### Error Handling

- Comprehensive error taxonomy covering all common LLM API failure modes
- Standardized error handling across all providers
- Provider-specific error mapping to unified error types
- Informative error messages with context and resolution suggestions

### API Capabilities

- Chat completions API (sync/async)
- Text completions API (sync/async)
- Embeddings API (sync/async)
- File upload/download capabilities
- Audio transcription and translation
- Text-to-speech (TTS) synthesis
- Image generation (DALL-E compatible)

### Advanced Features

- Streaming response support with token-by-token processing
- JSON mode for structured outputs
- Automatic retries with configurable backoff strategies
- Model fallback chains for enhanced reliability
- Provider capability flags for feature detection
- Multi-modal support (text, images, audio)

### OpenAI Provider

- Complete implementation of all OpenAI API endpoints
- Support for all GPT models
- DALL-E image generation
- Whisper transcription and translation
- TTS voice synthesis
- Full compatibility with OpenAI SDK patterns

### Multi-Modal Support

- Vision input for image analysis (GPT-4 Vision compatible)
- Audio transcription and translation
- Text-to-speech capabilities
- Image generation

### Quality Assurance

- Comprehensive test suite (96% code coverage)
- Unit and integration tests for all components
- Error handling and edge case testing
- Mock provider implementations for testing
- Test coverage for all API endpoints

### Documentation

- Complete API documentation via docstrings
- Annotated type definitions for all models and interfaces
- Comprehensive examples for all major features
- Detailed frontmatter in examples explaining purpose and usage
- Implementation guides for custom providers

### Developer Experience

- Drop-in replacement for OpenAI's client with minimal code changes
- Familiar API patterns for OpenAI users
- Consistent experience across providers
- Enhanced reliability through fallbacks and retries
- Simple configuration and setup

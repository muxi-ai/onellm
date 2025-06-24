# CHANGELOG

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

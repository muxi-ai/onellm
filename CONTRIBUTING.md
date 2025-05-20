# 🤝 Contributing to muxi-llm

I welcome contributions to muxi-llm! Whether you're fixing bugs, adding features, improving documentation, or supporting new providers, your help is appreciated.

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests as necessary
5. Submit a pull request

## Coding Standards

- **Type annotations**: Always use type hints for function parameters and return values
- **Docstrings**: All public functions, classes, and methods should have docstrings (follow the Google style)
- **Tests**: New features should include tests with >90% coverage
- **Error handling**: Use appropriate error types from `muxi_llm.errors`
- **Imports**: Group imports as standard library, third-party, and local
- **File structure**: Follow the existing project structure patterns
- **Naming**: Use clear, descriptive names for functions, classes, and variables

---

## Pull Request Process

1. **Open an issue first** to discuss proposed changes
2. **Reference the issue** in your PR description
3. **Keep PRs focused** on a single concern
4. **Add tests** covering your changes
5. **Update documentation** as needed
6. **Ensure CI passes** - all tests and linting checks should pass
7. **Request review** from maintainers
8. **Address feedback** promptly and thoroughly

---

## Common Contribution Examples

### Adding a new provider

1. **Create a new file** in the `muxi_llm/providers/` directory:

   ```python
   # muxi_llm/providers/my_provider.py
   from typing import Dict, List, Optional, Union, Any, AsyncIterator

   from muxi_llm.providers.base import Provider
   from muxi_llm.types import ChatCompletionChunk, ChatMessage

   class MyProvider(Provider):
       """My custom provider implementation."""

       # Set capability flags
       json_mode_support = True
       vision_support = False
       streaming_support = True

       # Required methods
       async def chat_completion(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
           """Implement chat completion for the provider."""
           # Your implementation here

       async def chat_completion_stream(self, messages: List[ChatMessage], **kwargs) -> AsyncIterator[ChatCompletionChunk]:
           """Implement streaming chat completion."""
           # Your implementation here
   ```

2. **Register the provider** in `muxi_llm/providers/__init__.py`
3. **Add tests** in `tests/providers/test_my_provider.py`
4. **Add documentation** with example usage
5. **Update provider compatibility tables** in README.md

### Fixing a bug

1. **Reproduce the issue** with a failing test
2. **Fix the code** to address the root cause
3. **Verify tests pass**
4. **Document the fix** in your PR description, including:
   - What was the issue
   - Why it occurred
   - How your solution fixes it
   - Any potential side effects

### Improving documentation

1. **Check for accuracy** - ensure all examples work as described
2. **Keep consistent style** - follow existing documentation patterns
3. **Include examples** for all new features
4. **Update relevant sections** in both docstrings and README.md

---

## Testing

- **Unit tests**: Required for all new features and bug fixes
- **Mock API responses**: Use the fixtures in `tests/fixtures`
- **Test coverage**: Aim for >90% coverage for new code
- **Test offline mode**: Ensure fallbacks work properly when providers are unavailable
- **Test edge cases**: Handle proper error cases, empty inputs, large inputs, etc.
- **Testing tools**:

  ```bash
  # Run all tests
  pytest

  # Run with coverage
  pytest --cov=muxi_llm

  # Generate coverage report
  pytest --cov=muxi_llm --cov-report=html
  ```

---

## 🛠️ Developing Custom Providers

### Provider Capability Flags

When developing custom providers for muxi-llm, use capability flags to indicate supported features:

```python
from muxi_llm.providers.base import Provider

class MyCustomProvider(Provider):
    """Custom provider implementation."""

    # Set capability flags
    json_mode_support = True         # Supports structured JSON output

    # Multi-modal capabilities
    vision_support = True            # Supports image inputs
    audio_input_support = False      # No audio input support
    video_input_support = False      # No video input support

    # Streaming capabilities
    streaming_support = True         # Supports streaming responses
    token_by_token_support = True    # Supports granular token streaming

    # Realtime capabilities
    realtime_support = False         # No realtime API support

    # Implement required methods...
```

The library automatically adapts requests based on each provider's capabilities:

1. When JSON mode is requested for providers without support, a system message is added
2. For streaming requests with non-streaming providers, streaming is disabled
3. Image/audio/video content is removed for providers without multimedia support

These flags help the library gracefully handle features across different provider implementations while maintaining consistent behavior.

---

## Contributor License Agreement (CLA)

By contributing to this project, you agree to the following terms:

1. **Grant of License:**
   You grant the project maintainer permission to license your contributions on any terms they choose. This includes, but is not limited to, open-source and commercial licensing models.

2. **Purpose:**
   This license is necessary to allow your contributions to be included in the project and to support future development, including potential commercialization or transfer of ownership to a company.

3. **Warranty and Liability:**
   Your contributions are provided "as is" without any warranty. You will not be liable for any damages related to the use of this software under any legal claim.

By submitting a pull request or other contributions, you agree to these terms.

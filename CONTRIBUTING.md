# 🤝 Contributing to OneLLM

I welcome contributions to OneLLM! Whether you're fixing bugs, adding features, improving documentation, or supporting new providers, your help is appreciated.

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests as necessary
5. Submit a pull request

## Style Guide

### Formatting & Linting

All code is enforced by CI. Run these before pushing:

```bash
# Format
black onellm tests            # line-length 100
isort onellm tests            # profile: black

# Lint
ruff check onellm tests       # rules: E, F, N, W, C90, I, B, UP, A

# Type check
mypy onellm
```

Configuration lives in `pyproject.toml` — do not duplicate settings elsewhere.

### Python Style

- **Target**: Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- **Line length**: 100 characters
- **Type annotations**: Required on all public function signatures and return values
- **Docstrings**: Google style on all public functions, classes, and methods
- **Imports**: Group as standard library → third-party → local, sorted by `isort`
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants
- **Error handling**: Raise appropriate types from `onellm.errors`, never bare `Exception`
- **Comments**: Only where the *why* isn't obvious from the code; avoid restating what the code does
- **Security**: Never log API keys, tokens, or credentials — not even prefixes

### Project Structure

```
onellm/              # Core package
  providers/         # One module per provider (inherit from Provider or OpenAICompatibleProvider)
  types/             # OpenAI-compatible type definitions
  utils/             # Shared utilities (validation, retry, token counting)
tests/
  unit/              # Fast, fully mocked — runs in CI
  integration/       # Real API calls — guarded by skipif on missing credentials
```

### Provider Conventions

- Model names follow `provider/model-name` format (e.g., `openai/gpt-4`)
- Declare capabilities via class attributes (`json_mode_support`, `vision_support`, etc.)
- Register new providers in `onellm/providers/__init__.py`
- Use `get_provider_config()` for configuration — never read env vars directly in providers

### Testing Conventions

- Use `pytest` with `pytest-asyncio` (asyncio_mode = auto)
- Aim for >90% coverage on new code
- Unit tests use mocked responses from `tests/patch_providers.py`
- Integration tests **must** use `pytest.mark.skipif` to guard missing credentials
- Never execute provider calls or load secrets at import/module level
- Avoid duplicate test module basenames across `unit/` and `integration/`

### Dependency Management

- `pyproject.toml` is the single source of truth for all metadata and dependencies
- `setup.py` is a thin shim — do not add deps or metadata there
- Pin transitive dependencies only when required by a known CVE
- Heavy optional deps (e.g., `sentence-transformers`, `faiss-cpu`) belong in extras, not core

### Git & CI

- **Branch model**: `develop` → `rc` → `main`
- **Default branch**: `develop` (all PRs target here)
- **Commit style**: conventional commits (`feat:`, `fix:`, `chore:`, `ci:`, `docs:`)
- All GitHub Actions are pinned to full commit SHAs
- All workflows use `permissions: {}` at top level with per-job grants

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

1. **Create a new file** in the `onellm/providers/` directory:

   ```python
   # onellm/providers/my_provider.py
   from typing import Dict, List, Optional, Union, Any, AsyncIterator

   from onellm.providers.base import Provider
   from onellm.types import ChatCompletionChunk, ChatMessage

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

2. **Register the provider** in `onellm/providers/__init__.py`
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
  pytest --cov=onellm

  # Generate coverage report
  pytest --cov=onellm --cov-report=html
  ```

---

## 🛠️ Developing Custom Providers

### Provider Capability Flags

When developing custom providers for OneLLM, use capability flags to indicate supported features:

```python
from onellm.providers.base import Provider

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

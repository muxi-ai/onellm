# AGENTS.md

This file provides guidance to AI coding agents (Factory Droid, Claude Code, Codex, Cursor, etc.) when working with code in this repository.

For the full style guide, coding conventions, testing rules, and CI practices see [CONTRIBUTING.md](CONTRIBUTING.md).

## Development Commands

### Building and Installation
```bash
pip install -e ".[dev]"        # dev install
python -m build                # build dist
pip install "onellm[all]"      # all provider deps
pip install "onellm[cache]"    # semantic cache deps
```

### Testing
```bash
pytest tests/unit/                                    # fast, mocked
pytest tests/unit/providers/test_openai.py            # single file
pytest --cov=onellm --cov-report=html --cov-report=term-missing  # coverage

# Integration tests (requires API keys)
source tests/artifacts/api-keys.sh
pytest tests/integration/
```

### Code Quality
```bash
black onellm tests             # format (line-length 100)
isort onellm tests             # sort imports
ruff check onellm tests        # lint
mypy onellm                    # type check
```

## Architecture Overview

OneLLM is a provider-agnostic Python library providing a unified interface for interacting with 300+ LLMs across 25+ providers. It's designed as a drop-in replacement for OpenAI's Python client.

### Key Design Patterns
1. **Factory Pattern**: Provider instantiation through registry
2. **Proxy Pattern**: Fallback provider wraps multiple providers for automatic failover
3. **Adapter Pattern**: Each provider adapts vendor-specific APIs to unified interface
4. **Capability Flags**: Providers declare supported features at class level

### Module Organization
- `onellm/` — Core package with API implementations
- `onellm/providers/` — Provider implementations (one per provider)
- `onellm/types/` — Type definitions for OpenAI compatibility
- `onellm/utils/` — Shared utilities (validation, token counting, retry)
- `tests/unit/` — Fast, fully mocked tests (runs in CI)
- `tests/integration/` — Real API tests (guarded by skipif)
- `examples/` — Usage examples

### Model Naming
Models use `provider/model-name` format (e.g., `openai/gpt-4`, `anthropic/claude-3-opus`).

### Client Architecture
`onellm/client.py` exposes an OpenAI-compatible interface (aliased as `OpenAI`), automatically routing requests to providers based on model name.

"""
Global pytest configuration for OneLLM tests.

This file contains fixtures and configuration to ensure proper test isolation
and consistent behavior, especially for asyncio-based tests.
"""

import os
import sys
import asyncio
import pathlib
import pytest

# Add the parent directory to sys.path to allow importing from onellm
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Set a consistent event loop policy for all tests
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

def pytest_configure(config):
    """Configure pytest_asyncio plugin to use session-scoped event loops by default."""
    # Set asyncio mode to auto
    config.option.asyncio_mode = "auto"

    # This will cause pytest_asyncio to use the specified scope for all event loops
    config.option.asyncio_default_fixture_loop_scope = "session"

@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset the provider registry between tests to ensure isolation."""
    # No need for any cleanup as each test uses a fresh registry

@pytest.fixture(autouse=True)
def clean_env_vars():
    """Clean environment variables that might interfere with tests."""
    original_env = os.environ.copy()

    # Remove any API keys or config vars that might be in the environment
    env_keys_to_remove = [
        k for k in os.environ if k.startswith(("OPENAI_", "ANTHROPIC_", "CLAUDE_", "MUXI_"))
    ]

    for key in env_keys_to_remove:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore the original environment
    os.environ.clear()
    os.environ.update(original_env)

# Mock function for replacing asyncio.run in tests
@pytest.fixture
def mock_asyncio_run():
    """
    Fixture to mock asyncio.run to work properly in tests.

    This helps with tests that call asyncio.run directly, which can cause
    issues when mixed with pytest-asyncio fixtures.
    """
    original_run = asyncio.run

    def run_coroutine(coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    asyncio.run = run_coroutine
    yield run_coroutine
    asyncio.run = original_run

# Import the patch_providers module to apply patches if it exists
patch_providers_path = pathlib.Path(__file__).parent / "patch_providers.py"
if patch_providers_path.exists():
    try:
        # This import is used for its side effects (applying patches) and not used directly
        import tests.patch_providers  # noqa: F401
        print("Using mocked providers to avoid real API calls")
    except ImportError:
        print("Warning: Could not import patch_providers.py")

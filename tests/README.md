# OneLLM Test Suite

This directory contains all tests for the OneLLM library, organized by test type and purpose.

## Directory Structure

```
tests/
├── unit/               # Unit tests with mocked dependencies
├── integration/        # Integration tests requiring real APIs
│   ├── providers/      # Provider-specific integration tests
│   │   ├── bedrock/    # AWS Bedrock test suite
│   │   └── vertexai/   # Google Vertex AI test suite
├── providers/          # Provider unit tests
├── utils/              # Test utilities and runners
└── artifacts/          # Test configuration files
```

## Test Categories

### Unit Tests
- Fast, isolated tests that mock external dependencies
- Run with: `pytest tests/ -k "not integration"`
- Located in main `tests/` directory and `tests/unit/`

### Integration Tests
- Tests that require real API keys and make actual API calls
- Located in `tests/integration/`
- Run with: `pytest tests/integration/` (requires API keys)

### Provider Tests
- Unit tests for provider implementations: `tests/providers/`
- Integration tests for providers: `tests/integration/providers/`

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/ -k "not integration"
```

### Integration Tests
```bash
# Requires API keys in environment
pytest tests/integration/
```

### Specific Provider
```bash
pytest tests/providers/test_openai.py
pytest tests/integration/providers/test_anyscale.py
```

### Coverage Report
```bash
pytest --cov=onellm --cov-report=html
```

## Test Utilities

### Test Runners
- `tests/utils/run_real_api_tests.py` - Run integration tests systematically
- `tests/utils/run_simple_tests.py` - Simple test runner for quick checks

### Configuration Files
- `tests/conftest.py` - Pytest configuration and fixtures
- `tests/artifacts/` - Test configuration files (API keys, credentials)

## Writing Tests

### Unit Test Example
```python
# tests/test_new_feature.py
import pytest
from unittest.mock import Mock, patch
from onellm import OpenAI

def test_feature():
    with patch('onellm.providers.openai.OpenAIProvider') as mock_provider:
        client = OpenAI()
        # Test implementation
```

### Integration Test Example
```python
# tests/integration/test_real_api.py
import pytest
from onellm import OpenAI

@pytest.mark.integration
def test_real_api_call():
    client = OpenAI()
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert response.choices[0].message['content']
```

## Environment Setup

### API Keys Configuration

1. **Edit the API keys file** with your credentials:
   ```bash
   # Edit tests/artifacts/api-keys.sh
   vi tests/artifacts/api-keys.sh
   ```

2. **Add your API keys** to the file:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."
   # ... add other provider keys as needed
   ```

3. **Source the file** before running tests:
   ```bash
   source tests/artifacts/api-keys.sh
   pytest tests/integration/
   ```

### Cloud Provider Configuration

For cloud providers, configuration files are also needed:
- `tests/artifacts/azure.json` - Azure OpenAI configuration
- `tests/artifacts/bedrock.json` - AWS Bedrock credentials
- `tests/artifacts/vertexai.json` - Google Vertex AI service account

### Test Artifacts

The `tests/artifacts/` directory contains:
- `api-keys.sh` - Template for API keys (add your keys here)
- `*.json` - Cloud provider configuration templates
- `*.gguf` - Placeholder model file for llama.cpp tests

**Note:** Never commit your actual API keys. The `api-keys.sh` file should be added to `.gitignore` if it contains real credentials.
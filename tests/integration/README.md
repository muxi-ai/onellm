# Integration Tests

Integration tests that interact with real provider APIs. These tests require valid API credentials and make actual network requests.

## Prerequisites

1. **Configure API keys** in the provided template:
   ```bash
   # Edit the API keys file
   vi tests/artifacts/api-keys.sh

   # Add your API keys:
   # export OPENAI_API_KEY="sk-..."
   # export ANTHROPIC_API_KEY="sk-ant-..."
   # etc.
   ```

2. **Source the API keys** before running tests:
   ```bash
   source tests/artifacts/api-keys.sh
   ```

3. **For cloud providers**, ensure configuration files are set up:
   - AWS Bedrock: Update `tests/artifacts/bedrock.json` with your AWS credentials
   - Azure OpenAI: Update `tests/artifacts/azure.json` with your Azure configuration
   - Vertex AI: Update `tests/artifacts/vertexai.json` with your service account

## Test Organization

### Main Integration Tests
- `test_cloud_providers.py` - Systematic tests for all cloud providers
- `test_local_providers.py` - Tests for Ollama and llama.cpp
- `test_local_simple.py` - Simple local provider checks

### Provider-Specific Tests
Located in `providers/` subdirectory:
- `test_anyscale.py` - Anyscale provider tests
- `test_google_security_fix.py` - Google provider security fix validation
- `test_openai_path_fix.py` - OpenAI path handling fix validation

### Provider Test Suites
- `providers/bedrock/` - AWS Bedrock comprehensive test suite
- `providers/vertexai/` - Google Vertex AI comprehensive test suite

## Running Integration Tests

### All Integration Tests
```bash
pytest tests/integration/
```

### Specific Provider
```bash
pytest tests/integration/providers/test_anyscale.py
```

### Cloud Providers Only
```bash
pytest tests/integration/test_cloud_providers.py
```

### Skip Slow Tests
```bash
pytest tests/integration/ -m "not slow"
```

## Writing Integration Tests

```python
import pytest
from onellm import OpenAI

@pytest.mark.integration
class TestProviderIntegration:
    def test_real_api_call(self):
        client = OpenAI()
        response = client.chat.completions.create(
            model="provider/model-name",
            messages=[{"role": "user", "content": "Test"}]
        )
        assert response.choices[0].message['content']
```

## Notes

- Integration tests are excluded from CI/CD by default
- Tests may incur API costs
- Use small models and short prompts to minimize costs
- Tests may fail due to rate limits or service availability

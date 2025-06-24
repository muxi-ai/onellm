# Test Utilities

This directory contains test runners and utilities for the OneLLM test suite.

## Test Runners

### run_real_api_tests.py
Comprehensive test runner for real API integration tests.

**Features:**
- Tests all supported providers with actual API calls
- Configurable test levels (basic, advanced, stress)
- Detailed logging and error reporting
- Cost tracking for API calls

**Usage:**
```bash
python tests/utils/run_real_api_tests.py

# Test specific providers
python tests/utils/run_real_api_tests.py --providers openai anthropic

# Run advanced tests
python tests/utils/run_real_api_tests.py --level advanced
```

### run_simple_tests.py
Lightweight test runner for quick validation.

**Features:**
- Fast smoke tests
- Basic connectivity checks
- Minimal API usage

**Usage:**
```bash
python tests/utils/run_simple_tests.py

# Test specific provider
python tests/utils/run_simple_tests.py --provider openai
```

## Adding New Utilities

Place new test utilities in this directory:
- Test data generators
- Mock response builders
- Performance benchmarking tools
- API cost calculators
- Test report generators

## Example Utility

```python
# tests/utils/test_helpers.py
def generate_test_messages(count=5):
    """Generate test message sequences."""
    return [
        {"role": "user", "content": f"Test message {i}"}
        for i in range(count)
    ]

def calculate_test_cost(provider, tokens):
    """Estimate API cost for tests."""
    rates = {
        "openai": 0.002,  # per 1K tokens
        "anthropic": 0.003,
        # ...
    }
    return (tokens / 1000) * rates.get(provider, 0.001)
```
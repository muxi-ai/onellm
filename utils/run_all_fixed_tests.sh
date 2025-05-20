#!/bin/bash

# Run fallback tests for OneLLM package
# This script runs just the fallback tests that have been fixed

set -e  # Exit on error

# Run the fixed fallback provider tests
echo "Running fallback provider tests..."
python -m pytest tests/test_fallback_100_percent.py tests/test_fallback_utils_coverage.py tests/test_fallback_provider_enhanced.py tests/test_openai_additional_coverage.py -v

# Check coverage for fallback modules
echo "Checking coverage..."
python -m pytest tests/test_fallback_100_percent.py tests/test_fallback_utils_coverage.py tests/test_fallback_provider_enhanced.py tests/test_openai_additional_coverage.py --cov=onellm.providers.fallback --cov=onellm.utils.fallback --cov=onellm.providers.openai --cov-report=term-missing -v

echo "All fallback tests completed successfully!"

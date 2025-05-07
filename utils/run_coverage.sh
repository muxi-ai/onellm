#!/bin/bash

# run_coverage.sh
# Script to run tests for each module separately and combine coverage data for an accurate report

set -e  # Exit on error

# Clean previous coverage data
echo "Clearing previous coverage data..."
python -m coverage erase

echo "Running tests with separate coverage data collection..."

# Test embedding module
echo "Testing embedding.py..."
python -m coverage run -m pytest ../tests/test_embedding_full_coverage.py

# Test provider base modules
echo "Testing providers/base.py..."
python -m coverage run -m pytest ../tests/test_provider_base_complete_coverage.py tests/test_provider_base_final_lines.py

# Test config module
echo "Testing config.py..."
python -m coverage run -m pytest ../tests/test_config_complete_coverage.py

# Test providers/fallback module
echo "Testing providers/fallback.py..."
python -m coverage run -m pytest ../tests/test_fallback_100_percent.py

# Test providers/openai module
echo "Testing providers/openai.py..."
python -m coverage run -m pytest ../tests/test_openai_provider.py tests/test_openai_provider_additional.py tests/test_openai_provider_expanded_coverage.py tests/test_openai_provider_final_coverage.py

# Test retry module
echo "Testing utils/retry.py..."
python -m coverage run -m pytest ../tests/test_retry_100_percent.py tests/test_retry_improved.py

# Test token counter module
echo "Testing utils/token_counter.py..."
python -m coverage run -m pytest ../tests/test_token_counter_coverage_improved.py

# Test streaming module
echo "Testing utils/streaming.py..."
python -m coverage run -m pytest ../tests/test_streaming_100_percent.py tests/test_streaming_fixed.py

# Test image module
echo "Testing image.py..."
python -m coverage run -m pytest ../tests/test_image_full_coverage.py tests/test_image_coverage_improved.py

# Test files module
echo "Testing files.py..."
python -m coverage run -m pytest ../tests/test_files_coverage_improved.py tests/test_files_full_coverage.py

# Test speech module
echo "Testing speech.py..."
python -m coverage run -m pytest ../tests/test_speech.py tests/test_speech_additional.py

# Combine coverage data from all test runs
echo "Combining coverage data..."
python -m coverage combine

# Generate coverage report
echo "Generating coverage report..."
python -m coverage report -m

# Generate HTML report for better visualization
echo "Generating HTML report..."
python -m coverage html

echo "Coverage analysis complete! HTML report available in htmlcov/index.html"

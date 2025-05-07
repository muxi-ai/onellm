#!/bin/bash

# Run all fixed tests for muxi-llm package
# This script runs all tests that have been fixed and are known to pass

set -e  # Exit on error

# Token counter tests
echo "Running token counter tests..."
python -m pytest ../tests/test_token_counter_basic.py tests/test_token_counter_improved.py -v

# Run the fixed streaming tests
echo "Running streaming tests..."
python -m pytest ../tests/test_streaming_basic.py tests/test_streaming_fixed_implementation.py tests/test_streaming_improved.py tests/test_stream_timeout.py tests/test_streaming_final.py tests/test_streaming_final_uncovered_lines.py -v

# Run the fixed fallback provider tests
echo "Running fallback provider tests..."
python -m pytest ../tests/test_fallback_complete.py tests/test_fallback_provider_advanced.py -v

# Run the fixed embedding tests
echo "Running embedding tests..."
python -m pytest ../tests/test_embedding_success.py tests/test_embedding_complete.py tests/test_embedding_property.py -v

# Run the fixed client interface tests
echo "Running client interface tests..."
python -m pytest ../tests/test_client_interface.py -v

# Run the fixed image tests
echo "Running image tests..."
python -m pytest ../tests/test_image_improved.py -v

# Run the fixed config tests
echo "Running config tests..."
python -m pytest ../tests/test_config_improved.py tests/test_config_all.py -v

# Run the fixed file operations tests (specifically the tests that work)
echo "Running file operations tests..."
python -m pytest ../tests/test_file_operations_parent_creation.py -v

# Check coverage for key modules
echo "Checking coverage..."
python -m pytest ../tests/test_streaming_basic.py tests/test_streaming_fixed_implementation.py tests/test_streaming_improved.py tests/test_stream_timeout.py tests/test_streaming_final.py tests/test_streaming_final_uncovered_lines.py --cov=muxi_llm.utils.streaming -v
python -m pytest ../tests/test_fallback_complete.py tests/test_fallback_provider_advanced.py --cov=muxi_llm.providers.fallback -v
python -m pytest ../tests/test_embedding_success.py tests/test_embedding_complete.py tests/test_embedding_property.py --cov=muxi_llm.embedding -v
python -m pytest ../tests/test_client_interface.py --cov=muxi_llm.client -v
python -m pytest ../tests/test_image_improved.py --cov=muxi_llm.image -v
python -m pytest ../tests/test_config_improved.py tests/test_config_all.py --cov=muxi_llm.config -v

python -m pytest ../tests/test_image_coverage.py
python -m pytest ../tests/test_client_interface.py
python -m pytest ../tests/test_client_interface_proxy.py
python -m pytest ../tests/test_streaming_coverage.py
python -m pytest ../tests/test_openai_streaming_and_tools.py
python -m pytest ../tests/test_openai_error_handling.py

echo "All tests completed successfully!"

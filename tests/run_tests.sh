#!/bin/bash
# Run OneLLM tests with API keys loaded

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if api-keys.sh exists
API_KEYS_FILE="$SCRIPT_DIR/artifacts/api-keys.sh"
if [ -f "$API_KEYS_FILE" ]; then
    echo "Loading API keys from $API_KEYS_FILE"
    source "$API_KEYS_FILE"
else
    echo "Warning: $API_KEYS_FILE not found"
    echo "Please create it by copying the template and adding your API keys"
    exit 1
fi

# Default to running all tests
TEST_PATH="${1:-$SCRIPT_DIR}"

# Run pytest with the specified path
echo "Running tests: $TEST_PATH"
cd "$SCRIPT_DIR/.." && python -m pytest "$TEST_PATH" "${@:2}"
# Test Artifacts

This directory contains configuration files and test data needed for running OneLLM tests.

## API Keys Configuration

### Setting Up API Keys

1. **Edit `api-keys.sh`** to add your API keys:
   ```bash
   vi api-keys.sh
   ```

2. **Add your keys** in the format:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."
   # Add other provider keys as needed
   ```

3. **Source the file** before running tests:
   ```bash
   # From the project root
   source tests/artifacts/api-keys.sh

   # Then run tests
   pytest tests/integration/
   ```

### Automated Loading

The test runners in `tests/utils/` automatically load API keys from this file, so you can also run:
```bash
python tests/utils/run_real_api_tests.py
python tests/utils/run_simple_tests.py
```

## Cloud Provider Configuration Files

### azure.json
Configuration for Azure OpenAI:
```json
{
    "endpoint": "https://your-resource.openai.azure.com",
    "api_key": "your-azure-api-key",
    "api_version": "2024-02-01",
    "deployments": {
        "gpt-4": "your-gpt4-deployment",
        "gpt-35-turbo": "your-gpt35-deployment"
    }
}
```

### bedrock.json
AWS Bedrock credentials:
```json
{
    "region": "us-east-1",
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "..."
}
```

### vertexai.json
Google Vertex AI service account:
```json
{
    "type": "service_account",
    "project_id": "your-project-id",
    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    "client_email": "your-service-account@your-project.iam.gserviceaccount.com"
}
```

## Model Files

### meta-llama-3-8b-instruct-q4_k_m.gguf
This is a placeholder file for llama.cpp tests. To run actual llama.cpp tests:

1. Download a real GGUF model:
   ```bash
   # Example: Download Llama 3 8B model
   wget https://huggingface.co/your-model.gguf
   ```

2. Replace the placeholder file or set the model path:
   ```bash
   export LLAMA_CPP_MODEL_DIR="/path/to/your/models"
   ```

## Security Notes

⚠️ **IMPORTANT**:
- Never commit real API keys or credentials to version control
- Add `api-keys.sh` to `.gitignore` if it contains real credentials
- Use environment-specific configuration files for different environments
- Rotate API keys regularly
- Use read-only credentials where possible for testing

## Using with CI/CD

For CI/CD pipelines, set environment variables directly:
```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  # ... other keys
```

Or create the api-keys.sh file dynamically:
```bash
cat > tests/artifacts/api-keys.sh << EOF
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
# ... other keys
EOF
```

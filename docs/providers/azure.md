# Azure OpenAI Provider

The Azure OpenAI provider allows you to use Azure OpenAI Service through OneLLM's unified interface.

## Configuration

The Azure provider requires a configuration file (`azure.json`) that contains your Azure OpenAI deployment details. The provider will look for this file in the following locations:

1. Path specified via `azure_config_path` parameter when initializing the provider
2. Path specified via `AZURE_OPENAI_CONFIG_PATH` environment variable
3. Default location: `azure.json` in the project root

### Configuration File Format

```json
{
    "key1": "your-primary-api-key",
    "key2": "your-secondary-api-key",
    "region": "uksouth",
    "endpoint": "https://your-resource.openai.azure.com/",
    "deployment": {
        "gpt-4o-mini": {
            "endpoint": "https://your-resource.openai.azure.com/",
            "model_name": "gpt-4o-mini",
            "deployment": "gpt-4o-mini-deployment",
            "subscription_key": "deployment-specific-key",
            "api_version": "2024-12-01-preview"
        },
        "o4-mini": {
            "endpoint": "https://another-resource.openai.azure.com/",
            "model_name": "o4-mini",
            "deployment": "o4-mini-deployment",
            "subscription_key": "another-deployment-key",
            "api_version": "2024-12-01-preview"
        }
    }
}
```

### Configuration Fields

- **key1, key2**: Primary and secondary API keys for your Azure OpenAI resource
- **region**: Azure region where your resource is deployed
- **endpoint**: Base endpoint URL for your Azure OpenAI resource
- **deployment**: Object containing deployment-specific configurations
  - Each deployment can have its own endpoint, API key, and API version
  - This allows you to use multiple Azure OpenAI resources and deployments

## Usage

### Basic Usage

```python
from onellm import Client

# Initialize client (will auto-detect azure.json)
client = Client()

# Use Azure OpenAI with the azure/ prefix
response = await client.chat.completions.create(
    model="azure/gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello, Azure!"}
    ]
)
```

### Direct Provider Usage

```python
from onellm.providers import get_provider

# Initialize with custom config path
azure = get_provider("azure", azure_config_path="/path/to/azure.json")

# Use the provider directly
response = await azure.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o-mini"
)
```

### Environment Variables

You can set the configuration file path via environment variable:

```bash
export AZURE_OPENAI_CONFIG_PATH=/path/to/your/azure.json
```

## Supported Features

The Azure provider supports all features available in Azure OpenAI Service:

### Chat Completions
- All GPT models deployed in your Azure resource
- Streaming support
- Function calling
- JSON mode (for supported models)

### Vision Models
- GPT-4 Vision models (gpt-4v, gpt-4o)
- Image input in chat messages

### Audio
- Whisper models for transcription and translation
- Text-to-speech with various voices

### Image Generation
- DALL-E 2 and DALL-E 3 models
- Various image sizes and quality settings

### Embeddings
- Text embedding models (text-embedding-ada-002, etc.)

## Model Naming

When using Azure OpenAI through OneLLM, prefix your model names with `azure/`:

- `azure/gpt-4o-mini`
- `azure/gpt-4-turbo`
- `azure/o4-mini`
- `azure/whisper-1`
- `azure/dall-e-3`

The model name after the prefix should match either:
1. A deployment name defined in your `azure.json` configuration
2. A standard Azure OpenAI deployment name

## Advanced Configuration

### Multiple Deployments

The Azure provider supports using multiple deployments with different configurations:

```json
{
    "deployment": {
        "fast-model": {
            "endpoint": "https://eastus-resource.openai.azure.com/",
            "deployment": "gpt-35-turbo",
            "subscription_key": "eastus-key",
            "api_version": "2024-12-01-preview"
        },
        "powerful-model": {
            "endpoint": "https://westus-resource.openai.azure.com/",
            "deployment": "gpt-4-turbo",
            "subscription_key": "westus-key",
            "api_version": "2024-12-01-preview"
        }
    }
}
```

### API Versions

You can specify different API versions for different deployments or set a default:

```python
azure = get_provider("azure", api_version="2024-12-01-preview")
```

## Error Handling

The Azure provider handles Azure-specific errors and maps them to OneLLM's standard error types:

- `AuthenticationError`: Invalid API key or authentication issues
- `RateLimitError`: Rate limit exceeded
- `InvalidRequestError`: Invalid request parameters
- `ResourceNotFoundError`: Deployment not found

## Best Practices

1. **Use deployment-specific configurations** for better control over routing and API versions
2. **Store your azure.json securely** and never commit it to version control
3. **Use secondary keys** for key rotation without downtime
4. **Monitor usage** through Azure Portal to track costs and performance
5. **Set appropriate timeouts** for long-running operations

## Limitations

- File upload/download operations are not supported (Azure uses blob storage separately)
- Some OpenAI-specific features may not be available in Azure OpenAI
- API versions and available features depend on your Azure region and deployment
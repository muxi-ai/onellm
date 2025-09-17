---
layout: default
title: AWS Bedrock
parent: Providers
nav_order: 4
---

# AWS Bedrock Provider

The AWS Bedrock provider enables access to foundation models from multiple providers (Anthropic, Meta, Mistral, Amazon, AI21 Labs, Cohere) through AWS's fully managed service.

## Installation

```bash
# Install OneLLM with Bedrock support
pip install "onellm[bedrock]"

# Or install boto3 separately
pip install boto3
```

## Configuration

### AWS Credentials

The Bedrock provider supports multiple authentication methods:

1. **AWS CLI Configuration** (Recommended)
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

2. **Environment Variables**
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

3. **AWS Profile**
   ```python
   from onellm import Client

   client = Client()
   # Uses profile from bedrock.json or default profile
   ```

4. **IAM Role** (for EC2/Lambda/ECS)
   - Automatically uses instance/task role

### bedrock.json Configuration

Create a `bedrock.json` file in your project root:

```json
{
    "profile": "bedrock",
    "region": "us-east-1"
}
```

### Required IAM Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Converse",
                "bedrock:ConverseStream"
            ],
            "Resource": "*"
        }
    ]
}
```

## Model Access

**Important**: AWS Bedrock requires explicit model access. You must request access to models in the AWS Console:

1. Navigate to Amazon Bedrock in AWS Console
2. Go to "Model access"
3. Request access to desired models
4. Wait for approval (usually instant for most models)

## Usage Examples

### Basic Chat Completion

```python
import asyncio
from onellm import Client

client = Client()

async def main():
    response = await client.chat.completions.create(
        model="bedrock/claude-3-5-sonnet",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        max_tokens=200,
        temperature=0.7
    )

    print(response.choices[0].message.content)

asyncio.run(main())
```

### Streaming Response

```python
async def stream_example():
    stream = await client.chat.completions.create(
        model="bedrock/claude-3-haiku",  # Faster model for streaming
        messages=[{"role": "user", "content": "Write a story about a robot."}],
        max_tokens=500,
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### Multi-Modal (Vision)

```python
import base64

async def vision_example():
    # Read and encode image
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = await client.chat.completions.create(
        model="bedrock/claude-3-5-sonnet",  # or nova-pro
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }],
        max_tokens=300
    )

    print(response.choices[0].message.content)
```

### Embeddings

```python
async def embedding_example():
    response = await client.embeddings.create(
        model="bedrock/titan-embed-text-v2",
        input="The quick brown fox jumps over the lazy dog."
    )

    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    print(f"First 5 values: {response.data[0].embedding[:5]}")
```

## Supported Models

### Chat Models

#### Anthropic Claude
- `bedrock/claude-3-5-sonnet` - Latest, most capable
- `bedrock/claude-3-5-haiku` - Fast, efficient
- `bedrock/claude-3-opus` - Most powerful (legacy)
- `bedrock/claude-3-sonnet` - Balanced
- `bedrock/claude-3-haiku` - Fastest

#### Meta Llama
- `bedrock/llama3-2-90b` - Large multimodal
- `bedrock/llama3-2-11b` - Medium multimodal
- `bedrock/llama3-2-3b` - Small efficient
- `bedrock/llama3-2-1b` - Tiny edge model
- `bedrock/llama3-1-405b` - Largest model
- `bedrock/llama3-1-70b` - Large model
- `bedrock/llama3-1-8b` - Medium model

#### Amazon Nova
- `bedrock/nova-pro` - Multimodal reasoning
- `bedrock/nova-lite` - Fast, cost-effective
- `bedrock/nova-micro` - Ultra-fast responses

#### Mistral
- `bedrock/mistral-7b` - Efficient model
- `bedrock/mixtral-8x7b` - Mixture of experts
- `bedrock/mistral-large` - Most capable

#### Others
- `bedrock/command-r` - Cohere Command R
- `bedrock/command-r-plus` - Cohere Command R+
- `bedrock/jamba-1-5-large` - AI21 Jamba Large
- `bedrock/jamba-1-5-mini` - AI21 Jamba Mini

### Embedding Models
- `bedrock/titan-embed-text` - Amazon Titan Embeddings v1
- `bedrock/titan-embed-text-v2` - Amazon Titan Embeddings v2
- `bedrock/embed-english` - Cohere English embeddings
- `bedrock/embed-multilingual` - Cohere multilingual embeddings

## Advanced Features

### Cross-Region Inference

Bedrock can automatically route requests to the best available region:

```python
response = await client.chat.completions.create(
    model="bedrock/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    # Cross-region inference is enabled by default
)
```

### Using Different AWS Regions

```python
# Method 1: Via bedrock.json
{
    "profile": "default",
    "region": "eu-west-1"
}

# Method 2: Environment variable
export AWS_DEFAULT_REGION="eu-west-1"

# Method 3: When initializing provider (requires custom client setup)
```

### Using Full Model IDs

You can also use full Bedrock model IDs:

```python
response = await client.chat.completions.create(
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Cost Optimization

1. **Choose the right model size** - Use smaller models when possible
2. **Use streaming** - Get responses faster and stop generation early if needed
3. **Batch requests** - Bedrock supports batch processing for 50% cost reduction
4. **Monitor usage** - Use CloudWatch to track token usage

## Common Issues

### Model Access Denied
```
Error: Model access denied: The requested model anthropic.claude-3-opus-20240229-v1:0 is not supported for inference in your account.
```
**Solution**: Request model access in AWS Bedrock console

### Rate Limits
```
Error: Too many requests, please try again later.
```
**Solution**: Implement retry logic or request quota increase

### Region Availability
Not all models are available in all regions. Check AWS documentation for model availability by region.

## Limitations

- No file upload/download support (images must be base64-encoded in messages)
- No explicit JSON mode (use prompt engineering or tool calling)
- Audio/video input not currently supported
- Model-specific features may vary

## Best Practices

1. **Use the Converse API** - The provider uses Bedrock's unified Converse API for consistency
2. **Handle errors gracefully** - Implement retry logic for transient errors
3. **Monitor costs** - Set up CloudWatch alarms for usage
4. **Choose appropriate models** - Balance capability vs. cost
5. **Request model access early** - Some models require manual approval

## See Also

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Model Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Available Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)

#!/usr/bin/env python3
"""
Test Vertex AI with legacy models and different endpoint formats.
"""

import json
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests

# Load service account
with open('tests/artifacts/vertexai.json', 'r') as f:
    service_account_info = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Refresh token
request = Request()
credentials.refresh(request)

project_id = service_account_info['project_id']
headers = {
    'Authorization': f'Bearer {credentials.token}',
    'Content-Type': 'application/json'
}

print(f"🔍 Testing various Vertex AI endpoints and models")
print(f"Project ID: {project_id}")

# Test different endpoint formats and models
test_cases = [
    # Gemini models with different endpoint formats
    {
        "name": "Gemini 1.5 Flash (generateContent)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/gemini-1.5-flash:generateContent",
        "data": {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
    },
    {
        "name": "Gemini Pro (generateContent)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/gemini-pro:generateContent",
        "data": {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
    },
    # Legacy text-bison model
    {
        "name": "Text Bison (predict)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/text-bison:predict",
        "data": {
            "instances": [{"prompt": "Hello, how are you?"}],
            "parameters": {"maxOutputTokens": 10}
        }
    },
    {
        "name": "Text Bison 001 (predict)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/text-bison-001:predict",
        "data": {
            "instances": [{"prompt": "Hello, how are you?"}],
            "parameters": {"maxOutputTokens": 10}
        }
    },
    # Chat bison model
    {
        "name": "Chat Bison (predict)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/chat-bison:predict",
        "data": {
            "instances": [{
                "context": "",
                "examples": [],
                "messages": [{"author": "user", "content": "Hello"}]
            }],
            "parameters": {"maxOutputTokens": 10}
        }
    },
    # Try different regions
    {
        "name": "Gemini Pro (europe-west4)",
        "url": f"https://europe-west4-aiplatform.googleapis.com/v1/projects/{project_id}/locations/europe-west4/publishers/google/models/gemini-pro:generateContent",
        "data": {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
    },
    # Try embedding model
    {
        "name": "Text Embedding Gecko (predict)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/textembedding-gecko:predict",
        "data": {
            "instances": [{"content": "Hello world"}]
        }
    },
    {
        "name": "Text Embedding Gecko 001 (predict)",
        "url": f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/textembedding-gecko-001:predict",
        "data": {
            "instances": [{"content": "Hello world"}]
        }
    }
]

success_count = 0
for test in test_cases:
    print(f"\n🧪 Testing: {test['name']}")
    print(f"   URL: {test['url'].split('/models/')[1] if '/models/' in test['url'] else test['url']}")

    response = requests.post(test['url'], json=test['data'], headers=headers)
    print(f"   Status: {response.status_code}")

    if response.status_code == 200:
        success_count += 1
        result = response.json()
        print(f"   ✅ SUCCESS!")

        # Extract response based on model type
        if 'candidates' in result:
            # Gemini response format
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"   Response: {text}")
        elif 'predictions' in result:
            # Legacy model response format
            predictions = result['predictions'][0]
            if 'content' in predictions:
                print(f"   Response: {predictions['content']}")
            elif 'embeddings' in predictions:
                print(f"   Embeddings generated (length: {len(predictions['embeddings']['values'])})")
            else:
                print(f"   Response: {str(predictions)[:100]}...")
    else:
        error_msg = "Unknown error"
        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', response.text)[:150]
        except:
            error_msg = response.text[:150]
        print(f"   ❌ Error: {error_msg}")

print(f"\n📊 Summary: {success_count}/{len(test_cases)} models worked")

if success_count == 0:
    print("\n⚠️  No models are accessible. Possible issues:")
    print("1. The project might not have access to these models")
    print("2. You might need to enable additional APIs:")
    print("   - Go to https://console.cloud.google.com/apis/library")
    print("   - Search for and enable 'Vertex AI API'")
    print("   - Also check 'Generative Language API'")
    print("3. The service account might need additional roles:")
    print("   - Vertex AI User")
    print("   - Vertex AI Service Agent")
    print("4. Some models might require allowlist access")
    print("   - Check https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini")

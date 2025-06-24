#!/usr/bin/env python3
"""
Test Vertex AI API directly to check access and available models.
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
location = 'us-central1'

print(f"🔍 Testing Vertex AI API directly")
print(f"Project ID: {project_id}")
print(f"Location: {location}")
print(f"Service Account: {service_account_info['client_email']}")

# Test 1: Check if Vertex AI API is enabled
print("\n1️⃣ Checking Vertex AI API status...")
api_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/aiplatform.googleapis.com"
headers = {
    'Authorization': f'Bearer {credentials.token}',
    'Content-Type': 'application/json'
}

response = requests.get(api_url, headers=headers)
if response.status_code == 200:
    api_info = response.json()
    state = api_info.get('state', 'UNKNOWN')
    print(f"  API State: {state}")
    if state != 'ENABLED':
        print("  ⚠️  Vertex AI API is not enabled!")
        print("  Enable it at: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
else:
    print(f"  ❌ Could not check API status: {response.status_code}")
    print(f"  Response: {response.text[:200]}")

# Test 2: List available models (if possible)
print("\n2️⃣ Trying to list available models...")
models_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/models"
response = requests.get(models_url, headers=headers)
print(f"  Response status: {response.status_code}")
if response.status_code == 200:
    models = response.json()
    print(f"  Models found: {len(models.get('models', []))}")
    for model in models.get('models', [])[:5]:  # Show first 5
        print(f"    - {model.get('name', 'Unknown')}")
else:
    print(f"  Response: {response.text[:200]}")

# Test 3: Try a direct API call to Gemini
print("\n3️⃣ Testing direct Gemini API call...")
gemini_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/gemini-1.5-flash:generateContent"
data = {
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    "generationConfig": {"maxOutputTokens": 10}
}

response = requests.post(gemini_url, json=data, headers=headers)
print(f"  Response status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print("  ✅ SUCCESS! Gemini API is accessible")
    if 'candidates' in result:
        text = result['candidates'][0]['content']['parts'][0]['text']
        print(f"  Response: {text}")
else:
    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
    error_msg = error_data.get('error', {}).get('message', response.text[:200])
    print(f"  ❌ Error: {error_msg}")

# Test 4: Check IAM permissions
print("\n4️⃣ Checking IAM permissions...")
test_permissions_url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:testIamPermissions"
permissions_data = {
    "permissions": [
        "aiplatform.endpoints.predict",
        "aiplatform.models.predict",
        "serviceusage.services.use"
    ]
}

response = requests.post(test_permissions_url, json=permissions_data, headers=headers)
if response.status_code == 200:
    result = response.json()
    granted = result.get('permissions', [])
    print(f"  Granted permissions: {len(granted)}/{len(permissions_data['permissions'])}")
    for perm in permissions_data['permissions']:
        status = "✅" if perm in granted else "❌"
        print(f"    {status} {perm}")
else:
    print(f"  ❌ Could not check permissions: {response.status_code}")

print("\n📋 Summary:")
print("If the API is not enabled or you're missing permissions, please:")
print("1. Go to https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
print("2. Enable the Vertex AI API")
print("3. Ensure your service account has the 'Vertex AI User' role")
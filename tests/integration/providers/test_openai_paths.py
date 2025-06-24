#!/usr/bin/env python3
"""
Test OpenAI provider file upload with cross-platform path fix.
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_openai_file_upload():
    """Test OpenAI file upload with os.path.basename fix."""
    print("🔍 Testing OpenAI file upload with cross-platform path fix...")
    
    from onellm import OpenAI
    
    try:
        client = OpenAI()
        
        # Create a temporary file with a path that includes directory separators
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for file upload")
            temp_path = f.name
        
        print(f"Test file path: {temp_path}")
        print(f"Expected filename: {os.path.basename(temp_path)}")
        
        try:
            # Test file upload
            print("\nTesting file upload...")
            file_obj = client.files.create(
                file=temp_path,
                purpose="assistants"
            )
            
            print(f"✅ File uploaded successfully!")
            print(f"File ID: {file_obj.id}")
            print(f"Filename: {file_obj.filename}")
            
            # Clean up - delete the uploaded file
            client.files.delete(file_obj.id)
            print("✅ File deleted successfully")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}...")
        return False

if __name__ == "__main__":
    result = test_openai_file_upload()
    print(f"\n🎯 Test result: {'PASSED' if result else 'FAILED'}")
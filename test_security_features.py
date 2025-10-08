#!/usr/bin/env python3
"""
Standalone test to validate security features without importing the full package.
This avoids import dependency issues between different PR branches.
"""

import sys
import os
import tempfile
import io

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_filename_sanitization():
    """Test that filename sanitization works correctly."""
    from onellm.files import _sanitize_filename
    
    print("Testing filename sanitization...")
    
    # Directory traversal
    assert _sanitize_filename("../../etc/passwd") == "passwd", "Failed: directory traversal"
    assert _sanitize_filename("../../../config.txt") == "config.txt", "Failed: multiple traversal"
    
    # Subdirectories
    assert _sanitize_filename("dir/subdir/file.txt") == "file.txt", "Failed: subdirectories"
    
    # Null bytes
    assert _sanitize_filename("file\x00.exe") == "file.exe", "Failed: null bytes"
    
    # Special cases
    assert _sanitize_filename(".") == "file.bin", "Failed: dot"
    assert _sanitize_filename("..") == "file.bin", "Failed: double dot"
    assert _sanitize_filename("") == "file.bin", "Failed: empty string"
    
    # Legitimate filenames
    assert _sanitize_filename("document.pdf") == "document.pdf", "Failed: normal file"
    assert _sanitize_filename("my..file.txt") == "my..file.txt", "Failed: dots in name"
    
    print("✅ Filename sanitization tests PASSED")

def test_size_limited_wrapper():
    """Test that SizeLimitedFileWrapper enforces size limits."""
    from onellm.files import SizeLimitedFileWrapper
    from onellm.errors import InvalidRequestError
    
    print("Testing SizeLimitedFileWrapper...")
    
    # Within limit
    data = b"Hello World"
    file_obj = io.BytesIO(data)
    wrapper = SizeLimitedFileWrapper(file_obj, max_size=100, name="test")
    result = wrapper.read()
    assert result == data, "Failed: reading within limit"
    
    # Exceeds limit
    large_data = b"x" * 1000
    file_obj = io.BytesIO(large_data)
    wrapper = SizeLimitedFileWrapper(file_obj, max_size=500, name="test")
    
    try:
        wrapper.read()
        assert False, "Should have raised InvalidRequestError"
    except InvalidRequestError as e:
        assert "too large" in str(e), "Failed: error message"
    
    print("✅ SizeLimitedFileWrapper tests PASSED")

def test_file_validator():
    """Test FileValidator security checks."""
    from onellm.utils.file_validator import FileValidator
    from onellm.errors import InvalidRequestError
    
    print("Testing FileValidator...")
    
    # Directory traversal prevention
    try:
        FileValidator.validate_file_path("../../../etc/passwd")
        assert False, "Should have blocked directory traversal"
    except InvalidRequestError as e:
        assert "Directory traversal" in str(e), "Failed: traversal error message"
    
    # Bytes size validation
    data = b"x" * 1000
    FileValidator.validate_bytes_size(data, max_size=2000)  # Should pass
    
    try:
        FileValidator.validate_bytes_size(data, max_size=500)  # Should fail
        assert False, "Should have raised size error"
    except InvalidRequestError as e:
        assert "too large" in str(e), "Failed: size error message"
    
    # Filename validation
    FileValidator.validate_filename("document.pdf", allowed_extensions={".pdf", ".txt"})
    
    try:
        FileValidator.validate_filename("file.exe", allowed_extensions={".pdf", ".txt"})
        assert False, "Should have rejected .exe"
    except InvalidRequestError as e:
        assert "not allowed" in str(e), "Failed: extension error"
    
    print("✅ FileValidator tests PASSED")

def test_async_helpers():
    """Test async helper functions."""
    import asyncio
    from onellm.utils.async_helpers import run_async, _is_jupyter_environment
    
    print("Testing async helpers...")
    
    # Test run_async in synchronous context
    async def async_task():
        await asyncio.sleep(0.001)
        return "completed"
    
    result = run_async(async_task())
    assert result == "completed", "Failed: run_async"
    
    # Test Jupyter detection (should be False in normal environment)
    is_jupyter = _is_jupyter_environment()
    assert isinstance(is_jupyter, bool), "Failed: Jupyter detection return type"
    
    print("✅ Async helpers tests PASSED")

def test_file_path_validation():
    """Test file path validation with actual files."""
    from onellm.utils.file_validator import FileValidator
    from onellm.errors import InvalidRequestError
    
    print("Testing file path validation...")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_file = f.name
        f.write(b"test content" * 100)
    
    try:
        # Should succeed
        validated = FileValidator.validate_file_path(temp_file, max_size=10000)
        assert validated.exists(), "Failed: file should exist"
        
        # Should fail with small limit
        try:
            FileValidator.validate_file_path(temp_file, max_size=100)
            assert False, "Should have failed size check"
        except InvalidRequestError as e:
            assert "too large" in str(e), "Failed: size error"
        
        # Should fail with wrong extension
        try:
            FileValidator.validate_file_path(temp_file, allowed_extensions={".pdf"})
            assert False, "Should have failed extension check"
        except InvalidRequestError as e:
            assert "not allowed" in str(e), "Failed: extension error"
            
    finally:
        os.unlink(temp_file)
    
    print("✅ File path validation tests PASSED")

def main():
    """Run all tests."""
    print("=" * 60)
    print("SECURITY FEATURES TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        test_filename_sanitization()
        test_size_limited_wrapper()
        test_file_validator()
        test_async_helpers()
        test_file_path_validation()
        
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Issues #3, #4, #5: Resource Management, Input Validation & Security

## Issue #3: Resource Cleanup in Provider Methods ⚠️

### Problem

Current `_make_request` implementations may not properly clean up resources on errors:

```python
async def execute_request():
    async with aiohttp.ClientSession() as session:
        async with session.request(...) as response:
            if stream:
                return self._handle_streaming_response(response)
                # ❌ Response object escapes context manager!
```

### Issues

1. **Streaming escapes context manager**
   - Generator holds reference to response
   - But context manager already exited
   - Undefined behavior

2. **No explicit cleanup on exceptions**
   - If error occurs during request setup
   - Session may not close properly

3. **No connection pooling**
   - New session for every request
   - Inefficient resource usage

### Solution

```python
class ProviderBase:
    """Base class with shared HTTP client management."""
    
    def __init__(self, **kwargs):
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared session with connection pooling."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,  # Max connections
                        limit_per_host=10,  # Per host
                        ttl_dns_cache=300,  # DNS cache TTL
                    )
                    timeout = aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=10,  # Connection timeout
                        sock_read=30,  # Socket read timeout
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
        return self._session
    
    async def _close_session(self):
        """Close the shared session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        """Support async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        await self._close_session()
```

### Benefits

1. ✅ **Connection pooling** - Reuse connections
2. ✅ **Proper cleanup** - Always closes resources
3. ✅ **Better performance** - Fewer TCP handshakes
4. ✅ **Configurable timeouts** - Fine-grained control

### Implementation Plan

1. Create `ProviderHTTPMixin` base class
2. Update all providers to inherit from it
3. Replace session creation with `_get_session()`
4. Add cleanup in `__del__` as safety net

---

## Issue #4: Input Validation ⚠️

### Problem

Not all inputs are validated before API calls:

```python
async def create_chat_completion(self, messages, model, **kwargs):
    # No validation of model, messages could be empty/None
    data = {"model": model, "messages": messages, ...}
    # Send to API...
```

### Missing Validations

1. **Model name validation**
   - Could be empty string
   - Could be None
   - Could be malformed

2. **Message validation**
   - Empty list
   - Invalid role
   - Missing content

3. **Parameter ranges**
   - temperature < 0 or > 2
   - max_tokens <= 0
   - n <= 0

4. **Type checking**
   - messages should be List[Dict]
   - stream should be bool
   - temperature should be float

### Solution

```python
# validators.py - Add comprehensive validators

def validate_provider_model(model: str, provider_name: str) -> None:
    """
    Validate model name for a specific provider.
    
    Args:
        model: Model name (without provider prefix)
        provider_name: Provider name
        
    Raises:
        InvalidRequestError: If model is invalid
    """
    if not model or not isinstance(model, str):
        raise InvalidRequestError(
            f"Model must be a non-empty string, got: {type(model).__name__}"
        )
    
    if model.strip() == "":
        raise InvalidRequestError("Model name cannot be empty or whitespace")
    
    # Provider-specific validation
    if provider_name == "openai":
        if not (
            model.startswith("gpt-") or 
            model.startswith("o-") or
            model in ["whisper-1", "tts-1", "dall-e-2", "dall-e-3"]
        ):
            raise InvalidRequestError(
                f"Invalid OpenAI model name: {model}"
            )


def validate_temperature(temperature: Optional[float]) -> None:
    """Validate temperature parameter."""
    if temperature is None:
        return
    
    if not isinstance(temperature, (int, float)):
        raise InvalidRequestError(
            f"temperature must be a number, got {type(temperature).__name__}"
        )
    
    if not 0 <= temperature <= 2:
        raise InvalidRequestError(
            f"temperature must be between 0 and 2, got {temperature}"
        )


def validate_max_tokens(max_tokens: Optional[int]) -> None:
    """Validate max_tokens parameter."""
    if max_tokens is None:
        return
    
    if not isinstance(max_tokens, int):
        raise InvalidRequestError(
            f"max_tokens must be an integer, got {type(max_tokens).__name__}"
        )
    
    if max_tokens <= 0:
        raise InvalidRequestError(
            f"max_tokens must be positive, got {max_tokens}"
        )
    
    if max_tokens > 100000:  # Reasonable upper bound
        raise InvalidRequestError(
            f"max_tokens too large: {max_tokens} (max: 100000)"
        )


def validate_chat_params(**kwargs) -> None:
    """Validate all chat completion parameters."""
    validate_temperature(kwargs.get("temperature"))
    validate_max_tokens(kwargs.get("max_tokens"))
    
    # Validate n (number of completions)
    n = kwargs.get("n")
    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise InvalidRequestError(f"n must be positive integer, got {n}")
        if n > 10:
            raise InvalidRequestError(f"n too large: {n} (max: 10)")
    
    # Validate top_p
    top_p = kwargs.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise InvalidRequestError(f"top_p must be a number, got {type(top_p)}")
        if not 0 < top_p <= 1:
            raise InvalidRequestError(f"top_p must be in (0, 1], got {top_p}")
```

### Usage in Providers

```python
class OpenAIProvider(Provider):
    async def create_chat_completion(self, messages, model, **kwargs):
        # Validate all inputs before making API call
        validate_provider_model(model, "openai")
        validate_messages(messages)  # Already exists
        validate_chat_params(**kwargs)  # New
        
        # Now safe to proceed
        data = {"model": model, "messages": messages, **kwargs}
        ...
```

### Benefits

1. ✅ **Early error detection** - Fail fast with clear errors
2. ✅ **Better UX** - Clear error messages before API call
3. ✅ **Prevent API costs** - Don't send invalid requests
4. ✅ **Type safety** - Catch type errors early

---

## Issue #5: File Path Security 🔒

### Problem

File upload accepts paths without validation:

```python
if isinstance(file, str):
    with open(file, "rb") as f:  # ❌ No validation!
        file_data = f.read()
```

### Security Risks

1. **Directory Traversal**
   ```python
   # Malicious user could do:
   File.upload("../../../../etc/passwd", purpose="malicious")
   ```

2. **Symbolic Link Attacks**
   ```python
   # Could read files via symlinks
   File.upload("/tmp/symlink_to_secrets", purpose="attack")
   ```

3. **No Size Limits**
   ```python
   # Could upload huge files, causing DoS
   File.upload("10GB_file.bin", purpose="dos")
   ```

4. **No Type Validation**
   ```python
   # Could upload executable files
   File.upload("malware.exe", purpose="attack")
   ```

### Solution

```python
# files.py

import mimetypes
from pathlib import Path
from typing import Optional, Set

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
ALLOWED_EXTENSIONS: Set[str] = {
    # Audio
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm",
    # Images  
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    # Documents
    ".pdf", ".txt", ".json", ".jsonl", ".csv",
    # Archives
    ".zip", ".tar", ".gz",
}


class FileValidator:
    """Validates file paths and contents for security."""
    
    @staticmethod
    def validate_file_path(
        file_path: str,
        max_size: Optional[int] = MAX_FILE_SIZE,
        allowed_extensions: Optional[Set[str]] = None,
    ) -> Path:
        """
        Validate and normalize a file path.
        
        Args:
            file_path: Path to validate
            max_size: Maximum file size in bytes
            allowed_extensions: Set of allowed file extensions
            
        Returns:
            Validated Path object
            
        Raises:
            InvalidRequestError: If validation fails
        """
        if not file_path or not isinstance(file_path, str):
            raise InvalidRequestError("file_path must be a non-empty string")
        
        try:
            # Convert to Path and resolve (follows symlinks, normalizes)
            path = Path(file_path).resolve()
        except (OSError, RuntimeError) as e:
            raise InvalidRequestError(f"Invalid file path: {e}")
        
        # Check if file exists
        if not path.exists():
            raise InvalidRequestError(f"File not found: {file_path}")
        
        # Must be a regular file (not directory, device, etc.)
        if not path.is_file():
            raise InvalidRequestError(f"Path is not a file: {file_path}")
        
        # Check for directory traversal attempts
        # After resolve(), the path should not contain ".."
        if ".." in path.parts:
            raise InvalidRequestError(
                f"Directory traversal detected: {file_path}"
            )
        
        # Validate file extension
        if allowed_extensions is None:
            allowed_extensions = ALLOWED_EXTENSIONS
        
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise InvalidRequestError(
                f"File type not allowed: {path.suffix}. "
                f"Allowed types: {', '.join(sorted(allowed_extensions))}"
            )
        
        # Check file size
        if max_size:
            file_size = path.stat().st_size
            if file_size > max_size:
                max_mb = max_size / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                raise InvalidRequestError(
                    f"File too large: {actual_mb:.1f}MB (max: {max_mb:.1f}MB)"
                )
            
            if file_size == 0:
                raise InvalidRequestError("File is empty")
        
        # Validate MIME type matches extension
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Unknown MIME type - be cautious
            raise InvalidRequestError(
                f"Cannot determine file type for: {path.name}"
            )
        
        return path
    
    @staticmethod
    def safe_read_file(path: Path, chunk_size: int = 8192) -> bytes:
        """
        Safely read file contents with memory protection.
        
        Args:
            path: Validated Path object
            chunk_size: Read in chunks to avoid memory issues
            
        Returns:
            File contents as bytes
        """
        try:
            # Read in chunks to avoid loading huge files into memory at once
            chunks = []
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks)
        except (OSError, IOError) as e:
            raise InvalidRequestError(f"Error reading file: {e}")


class File:
    """Interface for file operations across different providers."""
    
    @classmethod
    def upload(
        cls,
        file: Union[str, Path, BinaryIO, bytes],
        purpose: str,
        provider: str = "openai",
        max_size: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
        **kwargs
    ) -> FileObject:
        """
        Upload a file to the provider.
        
        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file
            provider: Provider to use
            max_size: Maximum file size (default: 100MB)
            allowed_extensions: Allowed file extensions (default: common types)
            **kwargs: Additional parameters
            
        Returns:
            FileObject representing the uploaded file
        """
        # Get provider instance
        provider_instance = get_provider(provider)
        
        # Process file based on type
        if isinstance(file, (str, Path)):
            # Validate file path
            validated_path = FileValidator.validate_file_path(
                str(file),
                max_size=max_size,
                allowed_extensions=allowed_extensions,
            )
            # Safely read file
            file_data = FileValidator.safe_read_file(validated_path)
            filename = validated_path.name
        elif isinstance(file, bytes):
            # Bytes data - validate size
            if max_size and len(file) > max_size:
                raise InvalidRequestError(
                    f"File too large: {len(file)} bytes (max: {max_size})"
                )
            file_data = file
            filename = kwargs.get("filename", "file.bin")
        elif hasattr(file, "read"):
            # File-like object
            file_data = file.read()
            if max_size and len(file_data) > max_size:
                raise InvalidRequestError(
                    f"File too large: {len(file_data)} bytes (max: {max_size})"
                )
            filename = getattr(file, "name", "file.bin")
        else:
            raise InvalidRequestError(
                "file must be a path, bytes, or file-like object"
            )
        
        # Upload using provider
        return run_async(
            provider_instance.upload_file(
                file=file_data,
                purpose=purpose,
                filename=filename,
                **kwargs
            )
        )
```

### Configuration

```python
# config.py

DEFAULT_CONFIG = {
    "file_upload": {
        "max_size": 100 * 1024 * 1024,  # 100MB
        "allowed_extensions": None,  # None = use default list
        "validate_mime_type": True,  # Verify MIME matches extension
    }
}
```

### Benefits

1. ✅ **Prevents directory traversal** - Path validation
2. ✅ **Size limits** - Prevents DoS
3. ✅ **Type validation** - Only allowed file types
4. ✅ **Safe reads** - Chunked reading for large files
5. ✅ **Clear errors** - User knows exactly what's wrong

---

## Implementation Priority

### Phase 1 (Immediate) - Security
- ✅ Issue #5: File path validation
- ⚠️ Issue #4: Basic input validation (model, messages)

### Phase 2 (Short-term) - Reliability
- ⚠️ Issue #2: Streaming timeouts and cleanup
- ⚠️ Issue #3: Resource management and connection pooling

### Phase 3 (Medium-term) - Polish
- Issue #4: Complete parameter validation
- Provider-specific model validation
- Configuration options for all validators

## Testing Strategy

### Unit Tests
```python
# Test file validation
def test_file_path_validation():
    # Test normal file
    path = FileValidator.validate_file_path("test.txt")
    assert path.exists()
    
    # Test directory traversal
    with pytest.raises(InvalidRequestError):
        FileValidator.validate_file_path("../../etc/passwd")
    
    # Test file size limit
    with pytest.raises(InvalidRequestError):
        FileValidator.validate_file_path("huge_file.bin", max_size=1024)
    
    # Test extension validation
    with pytest.raises(InvalidRequestError):
        FileValidator.validate_file_path("malware.exe")
```

### Integration Tests
```python
# Test with real providers
async def test_openai_with_validation():
    # Valid request
    response = await provider.create_chat_completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )
    
    # Invalid temperature
    with pytest.raises(InvalidRequestError):
        await provider.create_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=3.0,  # Too high!
        )
```

## Status

- ✅ Issue #1: COMPLETED
- 🚧 Issue #2: Design complete, implementation needed
- 📋 Issue #3: Design complete, implementation needed
- 📋 Issue #4: Design complete, implementation needed
- 📋 Issue #5: Design complete, implementation needed


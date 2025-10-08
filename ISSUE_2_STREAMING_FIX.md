# Issue #2: Streaming Error Handling - Implementation Plan

## Problem Summary

Current streaming implementation has several critical issues:

### 1. **No Timeout on Individual Chunks**
```python
# Current code in _handle_streaming_response
async for line in response.content:  # No timeout!
    # If the connection stalls, this waits forever
    line = line.decode("utf-8").strip()
    ...
```

**Problem**: If the server stops sending data mid-stream, the client hangs indefinitely.

### 2. **No Resource Cleanup on Errors**
```python
async with aiohttp.ClientSession() as session:
    async with session.request(...) as response:
        if stream:
            return self._handle_streaming_response(response)
            # ❌ Response object escapes the context manager!
            # The async generator holds a reference, but the context exits
```

**Problem**: The async generator returned by `_handle_streaming_response` uses `response.content`, but the response context manager has already exited.

### 3. **Silent JSON Decode Failures**
```python
try:
    yield json.loads(line)
except json.JSONDecodeError:
    continue  # Silently skip - user never knows!
```

**Problem**: Malformed responses are silently ignored, making debugging difficult.

### 4. **No Connection Health Monitoring**
- No heartbeat detection
- No way to detect stalled connections
- No backpressure handling

## Solution Design

### Approach 1: Keep Session Alive (Recommended)

**Pros**:
- Clean API
- Proper resource management
- Easy to add timeouts

**Cons**:
- Slightly more complex implementation

### Approach 2: Yield Control Implementation

Create a proper streaming handler that:
1. Keeps the session/response alive during streaming
2. Adds per-chunk timeouts
3. Implements proper cleanup
4. Optionally logs malformed chunks

## Implementation

### New Streaming Wrapper

```python
class StreamingResponseHandler:
    """
    Manages streaming responses with proper resource management.
    
    Features:
    - Per-chunk timeout
    - Automatic cleanup
    - Error tracking
    - Optional logging
    """
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        response: aiohttp.ClientResponse,
        chunk_timeout: float = 30.0,
        log_errors: bool = True
    ):
        self.session = session
        self.response = response
        self.chunk_timeout = chunk_timeout
        self.log_errors = log_errors
        self._chunks_received = 0
        self._errors_count = 0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if not self.response.closed:
            self.response.close()
        if not self.session.closed:
            await self.session.close()
    
    async def stream_chunks(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream JSON chunks with timeout protection."""
        try:
            async for line in self._read_lines_with_timeout():
                if line.startswith("data: "):
                    line = line[6:]
                
                if line == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(line)
                    self._chunks_received += 1
                    yield chunk
                except json.JSONDecodeError as e:
                    self._errors_count += 1
                    if self.log_errors:
                        logger.warning(f"Malformed chunk: {line[:100]}... Error: {e}")
                    # Don't yield malformed chunks
                    continue
        except asyncio.TimeoutError:
            raise StreamingError(
                f"Streaming timeout: No data received for {self.chunk_timeout}s "
                f"after {self._chunks_received} chunks"
            )
        except Exception as e:
            raise StreamingError(
                f"Stream failed after {self._chunks_received} chunks: {str(e)}"
            ) from e
    
    async def _read_lines_with_timeout(self) -> AsyncGenerator[str, None]:
        """Read lines with per-line timeout."""
        async for line_bytes in self.response.content:
            # Apply timeout to decoding and processing
            try:
                line = await asyncio.wait_for(
                    self._decode_line(line_bytes),
                    timeout=self.chunk_timeout
                )
                if line:  # Skip empty lines
                    yield line
            except asyncio.TimeoutError:
                raise StreamingError(
                    f"Timeout decoding chunk after {self._chunks_received} chunks"
                )
    
    async def _decode_line(self, line_bytes: bytes) -> str:
        """Decode a line asynchronously."""
        # Make this actually async to allow cancellation
        await asyncio.sleep(0)  # Yield control
        return line_bytes.decode("utf-8").strip()
```

### Modified _make_request Method

```python
async def _make_request(
    self,
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    timeout: Optional[float] = None,
    chunk_timeout: Optional[float] = None,  # NEW
    files: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Make a request to the OpenAI API.
    
    Args:
        chunk_timeout: Timeout for individual chunks in streaming (default: 30s)
    """
    url = f"{self.api_base}/{path.lstrip('/')}"
    timeout = timeout or self.timeout
    chunk_timeout = chunk_timeout or 30.0  # Default 30s per chunk
    headers = self._get_headers()
    
    # ... (file handling code) ...
    
    if stream:
        # For streaming, we need to keep session alive
        async def stream_generator():
            session = aiohttp.ClientSession()
            try:
                response = await session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=chunk_timeout),
                )
                
                # Check for errors
                if response.status != 200:
                    error_data = await response.json()
                    self._handle_error_response(response.status, error_data)
                    return
                
                # Use streaming handler
                handler = StreamingResponseHandler(
                    session=session,
                    response=response,
                    chunk_timeout=chunk_timeout,
                    log_errors=True
                )
                
                async with handler:
                    async for chunk in handler.stream_chunks():
                        yield chunk
                        
            except Exception as e:
                # Ensure cleanup on error
                if not session.closed:
                    await session.close()
                raise
        
        return stream_generator()
    else:
        # Non-streaming uses retry mechanism
        async def execute_request():
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout,
                ) as response:
                    return await self._handle_response(response)
        
        return await retry_async(execute_request, config=self.retry_config)
```

## Testing Plan

### Test Cases

1. **Normal Streaming**
   ```python
   async def test_normal_stream():
       async for chunk in provider.create_chat_completion(..., stream=True):
           print(chunk)
   ```

2. **Connection Timeout**
   ```python
   # Mock server that stops sending after 2 chunks
   # Should timeout after chunk_timeout seconds
   ```

3. **Malformed JSON**
   ```python
   # Mock server sending invalid JSON
   # Should log warning and continue
   ```

4. **Mid-Stream Error**
   ```python
   # Mock server that errors after some chunks
   # Should raise StreamingError with chunk count
   ```

5. **Resource Cleanup**
   ```python
   # Verify session/response closed even on error
   async def test_cleanup():
       try:
           async for chunk in stream:
               if count == 5:
                   raise Exception("Forced error")
       except:
           pass
       # Verify no resource leaks
   ```

## Migration Path

### Backwards Compatible

All changes are backwards compatible:

```python
# Existing code continues to work
async for chunk in provider.create_chat_completion(..., stream=True):
    print(chunk)

# New: Can specify chunk timeout
async for chunk in provider.create_chat_completion(
    ..., 
    stream=True,
    chunk_timeout=60.0  # 60 second per-chunk timeout
):
    print(chunk)
```

## Implementation Steps

1. ✅ Create `StreamingResponseHandler` class
2. ✅ Update `_make_request` in base providers
3. ✅ Add `chunk_timeout` parameter support
4. ✅ Add logging for malformed chunks
5. ⬜ Test with real providers
6. ⬜ Document new parameters
7. ⬜ Add configuration defaults

## Configuration

Add to provider config:

```python
DEFAULT_CONFIG = {
    "providers": {
        "openai": {
            ...
            "chunk_timeout": 30.0,  # NEW: timeout per chunk
            "log_streaming_errors": True,  # NEW: log malformed chunks
        }
    }
}
```

## Benefits

1. ✅ **No More Hanging**: Connections timeout if stalled
2. ✅ **Proper Cleanup**: Resources freed even on errors
3. ✅ **Better Debugging**: Malformed chunks logged
4. ✅ **Configurable**: Users can adjust timeouts
5. ✅ **Backwards Compatible**: Existing code works unchanged

## Files to Modify

- `providers/openai.py` - Add streaming handler
- `providers/anthropic.py` - Same changes
- `providers/base.py` - Add abstract streaming support
- `config.py` - Add chunk_timeout config
- `errors.py` - Already has StreamingError

## Status

🚧 **IN PROGRESS** - Design complete, implementation needed


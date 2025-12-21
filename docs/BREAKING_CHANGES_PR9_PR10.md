# Breaking Changes: OneLLM Exception Renames (PR #9 + PR #10)

This document lists all breaking changes introduced by PR #9 and PR #10 for the muxi runtime droid to handle.

## Summary

Two categories of renames:
1. **Builtin shadowing fix** (PR #9): Renamed exceptions that shadowed Python builtins
2. **Brand rename** (PR #10): Renamed `MuxiLLM` prefix to `OneLLM`

## Exception Class Renames

| Old Name | New Name | Reason |
|----------|----------|--------|
| `TimeoutError` | `RequestTimeoutError` | Shadowed Python builtin `TimeoutError` |
| `PermissionError` | `PermissionDeniedError` | Shadowed Python builtin `PermissionError` |
| `MuxiLLMError` | `OneLLMError` | Brand rename |

## Import Changes Required

### Before
```python
from onellm.errors import TimeoutError, PermissionError, MuxiLLMError

try:
    response = ChatCompletion.create(...)
except TimeoutError:
    # handle timeout
except PermissionError:
    # handle permission denied
except MuxiLLMError:
    # handle any onellm error
```

### After
```python
from onellm.errors import RequestTimeoutError, PermissionDeniedError, OneLLMError

try:
    response = ChatCompletion.create(...)
except RequestTimeoutError:
    # handle timeout
except PermissionDeniedError:
    # handle permission denied
except OneLLMError:
    # handle any onellm error
```

## Affected Files in onellm Package

These files have been updated internally:
- `onellm/errors.py` - Class definitions
- `onellm/__init__.py` - Exports
- `onellm/providers/openai.py`
- `onellm/providers/azure.py`
- `onellm/providers/anthropic.py`
- `onellm/providers/mistral.py`
- `onellm/providers/bedrock.py`
- `onellm/utils/retry.py`
- `onellm/utils/fallback.py`
- `onellm/utils/streaming.py`

## Impact Assessment

**Low impact expected for muxi runtime** because:
1. The runtime likely catches `OneLLMError` (base class) rather than specific subclasses
2. The builtin shadowing fix (`TimeoutError`/`PermissionError`) is actually a bug fix - catching Python's builtin `TimeoutError` would have caught unrelated timeout exceptions

## Search Patterns for Downstream Code

To find code that needs updating, search for:
```bash
# Find old exception imports
rg "from onellm.errors import.*TimeoutError"
rg "from onellm.errors import.*PermissionError"
rg "from onellm.errors import.*MuxiLLMError"
rg "from onellm import.*MuxiLLMError"

# Find exception catches
rg "except.*TimeoutError"
rg "except.*PermissionError"
rg "except.*MuxiLLMError"
```

## Backward Compatibility (Not Implemented)

These PRs do **not** include backward compatibility aliases. If needed, add to `onellm/errors.py`:
```python
# Deprecated aliases for backward compatibility
TimeoutError = RequestTimeoutError
PermissionError = PermissionDeniedError
MuxiLLMError = OneLLMError
```

And add deprecation warnings in `__init__.py` if desired.

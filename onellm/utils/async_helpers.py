#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/onellm
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Async helper utilities for OneLLM.

This module provides utilities for safely running async code from synchronous
contexts, handling edge cases like existing event loops in Jupyter notebooks,
web frameworks, and other environments.
"""

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Safely run an async coroutine from synchronous code.
    
    This function handles the complexity of running async code in sync contexts,
    including environments that may already have running event loops (Jupyter,
    IPython, web frameworks, etc.).
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
        
    Raises:
        RuntimeError: If called from within an async context (use await instead)
        
    Examples:
        >>> async def get_data():
        ...     return "data"
        >>> result = run_async(get_data())
        >>> print(result)
        'data'
    """
    # Check if there's already a running event loop
    try:
        loop = asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        # No running loop
        has_running_loop = False
        loop = None

    # If there's a running loop, check if we're in Jupyter/IPython first
    if has_running_loop:
        # Check if we're in Jupyter/IPython environment
        if _is_jupyter_environment():
            # Jupyter environment - try to use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                # Reuse the running loop with nest_asyncio
                return loop.run_until_complete(coro)
            except ImportError:
                # nest_asyncio not available, provide helpful error
                raise RuntimeError(
                    "Detected Jupyter/IPython environment with a running event loop. "
                    "Please install nest_asyncio to use synchronous methods:\n"
                    "  pip install nest_asyncio\n"
                    "Or use async methods (acreate, aupload, etc.) with await."
                )
        else:
            # Not Jupyter - we're in an async context (e.g., FastAPI, async function)
            # This is a programming error - the caller should use await
            raise RuntimeError(
                "Cannot use synchronous method from async context. "
                "Use the async version (acreate, aupload, etc.) instead."
            )

    # No running loop - standard case
    # Use asyncio.run() which creates a new event loop, runs the coroutine, and cleans up
    return asyncio.run(coro)


def _is_jupyter_environment() -> bool:
    """
    Detect if we're running in a Jupyter notebook or IPython environment.
    
    Returns:
        True if in Jupyter/IPython, False otherwise
    """
    try:
        # Check if IPython module is available in sys.modules
        ipython_module = sys.modules.get('IPython')
        if ipython_module is None:
            return False

        # Try to import and call get_ipython function
        # This import could fail if the module is invalid, so we handle it separately
        try:
            from IPython import get_ipython
        except ImportError:
            # Module exists in sys.modules but import failed - not a valid IPython env
            return False

        ipython_instance = get_ipython()

        if ipython_instance is None:
            return False

        # Check if it's a ZMQInteractiveShell (Jupyter) or TerminalInteractiveShell (IPython)
        return ipython_instance.__class__.__name__ in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except (ImportError, AttributeError):
        return False


async def maybe_await(obj: Any) -> Any:
    """
    Await an object if it's awaitable, otherwise return it directly.
    
    This is useful for functions that can accept both sync and async callables.
    
    Args:
        obj: The object to potentially await
        
    Returns:
        The result after awaiting (if awaitable) or the object itself
        
    Examples:
        >>> async def async_func():
        ...     return "async"
        >>> def sync_func():
        ...     return "sync"
        >>> await maybe_await(async_func())
        'async'
        >>> await maybe_await(sync_func())
        'sync'
    """
    import inspect

    if inspect.isawaitable(obj):
        return await obj
    return obj

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for testing the onellm package.
"""

import asyncio
from functools import wraps


def run_async(func):
    """
    Decorator to run an async function in a synchronous context.

    This is useful for tests that need to call async functions but aren't themselves async.

    Args:
        func: The async function to run

    Returns:
        A synchronous wrapper around the async function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    return wrapper


class AsyncMockMixin:
    """
    Mixin class to help with mocking async context managers.
    """
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncContextManager:
    """
    Helper class to create async context managers for testing.

    Example:
        async_cm = AsyncContextManager(return_value=mock_response)
        with mock.patch('aiohttp.ClientSession', return_value=async_cm):
            result = await some_async_function()
    """
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncGeneratorMock:
    """
    Mock class for async generators.

    Example:
        mock_generator = AsyncGeneratorMock(["item1", "item2"])
        async for item in mock_generator:
            process(item)
    """
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item

import asyncio
"""
Tests specifically targeting uncovered lines in streaming.py.

This test file focuses on the json_stream_generator function's data_key handling in
non-dict JSON objects (line 86) and dict.get(data_key) operations (lines 93-94).
"""

import pytest

from onellm.utils.streaming import json_stream_generator


class MockGenerator:
    """A mock async generator class for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.mark.asyncio
async def test_json_stream_with_non_dict_data():
    """Test json_stream_generator with data_key and non-dict data (Line 86)."""
    # Source generator returning non-dict JSON values
    source = MockGenerator([
        '"string value"',  # String JSON
        '[1, 2, 3]',       # Array JSON
        'null',            # null JSON
        'true',            # boolean JSON
        '123',             # number JSON
    ])

    # Collect results with a data_key parameter
    results = []
    async for item in json_stream_generator(source, data_key="some_key"):
        # In the current implementation, non-dict values are returned as-is
        results.append(item)

    # In current implementation, all non-dict values are returned as-is
    # and non-None values should be collected
    # The expected behavior is that all items are yielded since they're valid JSON
    assert "string value" in results
    assert [1, 2, 3] in results
    assert True in results
    assert 123 in results

    # Only None values might be filtered out by async for loop
    if None in results:
        assert len(results) == 5
    else:
        assert len(results) == 4


@pytest.mark.asyncio
async def test_json_stream_with_data_key_extraction():
    """Test json_stream_generator data_key extraction from dict objects (Lines 93-94)."""
    # Source generator with dict JSON containing data keys
    source = MockGenerator([
        '{"data": "value1"}',          # Has data key
        '{"data": null}',              # Has data key with null value
        '{"other": "value2"}',         # Missing data key
        '{"data": {"nested": true}}',  # Nested data
        '{}'                           # Empty dict
    ])

    # Collect results using data_key extraction
    results = []
    async for item in json_stream_generator(source, data_key="data"):
        if item is not None:  # Will be None for missing data key
            results.append(item)

    # Current implementation returns:
    # 1. "value1" (extracted from {"data": "value1"})
    # 2. None (extracted from {"data": null}) - filtered by our if check
    # 3. {"nested": true} (extracted from {"data": {"nested": true}})
    # Items without the data key are filtered

    # Verify only items with the data key are included
    assert "value1" in results
    assert {"nested": True} in results

    # Assume 2 items in the results (filtering out null value)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_mixed_json_stream_with_data_key():
    """Test json_stream_generator with a mix of dict and non-dict data."""
    # Source generator with mixed dict and non-dict JSON
    source = MockGenerator([
        '"string value"',       # Non-dict (line 86)
        '{"data": "value1"}',   # Dict with key (lines 93-94)
        '[1, 2, 3]',            # Non-dict (line 86)
        '{"other": "value2"}'   # Dict without key (lines 93-94)
    ])

    # Collect results using data_key extraction
    results = []
    async for item in json_stream_generator(source, data_key="data"):
        results.append(item)

    # Verify correct handling of both branches
    assert len(results) == 3  # 2 non-dicts + 1 dict with the key
    assert results[0] == "string value"  # Non-dict returned as-is (line 86)
    assert results[1] == "value1"       # Dict value extraction (lines 93-94)
    assert results[2] == [1, 2, 3]      # Non-dict returned as-is (line 86)
    # The dict without the key is filtered out

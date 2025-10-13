"""Unit tests for the semantic cache module."""

import pytest

from onellm.cache import CacheConfig, SimpleCache


class TestCacheConfig:
    """Tests for CacheConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.max_entries == 1000
        assert config.similarity_threshold == 0.95
        assert config.hash_only is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(max_entries=500, similarity_threshold=0.9, hash_only=True)
        assert config.max_entries == 500
        assert config.similarity_threshold == 0.9
        assert config.hash_only is True


class TestSimpleCache:
    """Tests for SimpleCache class."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        config = CacheConfig()
        cache = SimpleCache(config)
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.hash_cache) == 0

    def test_hash_exact_match(self):
        """Test exact hash matching returns cached response."""
        config = CacheConfig(hash_only=True)  # Skip semantic model load
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Cache miss on first access
        result = cache.get(model, messages)
        assert result is None
        assert cache.misses == 1

        # Set cache
        cache.set(model, messages, response)
        assert len(cache.hash_cache) == 1

        # Cache hit on second access
        result = cache.get(model, messages)
        assert result == response
        assert cache.hits == 1

    def test_different_messages_no_match(self):
        """Test different messages don't match in hash cache."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Goodbye"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        cache.set(model, messages1, response)

        # Different message should not hit cache
        result = cache.get(model, messages2)
        assert result is None
        assert cache.misses == 1

    def test_different_model_no_match(self):
        """Test different models don't share cache entries."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        messages = [{"role": "user", "content": "Hello"}]
        response1 = {"choices": [{"message": {"content": "GPT-4 response"}}]}
        response2 = {"choices": [{"message": {"content": "GPT-3.5 response"}}]}

        cache.set("openai/gpt-4", messages, response1)
        cache.set("openai/gpt-3.5-turbo", messages, response2)

        # Each model should have its own cached response
        assert cache.get("openai/gpt-4", messages) == response1
        assert cache.get("openai/gpt-3.5-turbo", messages) == response2

    def test_lru_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        config = CacheConfig(max_entries=3, hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"

        # Add 3 entries
        for i in range(3):
            messages = [{"role": "user", "content": f"Message {i}"}]
            response = {"choices": [{"message": {"content": f"Response {i}"}}]}
            cache.set(model, messages, response)

        assert len(cache.hash_cache) == 3

        # Add 4th entry - should evict oldest
        messages = [{"role": "user", "content": "Message 3"}]
        response = {"choices": [{"message": {"content": "Response 3"}}]}
        cache.set(model, messages, response)

        assert len(cache.hash_cache) == 3

        # First entry should be evicted
        messages0 = [{"role": "user", "content": "Message 0"}]
        result = cache.get(model, messages0)
        assert result is None

    def test_cache_clear(self):
        """Test clearing cache."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        cache.set(model, messages, response)
        assert len(cache.hash_cache) == 1
        assert cache.hits == 0  # No hits yet

        # Get once to increment hits
        cache.get(model, messages)
        assert cache.hits == 1

        # Clear cache
        cache.clear()
        assert len(cache.hash_cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Initial stats
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entries"] == 0

        # Cache miss
        cache.get(model, messages)
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

        # Cache set
        cache.set(model, messages, response)
        stats = cache.stats()
        assert stats["entries"] == 1

        # Cache hit
        cache.get(model, messages)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_kwargs_filtering(self):
        """Test that stream, timeout, metadata are filtered from cache key."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Set with extra kwargs
        cache.set(model, messages, response, stream=False, timeout=30, metadata={"foo": "bar"})

        # Get without kwargs should still hit cache
        result = cache.get(model, messages)
        assert result == response

        # Get with different filtered kwargs should still hit cache
        result = cache.get(model, messages, stream=True, timeout=60, metadata={"baz": "qux"})
        assert result == response

    def test_kwargs_inclusion(self):
        """Test that temperature and other params affect cache key."""
        config = CacheConfig(hash_only=True)
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response1 = {"choices": [{"message": {"content": "Response temp 0.5"}}]}
        response2 = {"choices": [{"message": {"content": "Response temp 0.9"}}]}

        # Set with temperature 0.5
        cache.set(model, messages, response1, temperature=0.5)

        # Get with temperature 0.5 should hit
        result = cache.get(model, messages, temperature=0.5)
        assert result == response1

        # Get with temperature 0.9 should miss
        result = cache.get(model, messages, temperature=0.9)
        assert result is None

        # Set with temperature 0.9
        cache.set(model, messages, response2, temperature=0.9)

        # Now should have both cached
        assert cache.get(model, messages, temperature=0.5) == response1
        assert cache.get(model, messages, temperature=0.9) == response2

    def test_simulate_streaming_words(self):
        """Test simulate_streaming with words strategy."""
        from onellm.models import ChatCompletionResponse

        config = CacheConfig(
            hash_only=True, stream_chunk_strategy="words", stream_chunk_length=3
        )
        cache = SimpleCache(config)

        # Create a mock cached response
        cached_response = ChatCompletionResponse(
            id="test",
            object="chat.completion",
            model="test",
            created=0,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello world this is a test"},
                    "finish_reason": "stop",
                }
            ],
        )

        chunks = list(cache.simulate_streaming(cached_response))

        # Should have content chunks + final chunk with finish_reason
        assert len(chunks) > 1

        # Check content chunks
        content_chunks = [c for c in chunks if c.choices[0].get("delta", {}).get("content")]
        assert len(content_chunks) == 2  # "Hello world this" and "is a test"

        # Check final chunk
        final_chunk = chunks[-1]
        assert final_chunk.choices[0]["finish_reason"] == "stop"
        assert final_chunk.choices[0]["delta"] == {}

    def test_simulate_streaming_sentences(self):
        """Test simulate_streaming with sentences strategy."""
        from onellm.models import ChatCompletionResponse

        config = CacheConfig(
            hash_only=True, stream_chunk_strategy="sentences", stream_chunk_length=1
        )
        cache = SimpleCache(config)

        cached_response = ChatCompletionResponse(
            id="test",
            object="chat.completion",
            model="test",
            created=0,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How are you?"},
                    "finish_reason": "stop",
                }
            ],
        )

        chunks = list(cache.simulate_streaming(cached_response))

        # Should have 2 sentence chunks + final chunk
        content_chunks = [c for c in chunks if c.choices[0].get("delta", {}).get("content")]
        assert len(content_chunks) == 2

    def test_simulate_streaming_characters(self):
        """Test simulate_streaming with characters strategy."""
        from onellm.models import ChatCompletionResponse

        config = CacheConfig(
            hash_only=True, stream_chunk_strategy="characters", stream_chunk_length=5
        )
        cache = SimpleCache(config)

        cached_response = ChatCompletionResponse(
            id="test",
            object="chat.completion",
            model="test",
            created=0,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
        )

        chunks = list(cache.simulate_streaming(cached_response))

        # Should have 3 character chunks (Hello, worl, d) + final chunk
        content_chunks = [c for c in chunks if c.choices[0].get("delta", {}).get("content")]
        assert len(content_chunks) == 3

        # Verify content
        accumulated = "".join([c.choices[0]["delta"]["content"] for c in content_chunks])
        assert accumulated == "Hello world"

    def test_simulate_streaming_paragraphs(self):
        """Test simulate_streaming with paragraphs strategy."""
        from onellm.models import ChatCompletionResponse

        config = CacheConfig(
            hash_only=True, stream_chunk_strategy="paragraphs", stream_chunk_length=1
        )
        cache = SimpleCache(config)

        cached_response = ChatCompletionResponse(
            id="test",
            object="chat.completion",
            model="test",
            created=0,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "First paragraph.\n\nSecond paragraph.",
                    },
                    "finish_reason": "stop",
                }
            ],
        )

        chunks = list(cache.simulate_streaming(cached_response))

        # Should have 2 paragraph chunks + final chunk
        content_chunks = [c for c in chunks if c.choices[0].get("delta", {}).get("content")]
        assert len(content_chunks) == 2

    def test_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        import time

        config = CacheConfig(hash_only=True, ttl=1)  # 1 second TTL
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Cache the response
        cache.set(model, messages, response)
        assert len(cache.hash_cache) == 1

        # Should hit cache immediately
        result = cache.get(model, messages)
        assert result == response
        assert cache.hits == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should miss cache after expiration
        result = cache.get(model, messages)
        assert result is None
        assert cache.misses == 1
        assert len(cache.hash_cache) == 0  # Expired entry removed

    def test_ttl_refresh_on_hit(self):
        """Test that accessing an entry refreshes its TTL."""
        import time

        config = CacheConfig(hash_only=True, ttl=2)  # 2 seconds TTL
        cache = SimpleCache(config)

        model = "openai/gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Cache the response
        cache.set(model, messages, response)

        # Wait 1 second, then access (should refresh TTL)
        time.sleep(1)
        result = cache.get(model, messages)
        assert result == response

        # Wait another 1.5 seconds (total 2.5s from initial set, but only 1.5s from refresh)
        time.sleep(1.5)

        # Should still be cached because TTL was refreshed
        result = cache.get(model, messages)
        assert result == response
        assert cache.hits == 2

    def test_semantic_eviction_sync(self):
        """Test that semantic data stays in sync with hash cache during eviction."""
        config = CacheConfig(max_entries=3, hash_only=True)  # Small cache for testing
        cache = SimpleCache(config)

        # Add 4 entries to trigger eviction
        for i in range(4):
            model = "openai/gpt-4"
            messages = [{"role": "user", "content": f"Message {i}"}]
            response = {"choices": [{"message": {"content": f"Response {i}"}}]}
            cache.set(model, messages, response)

        # Should have exactly max_entries (3) in cache after eviction
        assert len(cache.hash_cache) == 3

        # First message should have been evicted
        messages_0 = [{"role": "user", "content": "Message 0"}]
        result = cache.get(model, messages_0)
        assert result is None  # Evicted

        # Messages 1-3 should still be cached
        for i in range(1, 4):
            messages_i = [{"role": "user", "content": f"Message {i}"}]
            result = cache.get(model, messages_i)
            assert result is not None
            assert result["choices"][0]["message"]["content"] == f"Response {i}"

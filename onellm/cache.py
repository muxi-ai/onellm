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
Semantic cache for OneLLM.

This module provides a lightweight semantic caching layer that reduces API costs
and improves response times by intelligently caching LLM responses using a hybrid
approach: instant hash-based exact matching combined with semantic similarity search.
"""

import hashlib
import json
import logging
import time
import warnings
from collections import OrderedDict

logger = logging.getLogger("onellm.cache")


class CacheConfig:
    """Configuration for cache behavior."""

    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.95,
        hash_only: bool = False,
        stream_chunk_strategy: str = "words",
        stream_chunk_length: int = 8,
        ttl: int = 86400,
    ):
        """
        Initialize cache configuration.

        Args:
            max_entries: Maximum number of cache entries before LRU eviction (default: 1000)
            similarity_threshold: Minimum similarity score for semantic cache hit (default: 0.95)
            hash_only: Disable semantic matching, use only hash-based exact matches (default: False)
            stream_chunk_strategy: How to chunk cached streaming responses (default: "words")
            stream_chunk_length: Number of strategy units per chunk (default: 8)
            ttl: Time-to-live in seconds for cache entries (default: 86400, 1 day)
        """
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.hash_only = hash_only
        self.stream_chunk_strategy = stream_chunk_strategy
        self.stream_chunk_length = stream_chunk_length
        self.ttl = ttl

        # Validate strategy
        valid_strategies = {"words", "sentences", "paragraphs", "characters"}
        if stream_chunk_strategy not in valid_strategies:
            raise ValueError(
                f"stream_chunk_strategy must be one of {valid_strategies}, "
                f"got: {stream_chunk_strategy}"
            )


class SimpleCache:
    """
    Lightweight semantic cache for LLM responses.

    Uses a hybrid two-tier approach:
    1. Hash-based exact matching (instant, ~2µs)
    2. Semantic similarity with local embeddings (fast, ~18ms, zero API cost)

    The cache is memory-only and does not persist across process restarts.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize cache with local embedding model.

        Args:
            config: Cache configuration object
        """
        self.config = config
        self.hash_cache = OrderedDict()  # For LRU eviction
        self.hits = 0
        self.misses = 0

        # Semantic search components (lazy-loaded)
        self.embedder = None
        self.semantic_index = None
        self.semantic_data = []  # Stores (hash_key, embedding) tuples
        self._semantic_responses = []  # Stores responses parallel to semantic_data
        self.np = None

        # Only initialize semantic components if not hash-only mode
        if not self.config.hash_only:
            self._init_semantic()

    def _init_semantic(self):
        """Lazy load semantic search components."""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            logger.info("Loading cache embedding model (one-time, ~13s)...")
            self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.semantic_index = faiss.IndexFlatIP(384)  # Inner product for similarity
            self.np = np  # Store numpy reference
            logger.info("✅ Cache initialized with multilingual support (50+ languages)")

        except ImportError as e:
            warnings.warn(
                f"Semantic cache disabled due to missing dependencies: {e}. "
                f"Install with: pip install sentence-transformers faiss-cpu. "
                f"Falling back to hash-only mode (exact matches only).",
                UserWarning,
                stacklevel=2,
            )
            self.config.hash_only = True

    def _create_hash_key(self, model: str, messages: list[dict], **kwargs) -> str:
        """
        Create deterministic hash key for exact matching.

        Args:
            model: Model identifier
            messages: List of message dictionaries
            **kwargs: Additional parameters (stream, timeout, metadata excluded)

        Returns:
            SHA256 hash as hex string
        """
        # Exclude parameters that don't affect response content
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["stream", "timeout", "metadata"]
        }

        payload = {"model": model, "messages": messages, "kwargs": filtered_kwargs}

        # Create deterministic JSON string and hash it
        json_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _extract_system_hash(self, messages: list[dict]) -> str:
        """
        Extract and hash system prompt content.

        This hash is used alongside semantic similarity to ensure
        different system prompts don't get mixed up even if user
        messages are similar.

        Args:
            messages: List of message dictionaries

        Returns:
            SHA256 hash of system prompt content, or empty string if none
        """
        system_texts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_texts.append(content)

        if not system_texts:
            return ""

        combined = " ".join(system_texts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]  # Short hash is enough

    def _extract_text(self, messages: list[dict]) -> str:
        """
        Extract text content from USER messages only for embedding.

        Only user messages are used for semantic comparison to avoid
        false positives when system prompts are large and identical
        across different requests.

        Args:
            messages: List of message dictionaries

        Returns:
            Concatenated text content from user messages only
        """
        texts = []
        for msg in messages:
            # Only extract from user messages to avoid system prompt dominating similarity
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                # Handle multi-modal content (extract text parts)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))

        return " ".join(texts)

    def _semantic_search(self, text: str, system_hash: str = "") -> dict | None:
        """
        Search for semantically similar cached responses.

        Args:
            text: Text to search for
            system_hash: Hash of system prompt (must match for cache hit)

        Returns:
            Cached response if similarity above threshold AND system hash matches, None otherwise
        """
        if self.embedder is None or self.semantic_index.ntotal == 0:
            return None

        # Generate embedding for query (pass as list for batch processing)
        embedding = self.embedder.encode([text], convert_to_numpy=True)

        # Extract first embedding from batch
        if len(embedding.shape) == 2:
            embedding = embedding[0]

        embedding = embedding.reshape(1, -1).astype("float32")

        # Normalize for cosine similarity (IndexFlatIP computes inner product)
        import numpy as np

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm

        # Search for multiple candidates to find one with matching system hash
        k = min(10, self.semantic_index.ntotal)  # Check top 10 candidates
        scores, indices = self.semantic_index.search(embedding, k=k)

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score < self.config.similarity_threshold:
                break  # No more candidates above threshold

            # Check if system hash matches
            stored_hash, response = self._semantic_responses[idx]
            if stored_hash == system_hash:
                logger.debug(
                    f"Semantic cache hit (similarity: {score:.3f}, "
                    f"threshold: {self.config.similarity_threshold}, "
                    f"system_hash match: True)"
                )
                return response

        return None

    def get(self, model: str, messages: list[dict], **kwargs) -> dict | None:
        """
        Retrieve cached response if available.

        Args:
            model: Model identifier
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Cached response if found, None otherwise
        """
        # Step 1: Try exact hash match (instant, ~2µs)
        hash_key = self._create_hash_key(model, messages, **kwargs)

        if hash_key in self.hash_cache:
            cache_entry = self.hash_cache[hash_key]
            response, timestamp = cache_entry["response"], cache_entry["timestamp"]

            # Check TTL
            if time.time() - timestamp > self.config.ttl:
                # Entry expired, remove it
                del self.hash_cache[hash_key]
                self.misses += 1
                logger.debug("Hash cache entry expired")
                return None

            self.hits += 1
            # Update timestamp and move to end for LRU
            cache_entry["timestamp"] = time.time()
            self.hash_cache.move_to_end(hash_key)
            logger.debug("Hash cache hit")
            return response

        # Step 2: Try semantic similarity search (~18ms)
        if not self.config.hash_only and self.embedder is not None:
            text = self._extract_text(messages)
            system_hash = self._extract_system_hash(messages)
            # Skip semantic search for very short texts - they have misleadingly high similarity
            # (e.g., "hi", "why", "ok" would all match each other at >95%)
            if text and len(text) >= 15:
                result = self._semantic_search(text, system_hash)
                if result is not None:
                    self.hits += 1
                    # Update timestamp for semantic hit
                    if result in [entry["response"] for entry in self.hash_cache.values()]:
                        for key, entry in self.hash_cache.items():
                            if entry["response"] == result:
                                entry["timestamp"] = time.time()
                                self.hash_cache.move_to_end(key)
                                break
                    return result

        # Cache miss
        self.misses += 1
        return None

    def set(self, model: str, messages: list[dict], response: dict, **kwargs):
        """
        Cache a response.

        Args:
            model: Model identifier
            messages: List of message dictionaries
            response: Response to cache
            **kwargs: Additional parameters
        """
        hash_key = self._create_hash_key(model, messages, **kwargs)

        # Add to hash cache with timestamp (for exact matches)
        self.hash_cache[hash_key] = {"response": response, "timestamp": time.time()}

        # LRU eviction if needed
        if len(self.hash_cache) > self.config.max_entries:
            # Remove oldest entry (first item in OrderedDict)
            evicted_key, _ = self.hash_cache.popitem(last=False)

            # Also remove from semantic index if present
            if not self.config.hash_only and self.semantic_data:
                # Find and remove the evicted entry from semantic data
                for i, (stored_key, _) in enumerate(self.semantic_data):
                    if stored_key == evicted_key:
                        self.semantic_data.pop(i)
                        self._semantic_responses.pop(i)
                        # Rebuild index to reflect removal
                        self._rebuild_semantic_index()
                        break

        # Add to semantic index (for similarity search)
        if not self.config.hash_only and self.embedder is not None:
            text = self._extract_text(messages)
            system_hash = self._extract_system_hash(messages)
            # Skip semantic indexing for very short texts - they cause false matches
            if text and len(text) >= 15:
                try:
                    # Generate embedding
                    import numpy as np

                    # Encode text to embedding (text must be a string)
                    if not isinstance(text, str):
                        logger.warning(f"Expected string for embedding, got {type(text)}")
                        return

                    logger.debug(f"Encoding text for cache (length: {len(text)} chars)")
                    embedding = self.embedder.encode([text], convert_to_numpy=True)
                    logger.debug(f"Generated embedding (shape: {embedding.shape}, dtype: {embedding.dtype})")

                    # Extract first embedding (encode with list returns batch)
                    if len(embedding.shape) == 2:
                        embedding = embedding[0]

                    # Ensure it's a numpy array
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)

                    # Reshape for FAISS (needs 2D)
                    embedding = embedding.reshape(1, -1).astype("float32")

                    # Normalize for cosine similarity (manual normalization to avoid faiss issues)
                    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                    embedding = embedding / norm

                    # Add to FAISS index
                    self.semantic_index.add(embedding)

                    # Store embedding and response (keep parallel lists for efficient lookup)
                    self.semantic_data.append((hash_key, embedding[0]))  # Store 1D embedding
                    self._semantic_responses.append((system_hash, response))  # Include system hash

                    # Evict from semantic index if needed
                    if len(self.semantic_data) > self.config.max_entries:
                        # Remove oldest entry and rebuild index
                        self.semantic_data.pop(0)
                        self._semantic_responses.pop(0)
                        self._rebuild_semantic_index()

                except (ValueError, RuntimeError, ImportError, AttributeError) as e:
                    # Expected errors: invalid embeddings, FAISS issues, missing dependencies
                    logger.warning(f"Failed to add to semantic cache: {e}")
                except Exception as e:
                    # Unexpected errors should be logged as errors
                    logger.error(f"Unexpected error adding to semantic cache: {e}", exc_info=True)

    def _rebuild_semantic_index(self):
        """Rebuild semantic index after eviction."""
        if self.embedder is None or not self.semantic_data:
            return

        import faiss
        import numpy as np

        # Create new index with correct dimension
        self.semantic_index = faiss.IndexFlatIP(384)

        # Re-add all remaining embeddings in order
        if self.semantic_data:
            # Stack all embeddings into a 2D array
            embeddings = np.array([emb for _, emb in self.semantic_data], dtype="float32")

            # Add all embeddings to the new index
            self.semantic_index.add(embeddings)

            logger.debug(f"Rebuilt semantic index with {len(self.semantic_data)} entries")

    def clear(self):
        """Clear all cached entries."""
        self.hash_cache.clear()
        self.hits = 0
        self.misses = 0

        # Always clear semantic data structures (even if semantic_index is None)
        self.semantic_data.clear()
        self._semantic_responses.clear()

        # Recreate semantic index if it was initialized
        if self.semantic_index is not None:
            import faiss

            self.semantic_index = faiss.IndexFlatIP(384)

        logger.info("Cache cleared")

    def stats(self) -> dict:
        """
        Return cache statistics.

        Returns:
            Dictionary with hits, misses, and entries count
        """
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.hash_cache)}

    def simulate_streaming(self, cached_response: dict):
        """
        Generate chunks from a cached response to simulate streaming.

        This allows cached responses to feel like natural streaming while
        still saving API costs and improving response time.

        Args:
            cached_response: Complete cached response

        Yields:
            ChatCompletionChunk objects
        """
        import re

        from .models import ChatCompletionChunk

        # Extract the full text from the cached response
        choice = cached_response.choices[0]
        # Handle both dict and object formats
        if isinstance(choice, dict):
            full_text = choice.get("message", {}).get("content", "")
        else:
            full_text = choice.message.get("content", "") if hasattr(choice, "message") else ""
        strategy = self.config.stream_chunk_strategy
        length = self.config.stream_chunk_length

        chunks = []

        if strategy == "words":
            # Split by whitespace, preserving spaces
            words = full_text.split()
            for i in range(0, len(words), length):
                chunk_words = words[i : i + length]
                chunk_text = " ".join(chunk_words)
                # Add space after chunk unless it's the last one
                if i + length < len(words):
                    chunk_text += " "
                chunks.append(chunk_text)

        elif strategy == "sentences":
            # Split by sentence boundaries (., !, ?, newlines)
            sentences = re.split(r"([.!?\n]+)", full_text)
            # Rejoin punctuation with sentences
            sentence_list = []
            for i in range(0, len(sentences) - 1, 2):
                sentence_list.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
            if len(sentences) % 2 == 1:  # Last item if odd number
                sentence_list.append(sentences[-1])

            # Group into chunks
            for i in range(0, len(sentence_list), length):
                chunk_sentences = sentence_list[i : i + length]
                chunk_text = "".join(chunk_sentences)
                chunks.append(chunk_text)

        elif strategy == "paragraphs":
            # Split by double newlines (paragraph breaks)
            paragraphs = full_text.split("\n\n")
            for i in range(0, len(paragraphs), length):
                chunk_paras = paragraphs[i : i + length]
                chunk_text = "\n\n".join(chunk_paras)
                # Add paragraph break unless it's the last chunk
                if i + length < len(paragraphs):
                    chunk_text += "\n\n"
                chunks.append(chunk_text)

        elif strategy == "characters":
            # Split by character count
            for i in range(0, len(full_text), length):
                chunk_text = full_text[i : i + length]
                chunks.append(chunk_text)

        # Yield chunks as ChatCompletionChunk objects
        for chunk_text in chunks:
            if chunk_text:  # Skip empty chunks
                yield ChatCompletionChunk(
                    id=cached_response.id if hasattr(cached_response, "id") else "cached",
                    object="chat.completion.chunk",
                    model=cached_response.model if hasattr(cached_response, "model") else "unknown",
                    created=cached_response.created if hasattr(cached_response, "created") else 0,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": chunk_text, "role": None},
                            "finish_reason": None,
                        }
                    ],
                )

        # Final chunk with finish_reason
        yield ChatCompletionChunk(
            id=cached_response.id if hasattr(cached_response, "id") else "cached",
            object="chat.completion.chunk",
            model=cached_response.model if hasattr(cached_response, "model") else "unknown",
            created=cached_response.created if hasattr(cached_response, "created") else 0,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        )

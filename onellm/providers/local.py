#!/usr/bin/env python3
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
Local embedding provider implementation for OneLLM.

This module implements the in-process local embedding provider, allowing
callers to run HuggingFace sentence-transformers models through the standard
``Embedding.create()`` / ``Embedding.acreate()`` interface.

Model naming format: ``local/<hf-repo-id>``

The model name after the ``local/`` prefix is passed directly to HuggingFace
as a repo id. There is no curated alias table: what you write is what gets
loaded. Examples::

    local/nomic-ai/nomic-embed-text-v1.5
    local/nomic-ai/nomic-embed-text-v2-moe
    local/sentence-transformers/all-MiniLM-L6-v2
    local/BAAI/bge-small-en-v1.5

Weights live in the standard HuggingFace cache (``$HF_HOME`` or
``~/.cache/huggingface/hub/``) and are downloaded lazily on first use.

The provider supports Matryoshka truncation via ``dimensions=<int>`` (slice +
L2-renormalize, with no validation - the caller owns whether that makes
sense for the chosen model) and task-adaptive prompting via ``task=<str>``
(prepends ``f"{task}: "`` to every input, which is the Nomic-style convention;
models that don't use prefix conditioning will simply ignore the extra tokens
at a small quality cost).

Phase 1 uses sentence-transformers (PyTorch) as the inference backend. Phase 2
will swap in an ONNX Runtime path; the caller-facing API stays unchanged.
"""

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from ..errors import InvalidRequestError, OneLLMError
from ..models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    FileObject,
    UsageInfo,
)
from ..types import Message
from .base import Provider, register_provider

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_SIZE = 2
_CACHE_SIZE_ENV = "ONELLM_LOCAL_CACHE_SIZE"
# Kill switch: set to "false" to force trust_remote_code=False for every load,
# regardless of the caller's kwarg. Models that genuinely need custom code
# (Nomic, Jina v3, etc.) will fail to load from sentence-transformers with
# its own error; well-behaved plain-transformer models will still work.
_TRUST_REMOTE_CODE_ENV = "ONELLM_ALLOW_TRUST_REMOTE_CODE"


class LocalProvider(Provider):
    """In-process local embedding provider via sentence-transformers.

    Only :meth:`create_embedding` is implemented. Chat / completion / file
    methods raise :class:`NotImplementedError` because local embedding models
    don't serve those surfaces.

    The provider maintains a tiny LRU cache of loaded models so repeated
    requests against the same repo don't pay the (~500MB, multi-second)
    ``SentenceTransformer`` load cost each time. Cache key is the HF repo id.
    """

    # Embedding-only provider - no LLM capabilities.
    json_mode_support = False
    vision_support = False
    audio_input_support = False
    video_input_support = False
    streaming_support = False
    token_by_token_support = False
    realtime_support = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LocalProvider.

        Args:
            cache_size: Maximum number of models kept in the LRU cache at
                once. Defaults to the value of ``ONELLM_LOCAL_CACHE_SIZE``
                (falling back to ``2``). A value <=0 is treated as ``1``.
            **kwargs: Forward-compatible; no other options today.
        """
        try:
            env_cache_size = int(os.environ.get(_CACHE_SIZE_ENV, _DEFAULT_CACHE_SIZE))
        except ValueError:
            logger.warning(
                "Invalid %s value %r; falling back to default %d",
                _CACHE_SIZE_ENV,
                os.environ.get(_CACHE_SIZE_ENV),
                _DEFAULT_CACHE_SIZE,
            )
            env_cache_size = _DEFAULT_CACHE_SIZE

        cache_size = kwargs.get("cache_size", env_cache_size)
        self._cache: dict[str, Any] = {}
        self._cache_order: list[str] = []
        self._cache_max: int = max(1, int(cache_size))

    # ------------------------------------------------------------------
    # Resolution + loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_trust_remote_code(**kwargs: Any) -> bool:
        """Decide the ``trust_remote_code`` flag for a load.

        Precedence:
          1. ``ONELLM_ALLOW_TRUST_REMOTE_CODE=false`` forces ``False`` (kill
             switch; no caller kwarg can override it).
          2. Caller-supplied ``trust_remote_code=...`` kwarg.
          3. Default ``True`` - we trust the caller who typed the repo id.
        """
        if os.environ.get(_TRUST_REMOTE_CODE_ENV, "true").lower() == "false":
            return False
        caller_value = kwargs.get("trust_remote_code")
        if caller_value is not None:
            return bool(caller_value)
        return True

    def _load_model(self, repo: str, trust_remote_code: bool) -> Any:
        """Load (or return cached) sentence-transformers model for ``repo``."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OneLLMError(
                "Local embeddings require the [cache] extras. "
                "Install with: pip install 'onellm[cache]'",
                provider="local",
            ) from exc

        if repo in self._cache:
            # Mark as most-recently-used
            self._cache_order.remove(repo)
            self._cache_order.append(repo)
            return self._cache[repo]

        if trust_remote_code:
            logger.warning(
                "Loading %s with trust_remote_code=True. Disable via %s=false.",
                repo,
                _TRUST_REMOTE_CODE_ENV,
            )
        logger.info("Loading local embedding model %s", repo)

        model = SentenceTransformer(repo, trust_remote_code=trust_remote_code)

        # LRU eviction if at capacity.
        if len(self._cache) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            evicted = self._cache.pop(oldest, None)
            del evicted
            logger.debug("Evicted %s from local provider cache", oldest)

        self._cache[repo] = model
        self._cache_order.append(repo)
        return model

    # ------------------------------------------------------------------
    # Input / output preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_task_prefix(texts: list[str], task: str | None) -> list[str]:
        """Prepend ``f"{task}: "`` to each input when ``task`` is given.

        No validation: we don't know which models use prefix conditioning and
        which don't, and the Nomic prefix idiom is a simple string prepend.
        Callers who want something else can format the inputs themselves and
        leave ``task`` unset.
        """
        if not task:
            return texts
        return [f"{task}: {t}" for t in texts]

    @staticmethod
    def _truncate_and_renorm(vec: list[float], dim: int) -> list[float]:
        """Matryoshka truncation: slice to ``dim`` then L2-renormalize.

        No tier validation - if the caller asks for a dimension the model
        wasn't trained to support, the result is simply a truncated vector
        that may or may not behave well in a similarity index. That's on the
        caller.
        """
        # Local import keeps numpy out of the module-import path for users
        # who only use cloud providers (numpy is pulled in by the [cache]
        # extras via sentence-transformers, so it's always present when this
        # code path actually runs).
        import numpy as np

        arr = np.array(vec[:dim], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    # ------------------------------------------------------------------
    # Provider ABC methods
    # ------------------------------------------------------------------

    async def create_embedding(
        self,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings for one or more input strings.

        Args:
            input: A single string or a list of strings to embed.
            model: HuggingFace repo id (the ``local/`` prefix has already
                been stripped by the dispatcher), e.g.
                ``"nomic-ai/nomic-embed-text-v1.5"``.
            **kwargs: Optional controls -
                * ``dimensions`` (int): if set, truncate + L2-renormalize to
                  this size. No tier validation; caller owns correctness.
                * ``task`` (str): if set, prepend ``f"{task}: "`` to every
                  input before encoding. No validation; useful for Nomic-
                  style models that were trained with that prefix idiom.
                * ``trust_remote_code`` (bool): override the default (True).
                  The ``ONELLM_ALLOW_TRUST_REMOTE_CODE=false`` kill switch
                  always wins over caller kwargs.

        Returns:
            EmbeddingResponse: the OpenAI-shaped response, with ``model``
            echoed back as ``"local/<repo>"``.
        """
        # Normalize input shape.
        inputs: list[str] = [input] if isinstance(input, str) else list(input)
        if not inputs or all(not t for t in inputs):
            raise InvalidRequestError("Input cannot be empty", provider="local")

        # Apply task prefix (if any) and load the model.
        inputs = self._apply_task_prefix(inputs, kwargs.get("task"))
        trust_flag = self._resolve_trust_remote_code(**kwargs)
        st_model = self._load_model(model, trust_flag)

        # sentence-transformers has no native async API; the encode call is
        # synchronous and CPU/GPU bound. We rely on the caller's event-loop
        # tolerance for this; a future enhancement could offload to a thread
        # pool if callers report event-loop blocking.
        raw = st_model.encode(inputs, normalize_embeddings=True)
        vectors: list[list[float]] = [list(v) for v in raw.tolist()]

        # Matryoshka truncation (pass-through; no tier validation).
        dimensions = kwargs.get("dimensions")
        if dimensions is not None and vectors and dimensions < len(vectors[0]):
            vectors = [self._truncate_and_renorm(v, int(dimensions)) for v in vectors]

        data = [
            EmbeddingData(embedding=vec, index=i, object="embedding")
            for i, vec in enumerate(vectors)
        ]

        # Token accounting is best-effort for local models - no API meter.
        # Reporting character count gives callers a rough signal; anyone
        # needing real token counts should use tiktoken externally.
        total_chars = sum(len(t) for t in inputs)
        usage = UsageInfo(
            prompt_tokens=total_chars,
            completion_tokens=0,
            total_tokens=total_chars,
        )

        return EmbeddingResponse(
            object="list",
            data=data,
            model=f"local/{model}",
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Unsupported methods - raise clearly so callers pick an LLM provider.
    # ------------------------------------------------------------------

    async def create_chat_completion(
        self,
        messages: list[Message],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        raise NotImplementedError(
            "LocalProvider does not support chat completions. "
            "Use an LLM provider (e.g. openai/, anthropic/, ollama/)."
        )

    async def create_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        raise NotImplementedError(
            "LocalProvider does not support text completions. "
            "Use an LLM provider (e.g. openai/, anthropic/, ollama/)."
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs: Any) -> FileObject:
        raise NotImplementedError("LocalProvider does not support file uploads.")

    async def download_file(self, file_id: str, **kwargs: Any) -> bytes:
        raise NotImplementedError("LocalProvider does not support file downloads.")


# Register the provider class (NOT an instance) - see providers/base.py
register_provider("local", LocalProvider)

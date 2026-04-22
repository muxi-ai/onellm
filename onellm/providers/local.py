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
callers to run HuggingFace embedding models through the standard
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

Inference backend
-----------------

The default backend is ONNX Runtime (lean: ``onnxruntime`` + ``transformers``,
~60 MB install). Backend selection happens per-repo on cache miss:

  1. Look for ONNX weights in the HF repo (``onnx/model.onnx``,
     ``model.onnx``, or ``onnx/model_quantized.onnx`` in that order).
     If present, construct an ``_OnnxBackend`` that tokenizes via
     ``transformers.AutoTokenizer``, runs ``onnxruntime.InferenceSession``,
     pools token embeddings (mean-pooling by default, overridable via the
     ``pooling`` kwarg), and L2-normalizes.
  2. If the repo has no ONNX weights, fall back to
     ``sentence-transformers`` *if it is already installed* (via the
     ``onellm[local-pytorch]`` extra) and emit a warning suggesting the
     caller either switch to an ONNX-ready repo or pin the PyTorch extra.
  3. If neither is possible, raise ``InvalidConfigurationError`` with
     concrete remediation steps.

GPU users: install ``onellm[local-gpu]`` to pick up the ``onnxruntime-gpu``
wheel. The CUDA execution provider is selected automatically when present.

The provider supports Matryoshka truncation via ``dimensions=<int>`` (slice +
L2-renormalize, with no validation - the caller owns whether that makes
sense for the chosen model), task-adaptive prompting via ``task=<str>``
(prepends ``f"{task}: "`` to every input, Nomic-style convention),
pooling override via ``pooling=<"mean"|"cls"|"max">`` (ONNX backend only;
the PyTorch fallback uses whatever pooling the source model was trained
with and emits a warning if a different strategy is requested), and
per-call sequence-length override via ``max_length=<int>``.

Sequence length
---------------

Instead of a hard-coded 512-token cap, the backend resolves each repo's
real max sequence length from three authoritative sources (ONNX shape,
``AutoConfig.max_position_embeddings``, ``tokenizer.model_max_length``)
and uses the minimum of the sane values. A deployment-level safety
ceiling (``ONELLM_LOCAL_MAX_TOKEN_LENGTH``, default ``32768``) clamps
this at load time so memory-constrained hosts stay safe. Callers can
pass ``max_length=<int>`` on any embedding call to request a shorter
window for a specific batch; passing a value larger than the resolved
cap raises ``InvalidRequestError`` unless the caller also opts in via
``allow_exceed_model_max_length=True`` (for RoPE-extrapolation models
like Nomic v1.5 whose runtime ceiling exceeds the config's
``max_position_embeddings``).

Error contract
--------------

Failures surface as :class:`onellm.errors.OneLLMError` subclasses so local/
plays cleanly in fallback chains with cloud providers. The mapping is:

=======================================  =========================================  ==========
Failure mode                             OneLLMError class                          Retriable?
=======================================  =========================================  ==========
``onnxruntime`` / ``transformers`` missing  ``InvalidConfigurationError``           no
No ONNX weights + ``[local-pytorch]`` off   ``InvalidConfigurationError``           no
Missing HF repo / invalid id             ``ResourceNotFoundError``     (404)        no
Gated / private repo, no token           ``AuthenticationError``       (401)        no
HF returns 403                           ``PermissionDeniedError``     (403)        no
HF returns 429 (rate limit on download)  ``RateLimitError``            (429)        yes
HF returns 5xx                           ``ServiceUnavailableError``   (5xx)        yes
Network / connectivity failure           ``ServiceUnavailableError``                yes
Network timeout                          ``RequestTimeoutError``                    yes
``trust_remote_code`` kill-switch block  ``PermissionDeniedError``     (403)        no
Out of memory (CPU / GPU)                ``ServiceUnavailableError``                yes
Invalid ``dimensions`` type/value        ``InvalidRequestError``       (400)        no
Invalid ``pooling`` value                ``InvalidRequestError``       (400)        no
Empty input                              ``InvalidRequestError``       (400)        no
Unsupported method (chat, completion)    ``InvalidRequestError``       (400)        no
Anything else                            ``APIError``                               no
=======================================  =========================================  ==========

"Retriable" here means the exception type is in the default
``FallbackConfig.retriable_errors`` list, so a chain like
``local/foo -> openai/bar`` reroutes on these failures without the caller
having to customize the retry set.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import Any

from huggingface_hub import errors as hf_errors

from ..errors import (
    APIError,
    AuthenticationError,
    InvalidConfigurationError,
    InvalidRequestError,
    OneLLMError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
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

# Pooling strategies the ONNX backend can apply to token-level embeddings.
_POOLING_STRATEGIES: tuple[str, ...] = ("mean", "cls", "max")
_DEFAULT_POOLING = "mean"

# ONNX weight files we look for (in order). HuggingFace repos that ship
# ONNX export artifacts commonly place them at one of these paths. The
# quantized variant is checked last so full-precision wins when both exist.
_ONNX_WEIGHT_CANDIDATES: tuple[str, ...] = (
    "onnx/model.onnx",
    "model.onnx",
    "onnx/model_quantized.onnx",
)

# Sequence-length resolution constants.
#
# We no longer blindly cap every model at 512. Instead, we consult three
# authoritative sources in priority order (session shape, model config,
# tokenizer config) and take the minimum of the sane values, falling back
# to ``_MODEL_MAX_LENGTH_FALLBACK`` only when nothing advertises a sensible
# cap. Bogus "unlimited" sentinels (e.g. ``model_max_length=10**30`` that
# some tokenizer configs ship) are rejected by the sentinel threshold.
#
# A deployment-level safety ceiling (``ONELLM_LOCAL_MAX_TOKEN_LENGTH``) lets
# memory-constrained hosts enforce a lower ceiling without patching code.
# The default ceiling (32K tokens) is high enough that every popular
# embedding model - including 8K-context ones like Nomic v1.5 and Jina v3 -
# works out of the box, while still capping an 128K pathological config.
_MODEL_MAX_LENGTH_FALLBACK = 512
_TOKENIZER_SENTINEL_THRESHOLD = 1_000_000
_DEFAULT_SAFETY_CEILING = 32_768
_SAFETY_CEILING_ENV = "ONELLM_LOCAL_MAX_TOKEN_LENGTH"


# ---------------------------------------------------------------------------
# Error normalization
# ---------------------------------------------------------------------------

# Message patterns used when the underlying library raises a raw OSError or
# ValueError without preserving the HF-specific exception class. These are
# stable enough across transformers / sentence-transformers versions that
# matching on the text is less fragile than taking a transitive-version dep.
_REPO_NOT_FOUND_PATTERNS = (
    "is not a local folder",
    "is not a valid model identifier",
)
_TRUST_REMOTE_CODE_PATTERNS = (
    "trust_remote_code",
    "requires you to execute",
)
_OOM_PATTERNS = (
    "out of memory",
    "cannot allocate memory",
    "memory exhausted",
)


def _hf_http_status(exc: hf_errors.HfHubHTTPError) -> int | None:
    """Extract the HTTP status code from an HfHubHTTPError, if present."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def _map_hf_http_error(exc: hf_errors.HfHubHTTPError, repo: str) -> OneLLMError:
    """Map an HfHubHTTPError to the appropriate onellm error class."""
    status = _hf_http_status(exc)
    msg = f"[local/{repo}] {exc}"
    if status == 401:
        return AuthenticationError(msg, provider="local", status_code=status)
    if status == 403:
        return PermissionDeniedError(msg, provider="local", status_code=status)
    if status == 404:
        return ResourceNotFoundError(msg, provider="local", status_code=status)
    if status == 408:
        return RequestTimeoutError(msg, provider="local", status_code=status)
    if status == 429:
        return RateLimitError(msg, provider="local", status_code=status)
    if status is not None and 500 <= status < 600:
        return ServiceUnavailableError(msg, provider="local", status_code=status)
    return APIError(msg, provider="local", status_code=status)


def _map_local_exception(exc: BaseException, repo: str) -> OneLLMError | None:
    """Best-effort mapping from raw backend exceptions to onellm error types.

    Returns ``None`` when the exception can't be confidently classified, so
    the caller can wrap it as a generic :class:`APIError`.
    """
    # huggingface_hub: gate + repo-not-found subclasses must be checked
    # before the HfHubHTTPError base, and gated must be checked before
    # RepositoryNotFoundError since it's a subclass.
    if isinstance(exc, hf_errors.GatedRepoError):
        return AuthenticationError(
            f"[local/{repo}] Gated HF repo; a read token is required. {exc}",
            provider="local",
            status_code=401,
        )
    if isinstance(exc, hf_errors.RepositoryNotFoundError):
        return ResourceNotFoundError(
            f"[local/{repo}] {exc}",
            provider="local",
            status_code=404,
        )
    if isinstance(exc, hf_errors.RevisionNotFoundError):
        return ResourceNotFoundError(
            f"[local/{repo}] {exc}",
            provider="local",
            status_code=404,
        )
    if isinstance(exc, hf_errors.LocalEntryNotFoundError):
        return ResourceNotFoundError(
            f"[local/{repo}] {exc}",
            provider="local",
            status_code=404,
        )
    if isinstance(exc, hf_errors.HfHubHTTPError):
        return _map_hf_http_error(exc, repo)
    if isinstance(exc, hf_errors.HFValidationError):
        return InvalidRequestError(
            f"[local/{repo}] Invalid HuggingFace repo id. {exc}",
            provider="local",
            status_code=400,
        )

    # Timeouts (socket.timeout is an alias of TimeoutError in Python 3.10+,
    # so this catches both).
    if isinstance(exc, TimeoutError):
        return RequestTimeoutError(
            f"[local/{repo}] Network timeout during model load. {exc}",
            provider="local",
        )

    # HF offline mode or any general connectivity failure.
    if isinstance(exc, hf_errors.OfflineModeIsEnabled | ConnectionError):
        return ServiceUnavailableError(
            f"[local/{repo}] Network/connectivity failure. {exc}",
            provider="local",
        )

    msg_lower = str(exc).lower()

    # trust_remote_code policy rejection - raised as ValueError by
    # transformers when the model requires custom code but trust is off.
    if isinstance(exc, ValueError) and any(p in msg_lower for p in _TRUST_REMOTE_CODE_PATTERNS):
        return PermissionDeniedError(
            f"[local/{repo}] {exc}",
            provider="local",
            status_code=403,
        )

    # Out of memory: torch wraps as RuntimeError; the CPython allocator
    # raises MemoryError. Both are transient-enough to be worth retrying
    # via a fallback (often a smaller model).
    if isinstance(exc, MemoryError | RuntimeError) and any(p in msg_lower for p in _OOM_PATTERNS):
        return ServiceUnavailableError(
            f"[local/{repo}] Out of memory during model load or inference. {exc}",
            provider="local",
        )

    # Some transformers versions still raise plain OSError for missing
    # repos without wrapping as RepositoryNotFoundError; sniff the message.
    if isinstance(exc, OSError) and any(p in msg_lower for p in _REPO_NOT_FOUND_PATTERNS):
        return ResourceNotFoundError(
            f"[local/{repo}] {exc}",
            provider="local",
            status_code=404,
        )

    return None


@contextmanager
def _normalize_errors(repo: str) -> Generator[None, None, None]:
    """Convert raw backend exceptions into onellm.errors types.

    - ``OneLLMError`` subclasses pass through unchanged (already normalized).
    - ``KeyboardInterrupt`` / ``SystemExit`` pass through unchanged (never
      swallowed).
    - Everything else is routed through :func:`_map_local_exception`; if
      nothing matches, we wrap as a generic :class:`APIError` so callers
      always see a ``OneLLMError`` out of ``local/``.
    """
    try:
        yield
    except OneLLMError:
        raise
    except Exception as exc:
        mapped = _map_local_exception(exc, repo)
        if mapped is not None:
            raise mapped from exc
        raise APIError(
            f"[local/{repo}] Unexpected error: {type(exc).__name__}: {exc}",
            provider="local",
        ) from exc


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _pool_embeddings(
    token_embeddings: Any,  # np.ndarray (batch, seq_len, hidden)
    attention_mask: Any,  # np.ndarray (batch, seq_len)
    strategy: str,
) -> Any:
    """Collapse token-level embeddings into a single vector per sequence.

    The ONNX backend returns ``last_hidden_state`` which is a
    per-token tensor; downstream callers expect one vector per input.
    ``strategy`` picks the reduction:

    * ``"mean"`` - attention-mask-weighted mean (ignores padding tokens)
    * ``"cls"`` - the first token's embedding (BERT-style [CLS] pooling)
    * ``"max"`` - element-wise max over non-padding tokens

    No strategy validation here: ``_OnnxBackend.encode`` and
    ``LocalProvider.create_embedding`` handle that (raising
    ``InvalidRequestError`` on invalid values) so every entry point uses
    the same error class.
    """
    import numpy as np

    if strategy == "cls":
        return token_embeddings[:, 0, :]

    if strategy == "max":
        # Replace padding positions with -inf so they don't win the max.
        mask = attention_mask[..., np.newaxis].astype(bool)
        masked = np.where(mask, token_embeddings, -np.inf)
        return masked.max(axis=1)

    # Default: attention-mask-weighted mean. Dividing by the number of real
    # tokens (not the padded seq_len) keeps zero-padded positions from
    # diluting the mean of short inputs in a batch.
    mask = attention_mask[..., np.newaxis].astype(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def _l2_normalize(vectors: Any) -> Any:
    """Row-wise L2 normalization with a safe zero-norm fallback."""
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# Sequence-length resolution
# ---------------------------------------------------------------------------


def _get_safety_ceiling() -> int:
    """Read the deployment-level safety ceiling from the environment.

    Set ``ONELLM_LOCAL_MAX_TOKEN_LENGTH`` to a positive integer to clamp
    every ONNX backend's max sequence length at load time. Non-integer or
    non-positive values are ignored and the default ceiling is returned,
    logged at WARNING so misconfiguration is visible.
    """
    raw = os.environ.get(_SAFETY_CEILING_ENV)
    if raw is None:
        return _DEFAULT_SAFETY_CEILING
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not a valid integer. Using default ceiling %d.",
            _SAFETY_CEILING_ENV,
            raw,
            _DEFAULT_SAFETY_CEILING,
        )
        return _DEFAULT_SAFETY_CEILING
    if val < 1:
        logger.warning(
            "Ignoring %s=%d: must be >= 1. Using default ceiling %d.",
            _SAFETY_CEILING_ENV,
            val,
            _DEFAULT_SAFETY_CEILING,
        )
        return _DEFAULT_SAFETY_CEILING
    return val


def _resolve_model_max_length(
    repo: str,
    tokenizer: Any,
    session: Any,
    *,
    trust_remote_code: bool,
) -> int:
    """Determine the real max sequence length the repo supports.

    Source priority (most authoritative first):

      1. ONNX session's declared ``input_ids`` seq-length dim. Only set
         when the exporter hard-coded the seq dimension; rare but
         authoritative (a longer input would crash the session with
         ``InvalidArgument``). Dynamic dims (``None`` / ``-1`` /
         symbolic name) are ignored.
      2. ``AutoConfig.max_position_embeddings`` from the model config.
         This is the positional embedding table size the model was
         trained for (e.g. Nomic v1.5 = 8192, XLM-R = 514, BERT-base
         = 512). Authoritative for the vast majority of repos.
      3. ``tokenizer.model_max_length`` - fallback for repos that do not
         ship a model config readable by ``AutoConfig``. Often matches
         source 2 when both exist.

    Sentinel values (anything above ``_TOKENIZER_SENTINEL_THRESHOLD``) are
    rejected as bogus so the legacy "10**30 means unlimited" convention
    does not silently allocate enormous tokenizer buffers.

    Returns the minimum of the sane candidate caps, or
    ``_MODEL_MAX_LENGTH_FALLBACK`` when nothing advertises a valid value.
    """
    caps: list[int] = []

    # 1. ONNX session: check for hard-coded seq_len on input_ids.
    try:
        for inp in session.get_inputs():
            if getattr(inp, "name", None) == "input_ids":
                shape = getattr(inp, "shape", None) or []
                if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0:
                    caps.append(int(shape[1]))
                break
    except Exception:  # noqa: BLE001 - session introspection must not explode
        pass

    # 2. Model config's max_position_embeddings.
    try:
        from transformers import AutoConfig  # type: ignore[import-not-found]

        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trust_remote_code)
        raw = getattr(cfg, "max_position_embeddings", None)
        if isinstance(raw, int) and 0 < raw <= _TOKENIZER_SENTINEL_THRESHOLD:
            caps.append(raw)
    except Exception:  # noqa: BLE001 - config load failure is non-fatal here
        pass

    # 3. Tokenizer's model_max_length.
    raw = getattr(tokenizer, "model_max_length", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        if 0 < raw <= _TOKENIZER_SENTINEL_THRESHOLD:
            caps.append(raw)

    return min(caps) if caps else _MODEL_MAX_LENGTH_FALLBACK


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------


class _OnnxBackend:
    """ONNX Runtime embedding backend.

    Installed via ``onellm[cache]`` (``onnxruntime`` + ``transformers``).
    Prefer this over the PyTorch fallback - it is an order of magnitude
    smaller to install and typically 2-5x faster on CPU.
    """

    def __init__(
        self,
        repo: str,
        tokenizer: Any,  # transformers.PreTrainedTokenizerBase
        session: Any,  # onnxruntime.InferenceSession
        model_max_length: int,
    ) -> None:
        self.repo = repo
        self.tokenizer = tokenizer
        self.session = session
        # ``model_max_length`` is the ceiling: the repo's advertised max
        # sequence length, already clamped by the deployment safety
        # ceiling at load time. Per-call ``encode(max_length=...)`` may
        # request a smaller value (useful for batching or memory
        # shaping); anything larger is rejected as a caller error.
        self.model_max_length = model_max_length
        # Cache the set of input names the model declares. Some repos
        # ship ONNX exports that only take ``input_ids`` + ``attention_mask``
        # (e.g. MiniLM); others also take ``token_type_ids`` (BERT-family).
        # Passing unexpected inputs would raise ``InvalidArgument`` from
        # onnxruntime, so we filter the tokenizer's output.
        self._expected_inputs: set[str] = {i.name for i in session.get_inputs()}

    def _build_session_inputs(self, tokens: Any) -> dict[str, Any]:
        """Align tokenizer output with the model's declared input schema.

        Three cases have to be handled:

        * Tokenizer emits a name the session doesn't declare -> drop it,
          otherwise ``onnxruntime`` raises ``InvalidArgument``.
        * Session declares a name the tokenizer didn't emit. In practice
          this happens with ``token_type_ids`` on ONNX exports of
          RoBERTa/XLM-R-family models that were emitted via the BERT
          template (e.g. sentence-transformers/paraphrase-multilingual-
          MiniLM-L12-v2). Synthesize zeros of the same shape as
          ``input_ids`` - that's the "single-sentence" value every
          token-type-aware model expects.
        * Both agree -> pass through unchanged.
        """
        import numpy as np

        session_inputs: dict[str, Any] = {}
        for name in self._expected_inputs:
            value = tokens.get(name) if hasattr(tokens, "get") else None
            if value is None and name in tokens:
                # Some tokenizer-output wrappers (BatchEncoding) don't
                # implement Mapping.get the same way a dict does.
                value = tokens[name]
            if value is not None:
                session_inputs[name] = value
                continue
            if name == "token_type_ids" and "input_ids" in tokens:
                session_inputs[name] = np.zeros(tokens["input_ids"].shape, dtype=np.int64)
                continue
            raise APIError(
                f"[local/{self.repo}] ONNX session requires input {name!r} but the "
                "tokenizer produced no matching tensor. This usually means the "
                "model's ONNX export is mismatched with its tokenizer config.",
                provider="local",
            )
        return session_inputs

    _warned_exceed: bool = False

    def encode(
        self,
        texts: list[str],
        *,
        pooling: str = _DEFAULT_POOLING,
        max_length: int | None = None,
        allow_exceed_model_max_length: bool = False,
    ) -> Any:
        if pooling not in _POOLING_STRATEGIES:
            raise InvalidRequestError(
                f"pooling must be one of {_POOLING_STRATEGIES}, got {pooling!r}",
                provider="local",
                status_code=400,
            )
        if max_length is None:
            effective_max = self.model_max_length
        else:
            if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
                raise InvalidRequestError(
                    f"max_length must be a positive integer, got {max_length!r}",
                    provider="local",
                    status_code=400,
                )
            if max_length > self.model_max_length:
                if not allow_exceed_model_max_length:
                    raise InvalidRequestError(
                        f"[local/{self.repo}] requested max_length={max_length} "
                        f"exceeds this model's advertised cap of "
                        f"{self.model_max_length} (from model config / tokenizer). "
                        "Some models (e.g. Nomic v1.5 with RoPE NTK-scaling) can "
                        "handle longer inputs via runtime extrapolation; pass "
                        "``allow_exceed_model_max_length=True`` to opt into that "
                        "at your own risk, or pick a smaller value.",
                        provider="local",
                        status_code=400,
                    )
                if not self._warned_exceed:
                    logger.warning(
                        "[local/%s] encoding at max_length=%d, above the model's "
                        "advertised cap of %d. Output quality may degrade if the "
                        "model does not support the extrapolated length. Set "
                        "allow_exceed_model_max_length=False to re-enforce the cap.",
                        self.repo,
                        max_length,
                        self.model_max_length,
                    )
                    self._warned_exceed = True
            effective_max = max_length
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=effective_max,
            return_tensors="np",
        )
        session_inputs = self._build_session_inputs(tokens)
        outputs = self.session.run(None, session_inputs)
        # Embedding models conventionally emit ``last_hidden_state`` as the
        # first output (3D: batch, seq, hidden), but some ONNX exports
        # (Optimum with pooling fused in, sentence-transformers ONNX
        # exports with an explicit pooling head) emit an already-pooled
        # 2D tensor (batch, hidden). Detect that shape and skip pooling
        # in that case - otherwise the 2D slice index in _pool_embeddings
        # would raise IndexError (cls) or produce nonsense (mean/max).
        # We honor the caller's pooling kwarg only when the model actually
        # hands us token-level embeddings.
        token_embeddings = outputs[0]
        if token_embeddings.ndim == 2:
            return _l2_normalize(token_embeddings)
        attention_mask = tokens["attention_mask"]
        pooled = _pool_embeddings(token_embeddings, attention_mask, pooling)
        return _l2_normalize(pooled)


class _PyTorchBackend:
    """Fallback backend using ``sentence-transformers``.

    Installed via ``onellm[local-pytorch]``. Only selected when a repo has
    no ONNX weights. ``SentenceTransformer`` bakes pooling into its module
    pipeline at load time, so we can't honor a different ``pooling`` kwarg
    at runtime - we emit a one-shot warning if the caller requests one.
    """

    def __init__(self, repo: str, st_model: Any) -> None:
        self.repo = repo
        self.st_model = st_model
        self._warned_pooling = False
        # ``SentenceTransformer`` exposes its tokenizer cap as
        # ``max_seq_length``. Fall back to the safe default when missing.
        raw = getattr(st_model, "max_seq_length", None)
        if (
            isinstance(raw, int)
            and not isinstance(raw, bool)
            and 0 < raw <= _TOKENIZER_SENTINEL_THRESHOLD
        ):
            self.model_max_length = raw
        else:
            self.model_max_length = _MODEL_MAX_LENGTH_FALLBACK

    def encode(
        self,
        texts: list[str],
        *,
        pooling: str = _DEFAULT_POOLING,
        max_length: int | None = None,
        allow_exceed_model_max_length: bool = False,
    ) -> Any:
        if pooling not in _POOLING_STRATEGIES:
            raise InvalidRequestError(
                f"pooling must be one of {_POOLING_STRATEGIES}, got {pooling!r}",
                provider="local",
                status_code=400,
            )
        if pooling != _DEFAULT_POOLING and not self._warned_pooling:
            logger.warning(
                "pooling=%r requested but %s uses the PyTorch/sentence-transformers "
                "fallback, which bakes pooling into the model at load time. "
                "Using the model's configured pooling. Switch to an ONNX-ready "
                "repo (e.g. one that ships onnx/model.onnx) for runtime pooling "
                "control.",
                pooling,
                self.repo,
            )
            self._warned_pooling = True
        if max_length is not None:
            if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
                raise InvalidRequestError(
                    f"max_length must be a positive integer, got {max_length!r}",
                    provider="local",
                    status_code=400,
                )
            if max_length > self.model_max_length and not allow_exceed_model_max_length:
                raise InvalidRequestError(
                    f"[local/{self.repo}] requested max_length={max_length} exceeds "
                    f"this model's advertised cap of {self.model_max_length}. "
                    "Pass allow_exceed_model_max_length=True to bypass at your risk.",
                    provider="local",
                    status_code=400,
                )
            # SentenceTransformer respects max_seq_length attr at
            # encode-time, so we swap it in and restore after. This is
            # thread-unsafe but the backend is also memoized at the
            # class level and inference is CPU/GPU-bound, so contending
            # threads are rare; acceptable for the fallback path.
            prev = self.st_model.max_seq_length
            self.st_model.max_seq_length = max_length
            try:
                return self.st_model.encode(texts, normalize_embeddings=True)
            finally:
                self.st_model.max_seq_length = prev
        return self.st_model.encode(texts, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _try_download_onnx_weights(repo: str) -> str | None:
    """Return the local path to ONNX weights for ``repo``, or ``None``.

    Walks ``_ONNX_WEIGHT_CANDIDATES`` in order and returns the first file
    that resolves via ``hf_hub_download``. Returns ``None`` only when
    every candidate 404s (i.e. the repo genuinely has no ONNX export).
    Other errors (auth, rate-limit, network) propagate so the surrounding
    ``_normalize_errors`` context translates them to the right
    ``OneLLMError`` subclass.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for candidate in _ONNX_WEIGHT_CANDIDATES:
        try:
            return hf_hub_download(repo_id=repo, filename=candidate)
        except EntryNotFoundError:
            continue
    return None


def _instantiate_onnx_backend(
    repo: str, onnx_path: str, trust_remote_code: bool = True
) -> _OnnxBackend:
    """Build an ``_OnnxBackend`` from a downloaded ``.onnx`` file.

    ``trust_remote_code`` is forwarded to ``AutoTokenizer.from_pretrained``
    so repos whose tokenizer ships custom Python code (Jina v3, some
    Nomic variants) load via the same kill-switch semantics as the
    PyTorch fallback path.
    """
    try:
        # onnxruntime and transformers are optional (shipped via the
        # [cache] extra). Silence the mypy import-not-found check since
        # we guard the import behind ImportError.
        import onnxruntime as ort  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise InvalidConfigurationError(
            "Local embeddings via ONNX Runtime require onellm[cache]. "
            "Install with: pip install 'onellm[cache]'",
            provider="local",
        ) from exc

    with _normalize_errors(repo):
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        # ``get_available_providers`` lets onnxruntime-gpu (via
        # onellm[local-gpu]) pick up the CUDA EP automatically while
        # falling back to CPUExecutionProvider when no GPU wheel is
        # installed. The default CPU wheel reports only CPU.
        session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())

    advertised = _resolve_model_max_length(
        repo, tokenizer, session, trust_remote_code=trust_remote_code
    )
    safety_ceiling = _get_safety_ceiling()
    model_max_length = min(advertised, safety_ceiling)
    if advertised > safety_ceiling:
        logger.warning(
            "[local/%s] model advertises max_position_embeddings=%d but %s=%d "
            "caps it lower; inputs will truncate at %d. Raise %s if you need "
            "the full context window.",
            repo,
            advertised,
            _SAFETY_CEILING_ENV,
            safety_ceiling,
            model_max_length,
            _SAFETY_CEILING_ENV,
        )
    return _OnnxBackend(
        repo=repo,
        tokenizer=tokenizer,
        session=session,
        model_max_length=model_max_length,
    )


def _instantiate_pytorch_backend(repo: str, trust_remote_code: bool) -> _PyTorchBackend:
    """Build a ``_PyTorchBackend`` for ``repo``.

    Only called when ``repo`` has no ONNX weights. Emits a warning so
    callers notice the slow install + slow inference path.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise InvalidConfigurationError(
            f"[local/{repo}] Repo has no ONNX weights and sentence-transformers "
            "is not installed. Either pick a model that ships ONNX weights "
            "(e.g. local/nomic-ai/nomic-embed-text-v1.5) or install the PyTorch "
            "fallback: pip install 'onellm[local-pytorch]'",
            provider="local",
        ) from exc

    logger.warning(
        "[local/%s] Repo has no ONNX weights; using sentence-transformers/PyTorch "
        "fallback. Expect slower cold start and a ~1 GB larger install footprint. "
        "Consider a repo that ships ONNX weights for lean deployments.",
        repo,
    )
    with _normalize_errors(repo):
        model = SentenceTransformer(repo, trust_remote_code=trust_remote_code)
    return _PyTorchBackend(repo=repo, st_model=model)


class LocalProvider(Provider):
    """In-process local embedding provider.

    Only :meth:`create_embedding` is implemented. Chat / completion / file
    methods raise :class:`InvalidRequestError` because local embedding
    models don't serve those surfaces.

    The provider maintains a tiny LRU cache of loaded backends so repeated
    requests against the same repo don't pay the (~500 MB, multi-second)
    model load cost each time. Cache key is the HF repo id.

    The cache is held at the **class level**, not the instance level, because
    ``providers.base.get_provider("local")`` instantiates a fresh
    ``LocalProvider`` on every ``Embedding.acreate`` call. With per-instance
    caches, every request would reload the model - defeating the whole
    point. Class-level storage keeps the LRU alive across dispatcher calls
    while still allowing unit tests to instantiate the provider directly.
    Tests that exercise the cache should call
    :meth:`_reset_cache_for_tests` in a fixture to avoid cross-test leakage.
    """

    # Embedding-only provider - no LLM capabilities.
    json_mode_support = False
    vision_support = False
    audio_input_support = False
    video_input_support = False
    streaming_support = False
    token_by_token_support = False
    realtime_support = False

    # Class-level LRU state. Shared across every LocalProvider instance so
    # the cache survives get_provider()'s "new instance per request" pattern.
    # See class docstring for rationale.
    _cache: dict[str, Any] = {}
    _cache_order: list[str] = []
    _cache_max: int = _DEFAULT_CACHE_SIZE

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LocalProvider.

        Args:
            cache_size: Maximum number of models kept in the LRU cache at
                once. Defaults to the value of ``ONELLM_LOCAL_CACHE_SIZE``
                (falling back to ``2``). A value <=0 is treated as ``1``.
                Since the cache is class-level, later instances with a
                different ``cache_size`` update the shared ceiling rather
                than getting their own.
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
        # Update the shared cache ceiling. Don't reset the dict/list - they
        # are class attributes already and their contents survive across
        # dispatcher-driven reinstantiations.
        type(self)._cache_max = max(1, int(cache_size))

    @classmethod
    def _reset_cache_for_tests(cls) -> None:
        """Clear the class-level LRU state.

        Unit tests that exercise cache behavior should call this in a
        fixture (e.g. ``autouse`` in the local-provider test module) so
        tests don't inherit state from previous tests. Production code
        must not call this; it would silently invalidate in-flight
        request caches.
        """
        cls._cache.clear()
        cls._cache_order.clear()
        cls._cache_max = _DEFAULT_CACHE_SIZE

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

    def _instantiate_model(self, repo: str, trust_remote_code: bool) -> Any:
        """Construct an embedding backend for ``repo``.

        This is the *expensive* part of loading - on a cache miss the call
        downloads model weights from HuggingFace (hundreds of MBs to a few
        GBs) and loads them into memory, which can take multiple seconds.
        Pulled out as its own method so :meth:`_load_model_async` can push
        it onto a thread-pool executor while cache bookkeeping stays on
        the event loop.

        Backend selection (first match wins):

        1. ONNX Runtime - preferred lean path. Requires ``onellm[cache]``
           (onnxruntime + transformers) and an ONNX-exported repo.
        2. ``sentence-transformers`` - fallback. Requires
           ``onellm[local-pytorch]``. Emits a warning on use.
        3. Neither available -> ``InvalidConfigurationError`` with
           remediation.

        Wraps backend construction in :func:`_normalize_errors` so HF /
        network / OOM failures surface as the appropriate
        :class:`OneLLMError` subclass.
        """
        if trust_remote_code:
            logger.warning(
                "Loading %s with trust_remote_code=True. Disable via %s=false.",
                repo,
                _TRUST_REMOTE_CODE_ENV,
            )
        logger.info("Loading local embedding model %s", repo)

        # Phase 2: try ONNX first, fall back to PyTorch only if the repo
        # has no ONNX weights *and* sentence-transformers is already
        # installed. Otherwise raise a clear InvalidConfigurationError.
        with _normalize_errors(repo):
            onnx_path = _try_download_onnx_weights(repo)
        if onnx_path is not None:
            return _instantiate_onnx_backend(repo, onnx_path, trust_remote_code)
        return _instantiate_pytorch_backend(repo, trust_remote_code)

    def _cache_lookup(self, repo: str) -> Any | None:
        """Return the cached model for ``repo`` and mark it MRU, or
        ``None`` on miss. Plain dict/list ops - fast, safe to call on the
        event loop thread.
        """
        if repo in self._cache:
            self._cache_order.remove(repo)
            self._cache_order.append(repo)
            return self._cache[repo]
        return None

    def _cache_insert(self, repo: str, model: Any) -> None:
        """Insert ``model`` under ``repo`` with LRU eviction if at
        capacity. Fast dict/list ops; safe to call on the event loop.

        Guards against a concurrent-winner race: if two coroutines both
        observe a cache miss for ``repo`` before either resolves, both
        end up calling ``_cache_insert``. Without the early-return guard
        the second call would leave a duplicate entry in
        ``_cache_order`` (``len(self._cache) < _cache_max`` still holds
        because the key is already present, so eviction is skipped but
        the key is appended again). Later, when a third distinct repo
        triggers eviction, ``pop(0)`` removes one copy and the ghost
        entry perpetually floats in ``_cache_order``, causing premature
        eviction of real entries on every subsequent insert.

        The first winner's model is kept; the second is dropped. Clients
        who cared about which specific instance they got would already
        have called ``_cache_lookup`` to identify cache hits.
        """
        if repo in self._cache:
            # Second concurrent winner - discard to avoid corrupting the
            # LRU order list. The caller still gets the model they just
            # constructed (via the return value of _instantiate_model);
            # only the cache state is protected.
            return
        if len(self._cache) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            evicted = self._cache.pop(oldest, None)
            del evicted
            logger.debug("Evicted %s from local provider cache", oldest)
        self._cache[repo] = model
        self._cache_order.append(repo)

    def _load_model(self, repo: str, trust_remote_code: bool) -> Any:
        """Synchronous load (or return cached) for ``repo``.

        Kept as the supported test/helper entry point. Async callers
        (:meth:`create_embedding`) should use :meth:`_load_model_async`
        to avoid blocking the event loop on cache miss.
        """
        cached = self._cache_lookup(repo)
        if cached is not None:
            return cached
        model = self._instantiate_model(repo, trust_remote_code)
        self._cache_insert(repo, model)
        return model

    async def _load_model_async(self, repo: str, trust_remote_code: bool) -> Any:
        """Async load (or return cached) for ``repo``.

        On cache miss, offloads the multi-second
        ``SentenceTransformer(...)`` construction to the default executor
        so the event loop stays responsive. Cache lookup and bookkeeping
        run on the event loop (dict/list ops are microseconds and are
        serialized by the single-threaded loop).
        """
        cached = self._cache_lookup(repo)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(
            None, lambda: self._instantiate_model(repo, trust_remote_code)
        )
        self._cache_insert(repo, model)
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
                * ``pooling`` (``"mean"`` | ``"cls"`` | ``"max"``): override
                  the ONNX backend's default token-embedding reduction.
                  Defaults to ``"mean"``. Ignored (with a one-shot warning)
                  on the PyTorch fallback backend.
                * ``max_length`` (int): override the max sequence length
                  the tokenizer truncates to. Defaults to the model's
                  advertised cap (resolved from ONNX shape, model config,
                  and tokenizer config, clamped by
                  ``ONELLM_LOCAL_MAX_TOKEN_LENGTH``). Passing a value
                  higher than the advertised cap raises
                  ``InvalidRequestError`` unless
                  ``allow_exceed_model_max_length=True`` is also passed.
                * ``allow_exceed_model_max_length`` (bool): opt into
                  exceeding the config-advertised cap. Needed for
                  RoPE-extrapolation models (e.g. Nomic v1.5 advertises
                  2048 but supports 8192 at inference time). Logs a
                  one-shot warning when triggered. Default False.
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
            raise InvalidRequestError("Input cannot be empty", provider="local", status_code=400)

        # Validate dimensions up front so a nonsensical value produces a
        # clear InvalidRequestError instead of an opaque numpy TypeError
        # deep inside the truncation step.
        dimensions = kwargs.get("dimensions")
        if dimensions is not None:
            if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
                raise InvalidRequestError(
                    f"dimensions must be a positive integer, got {dimensions!r}",
                    provider="local",
                    status_code=400,
                )

        # Validate pooling up front with the same pattern - the backends
        # also validate, but catching it here gives us a consistent
        # error surface regardless of which backend gets selected.
        pooling = kwargs.get("pooling", _DEFAULT_POOLING)
        if pooling not in _POOLING_STRATEGIES:
            raise InvalidRequestError(
                f"pooling must be one of {_POOLING_STRATEGIES}, got {pooling!r}",
                provider="local",
                status_code=400,
            )

        # Validate max_length up front. Actual "exceeds model cap" check
        # happens inside the backend where the advertised cap is known.
        max_length = kwargs.get("max_length")
        if max_length is not None:
            if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
                raise InvalidRequestError(
                    f"max_length must be a positive integer, got {max_length!r}",
                    provider="local",
                    status_code=400,
                )
        allow_exceed = bool(kwargs.get("allow_exceed_model_max_length", False))

        # Apply task prefix (if any) and load the backend.
        inputs = self._apply_task_prefix(inputs, kwargs.get("task"))
        trust_flag = self._resolve_trust_remote_code(**kwargs)

        # Use the async loader so the *cold-start* download + backend
        # construction runs off the event loop on cache miss. The
        # cache-hit fast path is a single dict lookup and stays on the
        # loop.
        backend = await self._load_model_async(model, trust_flag)

        # Both backends expose a synchronous ``encode`` that is CPU/GPU-
        # bound and would stall every coroutine on the event loop for
        # the duration of inference. Offload to the default executor so
        # the event loop stays responsive for long batches or slow
        # hardware.
        loop = asyncio.get_running_loop()
        with _normalize_errors(model):
            raw = await loop.run_in_executor(
                None,
                lambda: backend.encode(
                    inputs,
                    pooling=pooling,
                    max_length=max_length,
                    allow_exceed_model_max_length=allow_exceed,
                ),
            )
        vectors: list[list[float]] = [list(v) for v in raw.tolist()]

        # Matryoshka truncation (pass-through; no tier validation).
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
    # Unsupported methods - surface as InvalidRequestError so fallback chains
    # can treat them like any other 400 from a cloud provider.
    # ------------------------------------------------------------------

    async def create_chat_completion(
        self,
        messages: list[Message],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionChunk, None]:
        raise InvalidRequestError(
            "LocalProvider does not support chat completions. "
            "Use an LLM provider (e.g. openai/, anthropic/, ollama/).",
            provider="local",
            status_code=400,
        )

    async def create_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[Any, None]:
        raise InvalidRequestError(
            "LocalProvider does not support text completions. "
            "Use an LLM provider (e.g. openai/, anthropic/, ollama/).",
            provider="local",
            status_code=400,
        )

    async def upload_file(self, file: Any, purpose: str, **kwargs: Any) -> FileObject:
        raise InvalidRequestError(
            "LocalProvider does not support file uploads.",
            provider="local",
            status_code=400,
        )

    async def download_file(self, file_id: str, **kwargs: Any) -> bytes:
        raise InvalidRequestError(
            "LocalProvider does not support file downloads.",
            provider="local",
            status_code=400,
        )


# Register the provider class (NOT an instance) - see providers/base.py
register_provider("local", LocalProvider)

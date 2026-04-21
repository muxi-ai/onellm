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
Unit tests for the local embedding provider.

Tests mock one of two backends - ``onnxruntime.InferenceSession`` +
``transformers.AutoTokenizer`` for the default ONNX path, or
``sentence_transformers.SentenceTransformer`` for the PyTorch fallback -
so no model downloads or inference happen. Real model load tests live
in ``tests/integration/test_local_provider.py`` behind a slow marker.
"""

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from huggingface_hub import errors as hf_errors

from onellm.errors import (
    APIError,
    AuthenticationError,
    InvalidConfigurationError,
    InvalidRequestError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from onellm.models import EmbeddingResponse
from onellm.providers.local import LocalProvider, _normalize_errors


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a fake ``sentence_transformers`` module and force the
    PyTorch fallback path by reporting no ONNX weights.

    Each call to ``SentenceTransformer(repo, trust_remote_code=...)`` returns
    a fresh :class:`MagicMock` whose ``.encode`` method returns an 8-dim
    vector per input. Tests that need a specific shape override
    ``SentenceTransformer.side_effect`` to build their own mocks.

    This fixture is intentionally narrow: it only exercises the PyTorch
    fallback branch. The ONNX backend has its own fixture
    (:func:`fake_onnx_backend`). Tests that need to verify cross-backend
    behavior (e.g. error normalization) can use either.
    """
    # Force the ONNX discovery step to miss, so _instantiate_model falls
    # through to _instantiate_pytorch_backend. Without this, the backend
    # selector would try hf_hub_download against the live HF hub.
    monkeypatch.setattr("onellm.providers.local._try_download_onnx_weights", lambda repo: None)

    fake_module = types.ModuleType("sentence_transformers")

    def _factory(repo, trust_remote_code=False, **kwargs):
        model = MagicMock(name=f"SentenceTransformer({repo!r})")

        def default_encode(inputs, normalize_embeddings=True):
            return np.full((len(inputs), 8), 0.1, dtype=np.float32)

        model.encode.side_effect = default_encode
        model._repo = repo
        model._trust_remote_code = trust_remote_code
        return model

    fake_st_class = MagicMock(side_effect=_factory)
    fake_module.SentenceTransformer = fake_st_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return fake_st_class


@pytest.fixture
def fake_onnx_backend(monkeypatch):
    """Install fake ``onnxruntime`` + ``transformers`` modules and route
    ``_try_download_onnx_weights`` to a non-None path so the ONNX backend
    is selected.

    Returns a dict with ``tokenizer_factory``, ``session_factory``, and
    ``inputs_record`` handles so tests can customize behavior (output
    shape, declared input names, captured tokenizer calls) without
    rebuilding the fixture.
    """
    # Force the ONNX selector to return a sentinel path; the value itself
    # is opaque since the fake InferenceSession below ignores it.
    monkeypatch.setattr(
        "onellm.providers.local._try_download_onnx_weights",
        lambda repo: "/fake/onnx/model.onnx",
    )

    tokenizer_calls: list[dict[str, Any]] = []
    session_calls: list[dict[str, Any]] = []

    def _default_tokenizer(repo):
        tok = MagicMock(name=f"AutoTokenizer({repo!r})")
        tok.model_max_length = 512

        def _tokenize(texts, padding=True, truncation=True, max_length=512, return_tensors="np"):
            # Store the call for inspection in tests.
            tokenizer_calls.append(
                {
                    "texts": list(texts),
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                    "return_tensors": return_tensors,
                }
            )
            # Return padded token ids + attention mask. Vary lengths so
            # attention-mask-aware pooling is actually tested.
            lens = [max(1, len(t.split())) for t in texts]
            max_len = max(lens)
            input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
            for i, ln in enumerate(lens):
                input_ids[i, :ln] = np.arange(1, ln + 1)
                attention_mask[i, :ln] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        tok.side_effect = _tokenize
        return tok

    def _default_session(_onnx_path, providers=None):
        sess = MagicMock(name="InferenceSession")
        # Declare both input_ids and attention_mask; tests for the
        # "filter unexpected inputs" branch override this.
        input1 = MagicMock()
        input1.name = "input_ids"
        input2 = MagicMock()
        input2.name = "attention_mask"
        sess.get_inputs.return_value = [input1, input2]

        def _run(_out_names, inputs):
            session_calls.append({"inputs": {k: v.copy() for k, v in inputs.items()}})
            # Return token-level embeddings of shape (batch, seq, hidden=4).
            # Values vary along BOTH the seq and hidden axes so mean/cls/max
            # pooling each produce distinct (and distinct-after-L2) outputs.
            batch, seq = inputs["input_ids"].shape
            positions = np.arange(1, seq + 1, dtype=np.float32)[:, None]  # (seq, 1)
            dims = np.arange(4, dtype=np.float32)  # (hidden,)
            token_emb = positions + dims * 0.1  # (seq, hidden) via broadcasting
            token_emb = np.broadcast_to(token_emb, (batch, seq, 4)).copy()
            return [token_emb]

        sess.run.side_effect = _run
        return sess

    state: dict[str, Any] = {
        "tokenizer_factory": _default_tokenizer,
        "session_factory": _default_session,
        "tokenizer_calls": tokenizer_calls,
        "session_calls": session_calls,
    }

    def _tokenizer_factory_wrapper(repo, **_kwargs):
        return state["tokenizer_factory"](repo)

    def _session_factory_wrapper(onnx_path, providers=None):
        return state["session_factory"](onnx_path, providers)

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = MagicMock(side_effect=_session_factory_wrapper)
    fake_ort.get_available_providers = MagicMock(return_value=["CPUExecutionProvider"])

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(
        side_effect=_tokenizer_factory_wrapper
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    return state


# ---------------------------------------------------------------------------
# Task prefix (pass-through; no validation, no registry)
# ---------------------------------------------------------------------------


class TestTaskPrefix:
    def test_no_task_returns_inputs_unchanged(self):
        assert LocalProvider._apply_task_prefix(["hi"], None) == ["hi"]

    def test_empty_task_is_treated_as_no_task(self):
        # Truthy check means "" does not trigger prefix prepending.
        assert LocalProvider._apply_task_prefix(["hi"], "") == ["hi"]

    def test_task_prepends_with_colon_space(self):
        out = LocalProvider._apply_task_prefix(["hi", "world"], "search_document")
        assert out == ["search_document: hi", "search_document: world"]

    def test_arbitrary_task_string_accepted(self):
        # No registry means no task whitelist - any string prepends.
        out = LocalProvider._apply_task_prefix(["x"], "made-up-task")
        assert out == ["made-up-task: x"]


# ---------------------------------------------------------------------------
# Matryoshka truncation (pass-through; no tier validation)
# ---------------------------------------------------------------------------


class TestMatryoshkaTruncation:
    def test_truncate_and_renorm_preserves_unit_norm(self):
        vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = LocalProvider._truncate_and_renorm(vec, 3)
        assert len(out) == 3
        assert abs(np.linalg.norm(out) - 1.0) < 1e-6

    def test_truncate_zero_vector_does_not_error(self):
        out = LocalProvider._truncate_and_renorm([0.0, 0.0, 0.0], 2)
        assert out == [0.0, 0.0]


# ---------------------------------------------------------------------------
# trust_remote_code resolution
# ---------------------------------------------------------------------------


class TestTrustRemoteCodeResolution:
    def test_default_is_true(self, monkeypatch):
        # No env, no kwarg -> True.
        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)
        assert LocalProvider._resolve_trust_remote_code() is True

    def test_caller_kwarg_can_disable(self, monkeypatch):
        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)
        assert LocalProvider._resolve_trust_remote_code(trust_remote_code=False) is False

    def test_caller_kwarg_can_explicitly_enable(self, monkeypatch):
        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)
        assert LocalProvider._resolve_trust_remote_code(trust_remote_code=True) is True

    def test_env_kill_switch_forces_false(self, monkeypatch):
        monkeypatch.setenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", "false")
        # Kill switch wins over every caller value.
        assert LocalProvider._resolve_trust_remote_code() is False
        assert LocalProvider._resolve_trust_remote_code(trust_remote_code=True) is False

    def test_env_kill_switch_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", "FALSE")
        assert LocalProvider._resolve_trust_remote_code() is False


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_cache_evicts_oldest_when_full(self, fake_sentence_transformers):
        provider = LocalProvider(cache_size=1)
        provider._load_model("a", trust_remote_code=False)
        provider._load_model("b", trust_remote_code=False)
        assert "a" not in provider._cache
        assert "b" in provider._cache

    def test_cache_hit_promotes_model_to_mru(self, fake_sentence_transformers):
        provider = LocalProvider(cache_size=2)
        provider._load_model("a", trust_remote_code=False)
        provider._load_model("b", trust_remote_code=False)
        # Touch 'a' -> becomes MRU; loading 'c' must evict 'b', not 'a'.
        provider._load_model("a", trust_remote_code=False)
        provider._load_model("c", trust_remote_code=False)
        assert "a" in provider._cache
        assert "b" not in provider._cache
        assert "c" in provider._cache

    def test_cache_size_minimum_one(self, fake_sentence_transformers):
        provider = LocalProvider(cache_size=0)
        assert provider._cache_max == 1

    def test_cache_size_env_var(self, monkeypatch, fake_sentence_transformers):
        monkeypatch.setenv("ONELLM_LOCAL_CACHE_SIZE", "5")
        provider = LocalProvider()
        assert provider._cache_max == 5

    def test_cache_size_env_var_invalid_falls_back_to_default(
        self, monkeypatch, fake_sentence_transformers
    ):
        monkeypatch.setenv("ONELLM_LOCAL_CACHE_SIZE", "not-an-int")
        provider = LocalProvider()
        assert provider._cache_max == 2

    def test_concurrent_winner_does_not_corrupt_cache_order(self, fake_sentence_transformers):
        """Regression: two coroutines observing a cache miss for the same
        repo at the same time both race through ``_instantiate_model`` and
        end up calling ``_cache_insert``. Without the early-return guard
        the second insert would leave a duplicate entry in
        ``_cache_order``; a later eviction would then remove the key from
        ``_cache`` while leaving a ghost in ``_cache_order``, causing
        premature eviction of real entries on every subsequent insert.
        Simulating the race via back-to-back inserts is deterministic and
        exercises the exact code path.
        """
        provider = LocalProvider(cache_size=2)
        provider._cache_insert("a", object())
        provider._cache_insert("a", object())  # second concurrent winner

        assert provider._cache_order == ["a"], "duplicate appended -> race corruption"

        # Fill to capacity, then evict via a third repo; the ghost entry
        # would cause 'a' to look absent from _cache but still present in
        # _cache_order, which corrupts the next eviction.
        provider._cache_insert("b", object())
        provider._cache_insert("c", object())  # triggers eviction of 'a'

        assert "a" not in provider._cache
        assert "a" not in provider._cache_order
        assert set(provider._cache) == {"b", "c"}
        assert provider._cache_order == ["b", "c"]


# ---------------------------------------------------------------------------
# Model loading: trust_remote_code flow
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_pytorch_fallback_passes_trust_flag_through(self, fake_sentence_transformers):
        provider = LocalProvider()
        backend = provider._load_model("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        # Backend unwrap: _PyTorchBackend.st_model is the MagicMock.
        assert backend.st_model._trust_remote_code is True

    def test_pytorch_fallback_can_refuse_trust_when_caller_passes_false(
        self, fake_sentence_transformers
    ):
        provider = LocalProvider()
        backend = provider._load_model("foo/bar", trust_remote_code=False)
        assert backend.st_model._trust_remote_code is False


# ---------------------------------------------------------------------------
# Missing backends
# ---------------------------------------------------------------------------


class TestMissingExtra:
    def test_load_model_raises_when_no_onnx_and_no_sentence_transformers(self, monkeypatch):
        """Hard break: no ONNX weights + no [local-pytorch] = clear install hint
        pointing to both remediation paths.
        """
        # Force ONNX discovery to miss.
        monkeypatch.setattr("onellm.providers.local._try_download_onnx_weights", lambda repo: None)
        # Block sentence_transformers import.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        provider = LocalProvider()
        with pytest.raises(InvalidConfigurationError) as exc:
            provider._load_model("foo/bar", trust_remote_code=False)
        msg = str(exc.value)
        # Remediation must mention both the ONNX-ready path and the
        # PyTorch fallback extra.
        assert "ONNX" in msg or "onnx" in msg or "local-pytorch" in msg
        assert exc.value.provider == "local"

    def test_load_model_raises_when_onnx_extra_missing(self, monkeypatch):
        """ONNX weights found but onnxruntime/transformers not installed -
        raise with [cache] install hint rather than falling through.
        """
        monkeypatch.setattr(
            "onellm.providers.local._try_download_onnx_weights",
            lambda repo: "/fake/onnx/model.onnx",
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        monkeypatch.setitem(sys.modules, "transformers", None)
        provider = LocalProvider()
        with pytest.raises(InvalidConfigurationError) as exc:
            provider._load_model("foo/bar", trust_remote_code=False)
        assert "onellm[cache]" in str(exc.value)


# ---------------------------------------------------------------------------
# create_embedding
# ---------------------------------------------------------------------------


class TestCreateEmbedding:
    async def test_returns_embedding_response_for_full_hf_id(
        self, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)

        def encode(inputs, normalize_embeddings=True):
            vec = np.zeros(768, dtype=np.float32)
            vec[0] = 1.0
            return np.tile(vec, (len(inputs), 1))

        def _factory(repo, **kw):
            m = MagicMock()
            m.encode.side_effect = encode
            m._trust_remote_code = kw.get("trust_remote_code", False)
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        resp = await provider.create_embedding(
            input="hello",
            model="nomic-ai/nomic-embed-text-v1.5",
        )

        assert isinstance(resp, EmbeddingResponse)
        assert resp.object == "list"
        # Model slug is echoed back as local/<repo>, using the full HF id.
        assert resp.model == "local/nomic-ai/nomic-embed-text-v1.5"
        assert len(resp.data) == 1
        assert len(resp.data[0].embedding) == 768
        assert resp.usage is not None
        # UsageInfo is a TypedDict - use dict-style access.
        assert resp.usage["prompt_tokens"] == len("hello")

    async def test_list_input_returns_multiple_embeddings(self, fake_sentence_transformers):
        provider = LocalProvider()
        resp = await provider.create_embedding(
            input=["a", "bb", "ccc"],
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        assert len(resp.data) == 3
        assert [d.index for d in resp.data] == [0, 1, 2]
        assert resp.usage is not None
        assert resp.usage["prompt_tokens"] == 1 + 2 + 3

    async def test_empty_input_raises(self, fake_sentence_transformers):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError):
            await provider.create_embedding(
                input="", model="sentence-transformers/all-MiniLM-L6-v2"
            )
        with pytest.raises(InvalidRequestError):
            await provider.create_embedding(
                input=["", ""], model="sentence-transformers/all-MiniLM-L6-v2"
            )

    async def test_matryoshka_truncation_reduces_dimension(self, fake_sentence_transformers):
        def encode(inputs, normalize_embeddings=True):
            return np.full((len(inputs), 768), 0.01, dtype=np.float32)

        def _factory(repo, **kw):
            m = MagicMock()
            m.encode.side_effect = encode
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        resp = await provider.create_embedding(
            input="hello",
            model="nomic-ai/nomic-embed-text-v1.5",
            dimensions=256,
        )
        assert len(resp.data[0].embedding) == 256
        # After L2-renormalization, the vector must be unit-norm.
        norm = float(np.linalg.norm(resp.data[0].embedding))
        assert abs(norm - 1.0) < 1e-5

    async def test_dimensions_larger_than_native_is_pass_through(self, fake_sentence_transformers):
        """dimensions > native should leave the vector untouched."""

        def encode(inputs, normalize_embeddings=True):
            return np.full((len(inputs), 8), 0.1, dtype=np.float32)

        def _factory(repo, **kw):
            m = MagicMock()
            m.encode.side_effect = encode
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        resp = await provider.create_embedding(
            input="hello",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=9999,
        )
        # Native dim wins when requested > native.
        assert len(resp.data[0].embedding) == 8

    async def test_task_prefix_applied(self, fake_sentence_transformers):
        captured: list[str] = []

        def encode(inputs, normalize_embeddings=True):
            captured.extend(inputs)
            return np.full((len(inputs), 8), 0.1, dtype=np.float32)

        def _factory(repo, **kw):
            m = MagicMock()
            m.encode.side_effect = encode
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        await provider.create_embedding(
            input="hello",
            model="nomic-ai/nomic-embed-text-v1.5",
            task="search_document",
        )
        assert captured == ["search_document: hello"]

    async def test_arbitrary_task_accepted_no_validation(self, fake_sentence_transformers):
        """No registry means no supported-task whitelist; any string prepends."""
        captured: list[str] = []

        def encode(inputs, normalize_embeddings=True):
            captured.extend(inputs)
            return np.full((len(inputs), 8), 0.1, dtype=np.float32)

        def _factory(repo, **kw):
            m = MagicMock()
            m.encode.side_effect = encode
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        await provider.create_embedding(
            input="hello",
            model="BAAI/bge-small-en-v1.5",
            task="classification",
        )
        assert captured == ["classification: hello"]

    async def test_trust_remote_code_default_true(self, fake_sentence_transformers, monkeypatch):
        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)
        captured: dict[str, bool] = {}

        def _factory(repo, trust_remote_code=False, **kw):
            captured["trust_remote_code"] = trust_remote_code
            m = MagicMock()
            m.encode.side_effect = lambda inputs, normalize_embeddings=True: np.full(
                (len(inputs), 8), 0.1, dtype=np.float32
            )
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        await provider.create_embedding(input="hi", model="nomic-ai/nomic-embed-text-v1.5")
        assert captured["trust_remote_code"] is True

    async def test_trust_remote_code_env_kill_switch(self, fake_sentence_transformers, monkeypatch):
        monkeypatch.setenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", "false")
        captured: dict[str, bool] = {}

        def _factory(repo, trust_remote_code=False, **kw):
            captured["trust_remote_code"] = trust_remote_code
            m = MagicMock()
            m.encode.side_effect = lambda inputs, normalize_embeddings=True: np.full(
                (len(inputs), 8), 0.1, dtype=np.float32
            )
            return m

        fake_sentence_transformers.side_effect = _factory

        provider = LocalProvider()
        # Even with caller asking for True, the env kill switch wins.
        await provider.create_embedding(
            input="hi",
            model="nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )
        assert captured["trust_remote_code"] is False


# ---------------------------------------------------------------------------
# Unsupported methods
# ---------------------------------------------------------------------------


class TestUnsupportedMethods:
    async def test_chat_completion_raises_invalid_request(self):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.create_chat_completion(messages=[], model="foo")
        assert exc.value.provider == "local"
        assert exc.value.status_code == 400

    async def test_completion_raises_invalid_request(self):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.create_completion(prompt="foo", model="bar")
        assert exc.value.provider == "local"
        assert exc.value.status_code == 400

    async def test_upload_file_raises_invalid_request(self):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.upload_file(file=b"x", purpose="embed")
        assert exc.value.provider == "local"

    async def test_download_file_raises_invalid_request(self):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.download_file(file_id="x")
        assert exc.value.provider == "local"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    def test_local_provider_is_registered(self):
        from onellm.providers.base import get_provider

        provider = get_provider("local")
        assert isinstance(provider, LocalProvider)

    def test_list_providers_includes_local(self):
        from onellm.providers.base import list_providers

        assert "local" in list_providers()


# ---------------------------------------------------------------------------
# CLI: local/ snapshot download
# ---------------------------------------------------------------------------


class TestDownloadLocalModel:
    def test_local_prefix_stripped_before_hf_call(self, monkeypatch, tmp_path, capsys):
        calls: dict[str, object] = {}

        def fake_snapshot_download(repo_id, cache_dir):
            calls["repo_id"] = repo_id
            calls["cache_dir"] = cache_dir
            return str(tmp_path / repo_id.replace("/", "_"))

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        from onellm.cli.download_model import download_local_model

        result = download_local_model(
            "local/nomic-ai/nomic-embed-text-v1.5", output_dir=str(tmp_path)
        )
        assert calls["repo_id"] == "nomic-ai/nomic-embed-text-v1.5"
        assert calls["cache_dir"] == str(tmp_path)
        assert result.endswith("nomic-ai_nomic-embed-text-v1.5")

        out = capsys.readouterr().out
        assert 'model="local/nomic-ai/nomic-embed-text-v1.5"' in out

    def test_slug_without_local_prefix_also_works(self, monkeypatch, tmp_path):
        calls: dict[str, object] = {}

        def fake_snapshot_download(repo_id, cache_dir):
            calls["repo_id"] = repo_id
            return "/tmp/fake-path"

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        from onellm.cli.download_model import download_local_model

        download_local_model("sentence-transformers/all-MiniLM-L6-v2", output_dir=str(tmp_path))
        assert calls["repo_id"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_no_explicit_cache_dir_defers_to_hf_hub_default(self, monkeypatch):
        """Without ``--output`` we must not pass ``cache_dir`` - we let
        ``huggingface_hub`` resolve it to ``$HUGGINGFACE_HUB_CACHE``
        (which falls back to ``$HF_HOME/hub``), matching the path
        ``SentenceTransformer`` consults at load time. Passing
        ``$HF_HOME`` directly would place the snapshot at
        ``$HF_HOME/models--*`` while the runtime looks at
        ``$HF_HOME/hub/models--*`` - the "cached" model would be invisible.
        """
        captured: dict[str, Any] = {}

        def fake_snapshot_download(repo_id, cache_dir):
            captured["cache_dir"] = cache_dir
            return "/tmp/fake"

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
        monkeypatch.setenv("HF_HOME", "/custom/hf/cache")

        from onellm.cli.download_model import download_local_model

        download_local_model("local/nomic-ai/nomic-embed-text-v1.5")
        assert captured["cache_dir"] is None


# ---------------------------------------------------------------------------
# Error normalization - raw exceptions from backend libs must surface as the
# correct onellm.errors subclass so fallback chains work without custom
# retriable_errors overrides.
# ---------------------------------------------------------------------------


def _fake_response(status_code: int) -> MagicMock:
    """Build a stand-in for an httpx.Response that only exposes status_code.

    huggingface_hub 1.x makes ``response`` a required keyword-only argument
    on ``HfHubHTTPError`` (typed as ``httpx.Response``). We only read
    ``.status_code`` in the normalization path, so a MagicMock satisfies the
    contract without pulling httpx into tests.
    """
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _make_hf_http_error(
    status_code: int, msg: str = "boom", cls: type = hf_errors.HfHubHTTPError
) -> hf_errors.HfHubHTTPError:
    """Build a (subclass of) HfHubHTTPError with a fake response."""
    return cls(msg, response=_fake_response(status_code))


class TestErrorNormalizationContext:
    """Unit-level tests of the _normalize_errors context manager."""

    def test_oneellm_errors_pass_through_unchanged(self):
        original = InvalidRequestError("keep me", provider="local", status_code=400)
        with pytest.raises(InvalidRequestError) as exc:
            with _normalize_errors("foo/bar"):
                raise original
        assert exc.value is original

    def test_keyboard_interrupt_is_not_swallowed(self):
        with pytest.raises(KeyboardInterrupt):
            with _normalize_errors("foo/bar"):
                raise KeyboardInterrupt

    def test_system_exit_is_not_swallowed(self):
        with pytest.raises(SystemExit):
            with _normalize_errors("foo/bar"):
                raise SystemExit(1)

    def test_unknown_exception_becomes_apierror(self):
        class WeirdError(Exception):
            pass

        with pytest.raises(APIError) as exc:
            with _normalize_errors("foo/bar"):
                raise WeirdError("something broke")
        assert "foo/bar" in str(exc.value)
        assert exc.value.provider == "local"
        assert exc.value.__cause__ is not None

    def test_repository_not_found_becomes_resource_not_found(self):
        with pytest.raises(ResourceNotFoundError) as exc:
            with _normalize_errors("ghost/repo"):
                raise _make_hf_http_error(
                    404, "404 at hf.co/ghost/repo", cls=hf_errors.RepositoryNotFoundError
                )
        assert exc.value.status_code == 404
        assert exc.value.provider == "local"
        assert "ghost/repo" in str(exc.value)

    def test_revision_not_found_becomes_resource_not_found(self):
        with pytest.raises(ResourceNotFoundError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(
                    404, "revision missing", cls=hf_errors.RevisionNotFoundError
                )
        assert exc.value.status_code == 404

    def test_local_entry_not_found_becomes_resource_not_found(self):
        with pytest.raises(ResourceNotFoundError) as exc:
            with _normalize_errors("foo/bar"):
                raise hf_errors.LocalEntryNotFoundError("no cached entry")
        assert exc.value.status_code == 404

    def test_gated_repo_becomes_authentication_error(self):
        # GatedRepoError is a subclass of RepositoryNotFoundError; the
        # normalizer must catch the gated case first rather than demoting
        # it to a 404.
        with pytest.raises(AuthenticationError) as exc:
            with _normalize_errors("meta-llama/Llama-2-7b"):
                raise _make_hf_http_error(
                    401, "gated repo requires login", cls=hf_errors.GatedRepoError
                )
        assert exc.value.status_code == 401

    def test_hf_http_401_becomes_authentication_error(self):
        with pytest.raises(AuthenticationError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(401)
        assert exc.value.status_code == 401

    def test_hf_http_403_becomes_permission_denied(self):
        with pytest.raises(PermissionDeniedError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(403)
        assert exc.value.status_code == 403

    def test_hf_http_429_becomes_rate_limit(self):
        with pytest.raises(RateLimitError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(429)
        assert exc.value.status_code == 429

    def test_hf_http_503_becomes_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(503)
        assert exc.value.status_code == 503

    def test_hf_http_502_becomes_service_unavailable(self):
        """5xx family should all route to ServiceUnavailableError."""
        with pytest.raises(ServiceUnavailableError):
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(502)

    def test_hf_http_408_becomes_request_timeout(self):
        with pytest.raises(RequestTimeoutError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(408)
        assert exc.value.status_code == 408

    def test_hf_http_418_becomes_generic_apierror(self):
        """Unmapped HTTP status codes fall through to APIError."""
        with pytest.raises(APIError) as exc:
            with _normalize_errors("foo/bar"):
                raise _make_hf_http_error(418)
        assert exc.value.status_code == 418

    def test_hf_validation_error_becomes_invalid_request(self):
        with pytest.raises(InvalidRequestError) as exc:
            with _normalize_errors("a/b"):
                raise hf_errors.HFValidationError("Invalid repo id")
        assert exc.value.status_code == 400

    def test_timeout_becomes_request_timeout(self):
        with pytest.raises(RequestTimeoutError):
            with _normalize_errors("foo/bar"):
                raise TimeoutError("network timeout")

    def test_connection_error_becomes_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError):
            with _normalize_errors("foo/bar"):
                raise ConnectionError("DNS failure")

    def test_offline_mode_becomes_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError):
            with _normalize_errors("foo/bar"):
                raise hf_errors.OfflineModeIsEnabled("offline")

    def test_trust_remote_code_rejection_becomes_permission_denied(self):
        with pytest.raises(PermissionDeniedError) as exc:
            with _normalize_errors("nomic-ai/nomic-embed-text-v1.5"):
                raise ValueError(
                    "Loading this model requires you to execute the "
                    "configuration file in that repo. Set trust_remote_code=True."
                )
        assert exc.value.status_code == 403

    def test_oom_runtimeerror_becomes_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError):
            with _normalize_errors("foo/bar"):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    def test_oom_memoryerror_becomes_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError):
            with _normalize_errors("foo/bar"):
                raise MemoryError("Cannot allocate memory")

    def test_raw_oserror_about_invalid_id_becomes_resource_not_found(self):
        with pytest.raises(ResourceNotFoundError) as exc:
            with _normalize_errors("ghost/repo"):
                raise OSError(
                    "ghost/repo is not a local folder and is not a valid "
                    "model identifier listed on 'https://huggingface.co/models'"
                )
        assert exc.value.status_code == 404

    def test_generic_valueerror_becomes_apierror(self):
        """A ValueError that doesn't match the trust_remote_code pattern
        should fall through to the APIError catch-all - we only upgrade to
        InvalidRequestError at explicit validation edges, not reactively."""
        with pytest.raises(APIError):
            with _normalize_errors("foo/bar"):
                raise ValueError("some other computation problem")


class TestErrorNormalizationIntegration:
    """End-to-end: errors raised by the fake backend surface through
    _load_model and create_embedding with the right onellm class.

    These tests exercise the PyTorch fallback path (simpler to fake
    exhaustively than onnxruntime); the normalization context wraps
    both backends identically so the PyTorch path is representative.
    """

    @staticmethod
    def _force_pytorch_path(monkeypatch):
        """Skip ONNX discovery so errors come from the pytorch backend."""
        monkeypatch.setattr("onellm.providers.local._try_download_onnx_weights", lambda repo: None)

    def test_load_model_surfaces_repo_not_found(self, monkeypatch):
        self._force_pytorch_path(monkeypatch)
        fake_module = types.ModuleType("sentence_transformers")

        def _factory(repo, trust_remote_code=False, **kw):
            raise _make_hf_http_error(
                404, f"404 at hf.co/{repo}", cls=hf_errors.RepositoryNotFoundError
            )

        fake_module.SentenceTransformer = MagicMock(side_effect=_factory)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = LocalProvider()
        with pytest.raises(ResourceNotFoundError) as exc:
            provider._load_model("ghost/repo", trust_remote_code=True)
        assert exc.value.status_code == 404

    def test_load_model_surfaces_rate_limit(self, monkeypatch):
        self._force_pytorch_path(monkeypatch)
        fake_module = types.ModuleType("sentence_transformers")

        def _factory(repo, trust_remote_code=False, **kw):
            raise _make_hf_http_error(429, "HF rate limit")

        fake_module.SentenceTransformer = MagicMock(side_effect=_factory)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = LocalProvider()
        with pytest.raises(RateLimitError) as exc:
            provider._load_model("foo/bar", trust_remote_code=True)
        assert exc.value.status_code == 429

    def test_load_model_surfaces_network_failure(self, monkeypatch):
        self._force_pytorch_path(monkeypatch)
        fake_module = types.ModuleType("sentence_transformers")

        def _factory(repo, trust_remote_code=False, **kw):
            raise ConnectionError("could not reach hf.co")

        fake_module.SentenceTransformer = MagicMock(side_effect=_factory)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = LocalProvider()
        with pytest.raises(ServiceUnavailableError):
            provider._load_model("foo/bar", trust_remote_code=True)

    def test_encode_failure_surfaces_as_apierror(self, monkeypatch):
        self._force_pytorch_path(monkeypatch)
        fake_module = types.ModuleType("sentence_transformers")

        def _factory(repo, trust_remote_code=False, **kw):
            m = MagicMock()
            m.encode.side_effect = RuntimeError("kaboom in forward pass")
            return m

        fake_module.SentenceTransformer = MagicMock(side_effect=_factory)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        import asyncio

        provider = LocalProvider()
        with pytest.raises(APIError):
            asyncio.run(provider.create_embedding(input="hi", model="foo/bar"))

    def test_encode_oom_surfaces_as_service_unavailable(self, monkeypatch):
        self._force_pytorch_path(monkeypatch)
        fake_module = types.ModuleType("sentence_transformers")

        def _factory(repo, trust_remote_code=False, **kw):
            m = MagicMock()
            m.encode.side_effect = RuntimeError("CUDA out of memory during encode")
            return m

        fake_module.SentenceTransformer = MagicMock(side_effect=_factory)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        import asyncio

        provider = LocalProvider()
        with pytest.raises(ServiceUnavailableError):
            asyncio.run(provider.create_embedding(input="hi", model="foo/bar"))


class TestDimensionsValidation:
    async def test_non_integer_dimensions_raises_invalid_request(self, fake_sentence_transformers):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.create_embedding(input="hi", model="foo/bar", dimensions="not an int")
        assert exc.value.status_code == 400
        assert exc.value.provider == "local"

    async def test_zero_dimensions_raises_invalid_request(self, fake_sentence_transformers):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError):
            await provider.create_embedding(input="hi", model="foo/bar", dimensions=0)

    async def test_negative_dimensions_raises_invalid_request(self, fake_sentence_transformers):
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError):
            await provider.create_embedding(input="hi", model="foo/bar", dimensions=-4)

    async def test_bool_dimensions_rejected(self, fake_sentence_transformers):
        """Python quirk: ``bool`` is a subclass of ``int``, so we must
        explicitly reject ``dimensions=True`` to catch ``True`` being
        silently treated as ``1``."""
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError):
            await provider.create_embedding(input="hi", model="foo/bar", dimensions=True)


class TestFallbackRetriableMapping:
    """The normalized errors should be in FallbackConfig's default
    retriable_errors list so local/ participates in fallback without the
    caller having to override the retry set."""

    def test_default_retriable_set_includes_all_retriable_local_errors(self):
        from onellm.utils.fallback import FallbackConfig

        retriable = tuple(FallbackConfig().retriable_errors)
        # Errors local/ emits for transient failures must be in this set.
        assert issubclass(ServiceUnavailableError, retriable)
        assert issubclass(RateLimitError, retriable)
        assert issubclass(RequestTimeoutError, retriable)

    def test_default_retriable_set_excludes_non_retriable_local_errors(self):
        from onellm.utils.fallback import FallbackConfig

        retriable = tuple(FallbackConfig().retriable_errors)
        # Errors local/ emits for permanent misconfigurations must NOT be
        # in this set (would cause pointless retries).
        assert not issubclass(ResourceNotFoundError, retriable)
        assert not issubclass(AuthenticationError, retriable)
        assert not issubclass(PermissionDeniedError, retriable)
        assert not issubclass(InvalidRequestError, retriable)
        assert not issubclass(InvalidConfigurationError, retriable)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_prefers_onnx_when_available(self, fake_onnx_backend):
        from onellm.providers.local import _OnnxBackend

        provider = LocalProvider()
        backend = provider._load_model("any/repo", trust_remote_code=False)
        assert isinstance(backend, _OnnxBackend)

    def test_falls_back_to_pytorch_when_no_onnx_weights(self, fake_sentence_transformers):
        """fake_sentence_transformers forces _try_download_onnx_weights to
        return None; we should land on _PyTorchBackend."""
        from onellm.providers.local import _PyTorchBackend

        provider = LocalProvider()
        backend = provider._load_model("any/repo", trust_remote_code=False)
        assert isinstance(backend, _PyTorchBackend)

    def test_pytorch_fallback_emits_warning(self, fake_sentence_transformers, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="onellm.providers.local")
        provider = LocalProvider()
        provider._load_model("any/repo", trust_remote_code=False)
        # One of the warnings should flag the ONNX-preferred path.
        assert any(
            "no ONNX weights" in rec.message or "ONNX" in rec.message for rec in caplog.records
        )

    def test_try_download_onnx_weights_walks_candidates(self, monkeypatch):
        """The selector must try all three weight paths before giving up."""
        from huggingface_hub.errors import EntryNotFoundError

        from onellm.providers.local import _ONNX_WEIGHT_CANDIDATES, _try_download_onnx_weights

        attempted: list[str] = []

        def fake_download(repo_id, filename):
            attempted.append(filename)
            raise EntryNotFoundError(f"no {filename}")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        # Re-export EntryNotFoundError into the huggingface_hub.errors
        # namespace our callee imports from.
        fake_hf_errors = types.ModuleType("huggingface_hub.errors")
        fake_hf_errors.EntryNotFoundError = EntryNotFoundError
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_hf_errors)

        result = _try_download_onnx_weights("foo/bar")
        assert result is None
        assert attempted == list(_ONNX_WEIGHT_CANDIDATES)

    def test_try_download_onnx_weights_returns_first_hit(self, monkeypatch):
        from huggingface_hub.errors import EntryNotFoundError

        from onellm.providers.local import _try_download_onnx_weights

        def fake_download(repo_id, filename):
            if filename == "onnx/model.onnx":
                return "/cached/onnx/model.onnx"
            raise EntryNotFoundError(f"no {filename}")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        fake_hf_errors = types.ModuleType("huggingface_hub.errors")
        fake_hf_errors.EntryNotFoundError = EntryNotFoundError
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_hf_errors)

        assert _try_download_onnx_weights("foo/bar") == "/cached/onnx/model.onnx"


# ---------------------------------------------------------------------------
# ONNX backend behavior
# ---------------------------------------------------------------------------


class TestOnnxBackend:
    async def test_encode_returns_l2_normalized_batch(self, fake_onnx_backend):
        provider = LocalProvider()
        resp = await provider.create_embedding(
            input=["hello world", "short"],
            model="any/repo",
        )
        assert len(resp.data) == 2
        # Each row must be unit-norm (L2 normalization in the backend).
        for item in resp.data:
            assert abs(float(np.linalg.norm(item.embedding)) - 1.0) < 1e-5

    async def test_tokenizer_called_with_padding_and_truncation(self, fake_onnx_backend):
        provider = LocalProvider()
        await provider.create_embedding(input="hello world", model="any/repo")
        calls = fake_onnx_backend["tokenizer_calls"]
        assert calls, "tokenizer was never called"
        last = calls[-1]
        assert last["padding"] is True
        assert last["truncation"] is True
        assert last["return_tensors"] == "np"
        # max_length is capped at 512 even if the tokenizer advertises
        # something higher.
        assert last["max_length"] <= 512

    async def test_only_declared_session_inputs_are_passed(self, fake_onnx_backend):
        """If a model declares only input_ids (no attention_mask), we must
        not pass attention_mask (would raise InvalidArgument in real ort)."""

        # Reconfigure the session to declare only input_ids.
        def one_input_session(_path, providers=None):
            sess = MagicMock()
            inp = MagicMock()
            inp.name = "input_ids"
            sess.get_inputs.return_value = [inp]

            def _run(_names, inputs):
                # Assert only input_ids was passed.
                assert list(inputs.keys()) == ["input_ids"]
                batch, seq = inputs["input_ids"].shape
                return [np.ones((batch, seq, 4), dtype=np.float32)]

            sess.run.side_effect = _run
            return sess

        fake_onnx_backend["session_factory"] = one_input_session
        provider = LocalProvider()
        resp = await provider.create_embedding(input=["hi"], model="any/repo")
        assert len(resp.data) == 1

    async def test_max_length_hard_capped_at_512(self, fake_onnx_backend):
        """Some tokenizer configs advertise absurd model_max_length; we
        cap at 512 to avoid OOM."""

        # Override the tokenizer to report a huge model_max_length.
        def big_tokenizer(repo):
            tok = MagicMock()
            tok.model_max_length = int(1e30)

            def _tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np"):
                fake_onnx_backend["tokenizer_calls"].append({"max_length": max_length})
                ids = np.ones((len(texts), 2), dtype=np.int64)
                mask = np.ones((len(texts), 2), dtype=np.int64)
                return {"input_ids": ids, "attention_mask": mask}

            tok.side_effect = _tok
            return tok

        fake_onnx_backend["tokenizer_factory"] = big_tokenizer

        provider = LocalProvider()
        await provider.create_embedding(input="hi", model="any/repo")
        assert fake_onnx_backend["tokenizer_calls"][-1]["max_length"] == 512

    async def test_pre_pooled_2d_output_skips_pooling(self, fake_onnx_backend):
        """Some ONNX exports (Optimum with fused pooling head, SBERT
        ONNX exports) emit an already-pooled (batch, hidden) tensor as
        outputs[0]. The backend must detect that shape and skip
        _pool_embeddings; otherwise cls pooling would IndexError on the
        2D slice and mean/max would produce nonsense.
        """

        def pre_pooled_session(_path, providers=None):
            sess = MagicMock()
            inp = MagicMock()
            inp.name = "input_ids"
            inp2 = MagicMock()
            inp2.name = "attention_mask"
            sess.get_inputs.return_value = [inp, inp2]

            def _run(_names, inputs):
                batch = inputs["input_ids"].shape[0]
                # Already pooled: (batch, hidden=4)
                return [np.array([[1.0, 2.0, 2.0, 1.0]] * batch, dtype=np.float32)]

            sess.run.side_effect = _run
            return sess

        fake_onnx_backend["session_factory"] = pre_pooled_session
        provider = LocalProvider()
        resp = await provider.create_embedding(input=["hi"], model="any/repo", pooling="cls")
        # Should not crash; result must still be L2-normalized.
        assert abs(float(np.linalg.norm(resp.data[0].embedding)) - 1.0) < 1e-5

    async def test_trust_remote_code_forwarded_to_onnx_tokenizer(
        self, fake_onnx_backend, monkeypatch
    ):
        """Regression: the ONNX path used to drop trust_remote_code on
        the floor (AutoTokenizer.from_pretrained was called without it),
        so repos whose tokenizer ships custom Python code (Jina v3,
        Nomic variants) would blow up with no hint. The flag must now
        be threaded through to AutoTokenizer.from_pretrained.
        """
        import transformers  # type: ignore[import-not-found]

        monkeypatch.delenv("ONELLM_ALLOW_TRUST_REMOTE_CODE", raising=False)
        # Use cache_size=1 + distinct repos so each call forces a fresh
        # tokenizer load - otherwise the LRU would serve the second call
        # from cache and the trust_remote_code flag would never propagate.
        provider = LocalProvider(cache_size=1)

        # Default: trust_remote_code=True.
        await provider.create_embedding(input="hi", model="repo-a/default")
        first = transformers.AutoTokenizer.from_pretrained.call_args_list[-1]
        assert first.kwargs.get("trust_remote_code") is True

        # Explicit False from the caller must propagate.
        await provider.create_embedding(
            input="hi", model="repo-b/opts-out", trust_remote_code=False
        )
        second = transformers.AutoTokenizer.from_pretrained.call_args_list[-1]
        assert second.kwargs.get("trust_remote_code") is False


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class TestPooling:
    async def test_default_pooling_is_mean(self, fake_onnx_backend):
        """Calling without a pooling kwarg should produce the same result
        as passing pooling='mean'."""
        provider = LocalProvider()
        default = await provider.create_embedding(input=["hello world"], model="any/repo")
        explicit = await provider.create_embedding(
            input=["hello world"], model="any/repo", pooling="mean"
        )
        np.testing.assert_allclose(default.data[0].embedding, explicit.data[0].embedding, atol=1e-6)

    async def test_cls_pooling_differs_from_mean(self, fake_onnx_backend):
        """CLS pooling picks position 0; mean averages across all positions.
        The fake backend emits distinct per-position values so the outputs
        must differ."""
        provider = LocalProvider()
        mean = await provider.create_embedding(
            input=["hello world"], model="any/repo", pooling="mean"
        )
        cls = await provider.create_embedding(
            input=["hello world"], model="any/repo", pooling="cls"
        )
        # Both unit-norm, but vectors should not be identical.
        assert not np.allclose(mean.data[0].embedding, cls.data[0].embedding)

    async def test_max_pooling_differs_from_mean(self, fake_onnx_backend):
        provider = LocalProvider()
        mean = await provider.create_embedding(
            input=["hello world"], model="any/repo", pooling="mean"
        )
        mx = await provider.create_embedding(input=["hello world"], model="any/repo", pooling="max")
        assert not np.allclose(mean.data[0].embedding, mx.data[0].embedding)

    async def test_invalid_pooling_rejected_before_load(self, monkeypatch):
        """Validation happens before the backend is touched, so this must
        not require any backend fixture."""
        provider = LocalProvider()
        with pytest.raises(InvalidRequestError) as exc:
            await provider.create_embedding(input="hi", model="foo/bar", pooling="weighted")
        assert exc.value.status_code == 400
        assert exc.value.provider == "local"

    async def test_pytorch_backend_warns_on_pooling_override(
        self, fake_sentence_transformers, caplog
    ):
        import logging

        caplog.set_level(logging.WARNING, logger="onellm.providers.local")
        provider = LocalProvider()
        await provider.create_embedding(input="hi", model="pytorch-only/repo", pooling="cls")
        # Backend should log a warning about the override being ignored.
        assert any("pooling" in rec.message and "PyTorch" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Pooling helpers (unit-level)
# ---------------------------------------------------------------------------


class TestPoolingHelpers:
    def test_mean_pool_respects_attention_mask(self):
        """Padding positions (mask=0) must not dilute the mean of short
        inputs in a batch."""
        from onellm.providers.local import _pool_embeddings

        # Batch of 2 with different real lengths (2 and 3).
        tokens = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0], [999.0, 999.0]],  # last is padding
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        )
        mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
        out = _pool_embeddings(tokens, mask, "mean")
        # Row 0 averages only the first two tokens -> [1.0, 1.0]
        np.testing.assert_allclose(out[0], [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out[1], [2.0, 2.0], atol=1e-6)

    def test_cls_pool_picks_first_token(self):
        from onellm.providers.local import _pool_embeddings

        tokens = np.array(
            [
                [[7.0, 7.0], [1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        mask = np.ones((1, 3), dtype=np.int64)
        out = _pool_embeddings(tokens, mask, "cls")
        np.testing.assert_allclose(out[0], [7.0, 7.0])

    def test_max_pool_ignores_padding(self):
        """Padding positions must be masked with -inf so they don't win."""
        from onellm.providers.local import _pool_embeddings

        tokens = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [1e9, 1e9]],  # last is padding
            ],
            dtype=np.float32,
        )
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        out = _pool_embeddings(tokens, mask, "max")
        np.testing.assert_allclose(out[0], [2.0, 2.0])

    def test_l2_normalize_handles_zero_vector(self):
        from onellm.providers.local import _l2_normalize

        vectors = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        out = _l2_normalize(vectors)
        # Zero row stays zero (safe division), non-zero row becomes unit.
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [0.6, 0.8], atol=1e-6)

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

All tests mock ``sentence_transformers.SentenceTransformer`` so no model
downloads or inference happen. Real model load tests live in
``tests/integration/test_local_provider.py`` behind a slow marker.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from onellm.errors import InvalidRequestError, OneLLMError
from onellm.models import EmbeddingResponse
from onellm.providers.local import LocalProvider


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a fake ``sentence_transformers`` module in ``sys.modules``.

    Each call to ``SentenceTransformer(repo, trust_remote_code=...)`` returns
    a fresh :class:`MagicMock` whose ``.encode`` method returns an 8-dim
    vector per input. Tests that need a specific shape override
    ``SentenceTransformer.side_effect`` to build their own mocks.
    """
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


# ---------------------------------------------------------------------------
# Model loading: trust_remote_code flow
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_passes_trust_flag_through_to_sentence_transformers(
        self, fake_sentence_transformers
    ):
        provider = LocalProvider()
        model = provider._load_model("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        assert model._trust_remote_code is True

    def test_load_can_refuse_trust_when_caller_passes_false(self, fake_sentence_transformers):
        provider = LocalProvider()
        model = provider._load_model("foo/bar", trust_remote_code=False)
        assert model._trust_remote_code is False


# ---------------------------------------------------------------------------
# Missing [cache] extra
# ---------------------------------------------------------------------------


class TestMissingExtra:
    def test_load_model_raises_when_sentence_transformers_missing(self, monkeypatch):
        """If sentence-transformers can't import, give a clear install hint."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        provider = LocalProvider()
        with pytest.raises(OneLLMError) as exc:
            provider._load_model("foo/bar", trust_remote_code=False)
        assert "[cache]" in str(exc.value) or "cache" in str(exc.value)


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
    async def test_chat_completion_raises(self):
        provider = LocalProvider()
        with pytest.raises(NotImplementedError):
            await provider.create_chat_completion(messages=[], model="foo")

    async def test_completion_raises(self):
        provider = LocalProvider()
        with pytest.raises(NotImplementedError):
            await provider.create_completion(prompt="foo", model="bar")

    async def test_upload_file_raises(self):
        provider = LocalProvider()
        with pytest.raises(NotImplementedError):
            await provider.upload_file(file=b"x", purpose="embed")

    async def test_download_file_raises(self):
        provider = LocalProvider()
        with pytest.raises(NotImplementedError):
            await provider.download_file(file_id="x")


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

    def test_hf_home_env_var_respected(self, monkeypatch):
        captured: dict[str, str] = {}

        def fake_snapshot_download(repo_id, cache_dir):
            captured["cache_dir"] = cache_dir
            return "/tmp/fake"

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
        monkeypatch.setenv("HF_HOME", "/custom/hf/cache")

        from onellm.cli.download_model import download_local_model

        download_local_model("local/nomic-ai/nomic-embed-text-v1.5")
        assert captured["cache_dir"] == "/custom/hf/cache"

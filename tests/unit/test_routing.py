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
Unit tests for auto routing (onellm/routing.py).

Fully mocked: a deterministic keyword-based fake embedder replaces the ONNX
backend so no model is ever downloaded. Cosine behavior is controlled by
axis words - texts sharing an axis word score high against each other,
texts with no axis words are equidistant from every label (which exercises
the margin-based fallback to a group's default).
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import onellm
import onellm.routing as routing_mod
from onellm.chat_completion import ChatCompletion
from onellm.completion import Completion
from onellm.errors import (
    InvalidConfigurationError,
    InvalidRequestError,
    RoutingConfigurationError,
)
from onellm.models import ChatCompletionResponse
from onellm.routing import (
    Router,
    RoutingConfig,
    _apply_api_keys,
    _assemble_slice_sections,
    _compile_map,
    is_auto_model,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

# Axis words: any text containing one of these gets weight on that axis.
_AXES = ["code", "frontend", "typescript", "research", "summarize", "tl;dr"]


def _fake_vector(text: str) -> np.ndarray:
    words = text.lower().split()
    vec = np.array([float(words.count(axis)) for axis in _AXES] + [1.0])
    return vec / np.linalg.norm(vec)


class FakeTokenizer:
    """Word-level tokenizer: token ids are the words themselves."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeBackend:
    """Deterministic embedder matching the local backend interface."""

    model_max_length = 128

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.encode_calls = 0
        self.encode_delay = 0.0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        if self.encode_delay:
            time.sleep(self.encode_delay)
        return np.stack([_fake_vector(t) for t in texts])


REFERENCE_MAP = {
    "default": "openai/gpt-4",
    "code": {
        "default": "anthropic/claude-3-sonnet",
        "frontend": "google/gemini-pro",
        "typescript": ["anthropic/claude-3-sonnet", "openai/gpt-4"],
    },
    "research": "perplexity/sonar-pro",
    "summarize": {
        "models": "openai/gpt-4o-mini",
        "description": "condensing or extracting from supplied text",
        "examples": ["tl;dr this thread"],
    },
}


def make_router(monkeypatch, routing_map=None, backend=None, **config_overrides):
    """Build a Router with the fake backend and credential checks disabled."""
    backend = backend or FakeBackend()
    monkeypatch.setattr(Router, "_load_backend", lambda self: (np, backend))
    monkeypatch.setattr(routing_mod, "_validate_credentials", lambda root: None)
    config = RoutingConfig(embedding_model="local/fake/repo", **config_overrides)
    router = Router(routing_map or REFERENCE_MAP, config)
    router._test_backend = backend
    return router


def _user(text):
    return {"role": "user", "content": text}


# ---------------------------------------------------------------------------
# Map compilation
# ---------------------------------------------------------------------------


class TestMapCompilation:
    def test_valid_map_compiles(self):
        root = _compile_map(REFERENCE_MAP)
        assert sorted(root.children) == ["code", "research", "summarize"]
        assert root.default.models == ["openai/gpt-4"]
        assert root.children["code"].children["typescript"].models == [
            "anthropic/claude-3-sonnet",
            "openai/gpt-4",
        ]
        summarize = root.children["summarize"]
        assert summarize.is_leaf
        assert summarize.examples == ["tl;dr this thread"]

    def test_root_default_required(self):
        with pytest.raises(RoutingConfigurationError, match="default"):
            _compile_map({"code": "openai/gpt-4"})

    def test_nested_group_default_required(self):
        with pytest.raises(RoutingConfigurationError, match="'default'"):
            _compile_map({"default": "openai/gpt-4", "code": {"frontend": "openai/gpt-4o"}})

    def test_default_must_be_leaf(self):
        with pytest.raises(RoutingConfigurationError, match="leaf"):
            _compile_map({"default": {"nested": "openai/gpt-4", "default": "openai/gpt-4"}})

    def test_invalid_model_string(self):
        with pytest.raises(RoutingConfigurationError, match="provider prefix"):
            _compile_map({"default": "gpt-4"})

    def test_empty_list_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="empty list"):
            _compile_map({"default": []})

    def test_empty_group_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="empty group"):
            _compile_map({"default": "openai/gpt-4", "code": {}})

    def test_reserved_key_as_label(self):
        # api_keys is only legal at the top level; nested it's a reserved key
        with pytest.raises(RoutingConfigurationError, match="reserved"):
            _compile_map(
                {
                    "default": "openai/gpt-4",
                    "code": {"default": "openai/gpt-4", "api_keys": "openai/gpt-4"},
                }
            )

    def test_models_inside_group_makes_it_a_leaf_and_rejects_default(self):
        # A dict containing "models" is an annotated leaf; a stray "default"
        # inside it is an unexpected key, not a group marker
        with pytest.raises(RoutingConfigurationError, match="unexpected keys"):
            _compile_map(
                {
                    "default": "openai/gpt-4",
                    "code": {"default": "openai/gpt-4", "models": "openai/gpt-4"},
                }
            )

    def test_label_name_pattern(self):
        with pytest.raises(RoutingConfigurationError, match="labels must match"):
            _compile_map({"default": "openai/gpt-4", "Bad Label": "openai/gpt-4"})

    def test_annotated_leaf_unknown_keys(self):
        with pytest.raises(RoutingConfigurationError, match="unexpected keys"):
            _compile_map({"default": "openai/gpt-4", "x": {"models": "openai/gpt-4", "oops": 1}})

    def test_annotated_leaf_is_never_a_group(self):
        # A dict containing "models" is a leaf even if it looks group-ish
        root = _compile_map({"default": "openai/gpt-4", "x": {"models": ["openai/gpt-4"]}})
        assert root.children["x"].is_leaf

    def test_root_cannot_be_annotated_leaf(self):
        with pytest.raises(RoutingConfigurationError, match="root must be a group"):
            _compile_map({"models": "openai/gpt-4", "default": "openai/gpt-4"})

    def test_non_dict_map_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="must be a dict"):
            _compile_map("openai/gpt-4")


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


class TestApiKeys:
    def test_literal_key_applied(self):
        with patch("onellm.config.set_api_key") as set_key:
            _apply_api_keys({"openai": "sk-test"})
            set_key.assert_called_once_with("sk-test", "openai")

    def test_env_indirection_resolves(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "sk-from-env")
        with patch("onellm.config.set_api_key") as set_key:
            _apply_api_keys({"openai": "env:MY_TEST_KEY"})
            set_key.assert_called_once_with("sk-from-env", "openai")

    def test_env_indirection_missing_fails_fast(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING_KEY", raising=False)
        with pytest.raises(RoutingConfigurationError, match="MY_MISSING_KEY"):
            _apply_api_keys({"openai": "env:MY_MISSING_KEY"})

    def test_unknown_provider_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="unknown provider"):
            _apply_api_keys({"notaprovider": "sk-test"})

    def test_dict_value_updates_provider_config(self):
        with patch("onellm.config.update_provider_config") as update:
            _apply_api_keys({"vertexai": {"project_id": "proj"}})
            update.assert_called_once_with("vertexai", project_id="proj")

    def test_empty_key_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="empty string"):
            _apply_api_keys({"openai": ""})

    def test_invalid_value_type_rejected(self):
        with pytest.raises(RoutingConfigurationError, match="key string"):
            _apply_api_keys({"openai": 42})

    def test_map_level_api_keys_merged_kwarg_wins(self, monkeypatch):
        applied = {}
        monkeypatch.setattr(routing_mod, "_apply_api_keys", lambda keys: applied.update(keys))
        routing_map = dict(REFERENCE_MAP)
        routing_map["api_keys"] = {"openai": "sk-from-file", "groq": "sk-groq"}
        make_router(monkeypatch, routing_map=routing_map)
        # Router strips api_keys before compilation and applies the merge
        assert applied == {"openai": "sk-from-file", "groq": "sk-groq"}

        applied.clear()
        make_router(monkeypatch, routing_map=dict(routing_map))  # sanity: no kwarg -> file values
        # Now with kwarg override
        backend = FakeBackend()
        monkeypatch.setattr(Router, "_load_backend", lambda self: (np, backend))
        applied.clear()
        Router(
            dict(routing_map),
            RoutingConfig(embedding_model="local/fake/repo"),
            api_keys={"openai": "sk-from-kwarg"},
        )
        assert applied["openai"] == "sk-from-kwarg"
        assert applied["groq"] == "sk-groq"


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------


class TestCredentialValidation:
    def _validate(self, monkeypatch, providers_config, registry=("openai", "ollama")):
        monkeypatch.setattr("onellm.providers.list_providers", lambda: list(registry))
        monkeypatch.setattr(
            "onellm.config.get_provider_config",
            lambda p: providers_config.get(p, {}),
        )
        root = _compile_map({"default": f"{registry[0]}/some-model"})
        routing_mod._validate_credentials(root)

    def test_unknown_provider_hard_error(self, monkeypatch):
        monkeypatch.setattr("onellm.providers.list_providers", lambda: ["openai"])
        root = _compile_map({"default": "notreal/some-model"})
        with pytest.raises(RoutingConfigurationError, match="not supported"):
            routing_mod._validate_credentials(root)

    def test_missing_api_key_hard_error(self, monkeypatch):
        with pytest.raises(RoutingConfigurationError, match="no resolvable credentials"):
            self._validate(monkeypatch, {"openai": {"api_key": None}})

    def test_present_api_key_passes(self, monkeypatch):
        self._validate(monkeypatch, {"openai": {"api_key": "sk-x"}})

    def test_no_credential_provider_skipped(self, monkeypatch):
        # ollama has no api_key requirement; must not error
        self._validate(monkeypatch, {"ollama": {"api_key": None}}, registry=("ollama", "openai"))


# ---------------------------------------------------------------------------
# Slice assembly
# ---------------------------------------------------------------------------


class TestSliceAssembly:
    def test_tools_lead_the_slice(self, monkeypatch):
        router = make_router(monkeypatch)
        tools = [
            {
                "type": "function",
                "function": {"name": "run_sql", "description": "Run a query.\nMore."},
            }
        ]
        doc = router._assemble_slice([_user("hello")], tools, None)
        assert doc.startswith("tool: run_sql - Run a query.")
        assert "More." not in doc  # first line only

    def test_assistant_turns_dropped(self, monkeypatch):
        router = make_router(monkeypatch)
        messages = [
            _user("first question"),
            {"role": "assistant", "content": "assistant answer"},
            _user("second question"),
        ]
        doc = router._assemble_slice(messages, None, None)
        assert "assistant answer" not in doc
        assert "first question" in doc
        assert "second question" in doc

    def test_first_user_turn_deduped(self, monkeypatch):
        router = make_router(monkeypatch, recent_user_turns=3)
        messages = [_user("only turn")]
        doc = router._assemble_slice(messages, None, None)
        assert doc.count("only turn") == 1

    def test_first_user_turn_kept_outside_recent_window(self, monkeypatch):
        router = make_router(monkeypatch, recent_user_turns=2)
        messages = [_user(f"turn {i}") for i in range(10)]
        doc = router._assemble_slice(messages, None, None)
        assert "turn 0" in doc  # domain signal from turn one survives
        assert "turn 8" in doc and "turn 9" in doc
        assert "turn 4" not in doc

    def test_content_parts_text_extracted(self, monkeypatch):
        router = make_router(monkeypatch)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ]
        doc = router._assemble_slice(messages, None, None)
        assert "describe this" in doc
        assert "http://x/y.png" not in doc

    def test_prompt_path(self, monkeypatch):
        router = make_router(monkeypatch)
        doc = router._assemble_slice(None, None, "complete this story")
        assert "complete this story" in doc

    def test_system_prompt_head_truncated(self, monkeypatch):
        router = make_router(monkeypatch)
        long_system = " ".join(f"sys{i}" for i in range(500))
        messages = [{"role": "system", "content": long_system}, _user("hi")]
        doc = router._assemble_slice(messages, None, None)
        assert "sys0" in doc
        assert "sys199" in doc
        assert "sys200" not in doc  # head-truncated at 200 tokens

    def test_cap_trims_middle_of_recent_window(self, monkeypatch):
        router = make_router(monkeypatch, max_slice_tokens=30, recent_user_turns=5)
        messages = [_user(f"turn{i} " * 10) for i in range(6)]
        doc = router._assemble_slice(messages, None, None)
        assert "turn0" in doc  # first turn survives
        assert "turn5" in doc  # most recent survives
        assert router._count_tokens(doc) <= 30

    def test_sections_helper_empty_messages(self):
        fixed, recent = _assemble_slice_sections(None, None, None, 3)
        assert fixed == [] and recent == []


# ---------------------------------------------------------------------------
# Explicit segments
# ---------------------------------------------------------------------------


class TestExplicitSegments:
    def test_known_segment_descends_without_inference(self, monkeypatch):
        router = make_router(monkeypatch)
        calls_after_init = router._test_backend.encode_calls
        result = router.resolve("auto/code/frontend", messages=[_user("anything")])
        assert result.model == "google/gemini-pro"
        assert result.source == "explicit"
        assert result.resolved_path == "code/frontend"
        # Fully explicit: no embedding call was made
        assert router._test_backend.encode_calls == calls_after_init

    def test_unknown_segment_raises(self, monkeypatch):
        router = make_router(monkeypatch)
        with pytest.raises(InvalidRequestError, match="reserch"):
            router.resolve("auto/reserch", messages=[_user("hi")])

    def test_segment_past_leaf_raises(self, monkeypatch):
        router = make_router(monkeypatch)
        with pytest.raises(InvalidRequestError, match="goes deeper"):
            router.resolve("auto/code/frontend/react", messages=[_user("hi")])

    def test_error_lists_valid_labels(self, monkeypatch):
        router = make_router(monkeypatch)
        with pytest.raises(InvalidRequestError, match="frontend"):
            router.resolve("auto/code/backend", messages=[_user("hi")])


# ---------------------------------------------------------------------------
# Classification and resolution
# ---------------------------------------------------------------------------


class TestClassification:
    def test_clear_signal_routes_to_label(self, monkeypatch):
        router = make_router(monkeypatch)
        result = router.resolve("auto/code", messages=[_user("fix the frontend layout")])
        assert result.model == "google/gemini-pro"
        assert result.source == "classified"
        assert result.resolved_path == "code/frontend"
        assert result.scores["frontend"] > result.scores["typescript"]

    def test_example_exemplar_wins_via_max_pool(self, monkeypatch):
        # The query matches an example text, not the label name
        router = make_router(monkeypatch)
        result = router.resolve("auto", messages=[_user("tl;dr this thread for me")])
        assert result.model == "openai/gpt-4o-mini"
        assert result.resolved_path == "summarize"

    def test_ambiguous_falls_back_to_group_default(self, monkeypatch):
        router = make_router(monkeypatch)
        result = router.resolve("auto/code", messages=[_user("hello there friend")])
        assert result.model == "anthropic/claude-3-sonnet"
        assert result.source == "default"
        assert router.stats()["fell_back_to_default"] == 1

    def test_margin_boundary_stops(self, monkeypatch):
        # frontend and typescript both score identically -> margin 0 -> stop
        router = make_router(monkeypatch)
        result = router.resolve("auto/code", messages=[_user("frontend typescript together")])
        assert result.model == "anthropic/claude-3-sonnet"  # code default
        assert result.source == "default"

    def test_min_score_floor(self, monkeypatch):
        router = make_router(monkeypatch, min_score=0.99)
        # "research" dilutes the vector: the frontend-vs-typescript margin
        # stays large, but the absolute top score drops below the floor
        result = router.resolve("auto/code", messages=[_user("frontend research task")])
        assert result.model == "anthropic/claude-3-sonnet"
        assert result.scores["frontend"] > result.scores["typescript"] + 0.05

    def test_fallback_chain_from_list_leaf(self, monkeypatch):
        router = make_router(monkeypatch)
        result = router.resolve("auto/code", messages=[_user("typescript typescript typescript")])
        assert result.model == "anthropic/claude-3-sonnet"
        assert result.fallback_chain == ["openai/gpt-4"]
        assert result.resolved_path == "code/typescript"

    def test_single_child_descends_unconditionally(self, monkeypatch):
        single = {
            "default": "openai/gpt-4",
            "code": {"default": "anthropic/claude-3-sonnet", "frontend": "google/gemini-pro"},
        }
        router = make_router(monkeypatch, routing_map=single)
        # Entry "auto" has one child ("code") -> descend without scoring;
        # inside, "frontend" is single -> descend again
        result = router.resolve("auto", messages=[_user("no axis words at all")])
        assert result.model == "google/gemini-pro"
        assert result.source == "classified"

    def test_max_depth_cutoff(self, monkeypatch):
        deep = {
            "default": "openai/root-default",
            "a": {
                "default": "openai/a-default",
                "b": {
                    "default": "openai/b-default",
                    "c": "openai/c-model",
                },
            },
        }
        router = make_router(monkeypatch, routing_map=deep, max_depth=1)
        result = router.resolve("auto", messages=[_user("whatever")])
        # Single-child chains descend, but only max_depth levels
        assert result.model == "openai/a-default"

    def test_group_with_only_default_no_embedding(self, monkeypatch):
        router = make_router(monkeypatch, routing_map={"default": "openai/gpt-4"})
        calls_after_init = router._test_backend.encode_calls
        result = router.resolve("auto", messages=[_user("anything")])
        assert result.model == "openai/gpt-4"
        assert result.source == "default"
        assert router._test_backend.encode_calls == calls_after_init

    def test_empty_slice_resolves_to_default(self, monkeypatch):
        router = make_router(monkeypatch)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "http://x"}}],
            }
        ]
        result = router.resolve("auto", messages=messages)
        assert result.model == "openai/gpt-4"
        assert result.source == "default"

    def test_route_result_dict_shape(self, monkeypatch):
        router = make_router(monkeypatch)
        info = router.resolve("auto/code", messages=[_user("fix the frontend")]).to_dict()
        assert set(info) == {
            "entry",
            "resolved_path",
            "model",
            "fallback_chain",
            "source",
            "scores",
            "margin",
            "slice_tokens",
            "latency_ms",
        }
        assert info["entry"] == "auto/code"
        assert info["slice_tokens"] > 0


# ---------------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------------


class TestMemo:
    def test_identical_slice_hits(self, monkeypatch):
        router = make_router(monkeypatch)
        messages = [_user("fix the frontend layout")]
        first = router.resolve("auto/code", messages=messages)
        calls = router._test_backend.encode_calls
        second = router.resolve("auto/code", messages=messages)
        assert second.source == "memo"
        assert second.model == first.model
        assert router._test_backend.encode_calls == calls  # no embedding on hit
        assert router.stats()["memo_hits"] == 1

    def test_memo_keyed_on_entry_path(self, monkeypatch):
        router = make_router(monkeypatch)
        messages = [_user("fix the frontend layout")]
        router.resolve("auto/code", messages=messages)
        result = router.resolve("auto", messages=messages)
        assert result.source != "memo"  # same slice, different entry point

    def test_changed_slice_misses(self, monkeypatch):
        router = make_router(monkeypatch)
        router.resolve("auto/code", messages=[_user("fix the frontend layout")])
        result = router.resolve("auto/code", messages=[_user("fix the frontend colors")])
        assert result.source != "memo"

    def test_lru_eviction(self, monkeypatch):
        router = make_router(monkeypatch, memo_size=1)
        m1 = [_user("fix the frontend layout")]
        m2 = [_user("research research research")]
        router.resolve("auto", messages=m1)
        router.resolve("auto", messages=m2)  # evicts m1
        assert router.resolve("auto", messages=m1).source != "memo"

    def test_thread_safety_under_concurrent_resolves(self, monkeypatch):
        router = make_router(monkeypatch)
        errors = []

        def worker(i):
            try:
                for _ in range(20):
                    router.resolve("auto/code", messages=[_user(f"frontend task {i}")])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        stats = router.stats()
        total = stats["classified"] + stats["memo_hits"]
        assert total == 8 * 20


# ---------------------------------------------------------------------------
# Sibling similarity (advisory)
# ---------------------------------------------------------------------------


class TestSiblingSimilarity:
    def test_indistinguishable_siblings_warn_but_boot(self, monkeypatch, caplog):
        confusable = {
            "default": "openai/gpt-4",
            "frontend": "openai/gpt-4o",
            # identical exemplar direction as "frontend"
            "frontend-x": {"models": "google/gemini-pro", "description": "frontend"},
        }
        with caplog.at_level("WARNING", logger="onellm.routing"):
            router = make_router(monkeypatch, routing_map=confusable)
        assert router is not None  # never fatal
        assert any("indistinguishable" in r.message for r in caplog.records)

    def test_validate_similarity_false_silences(self, monkeypatch, caplog):
        confusable = {
            "default": "openai/gpt-4",
            "frontend": "openai/gpt-4o",
            "frontend-x": {"models": "google/gemini-pro", "description": "frontend"},
        }
        with caplog.at_level("WARNING", logger="onellm.routing"):
            make_router(monkeypatch, routing_map=confusable, validate_similarity=False)
        assert not any("indistinguishable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsync:
    async def test_aresolve_matches_resolve(self, monkeypatch):
        router = make_router(monkeypatch)
        result = await router.aresolve("auto/code", messages=[_user("fix the frontend")])
        assert result.model == "google/gemini-pro"

    async def test_event_loop_not_blocked(self, monkeypatch):
        backend = FakeBackend()
        backend.encode_delay = 0.2  # slow inference
        router = make_router(monkeypatch, backend=backend)

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await router.aresolve("auto/code", messages=[_user("fix the frontend")])
        task.cancel()
        # If inference blocked the loop, the ticker would barely have run
        assert ticks >= 5


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicApi:
    @pytest.fixture(autouse=True)
    def _clean_routing(self):
        yield
        onellm.disable_routing()

    def test_init_routing_defaults_to_cache_model(self, monkeypatch):
        captured = {}

        class FakeRouter:
            def __init__(self, routing_map, config, api_keys=None):
                captured["config"] = config

        monkeypatch.setattr("onellm.routing.Router", FakeRouter)
        onellm.init_routing({"default": "openai/gpt-4"})
        from onellm.cache import _CACHE_MODEL_REPO

        assert captured["config"].embedding_model == f"local/{_CACHE_MODEL_REPO}"

    def test_disable_routing_clears_global(self, monkeypatch):
        monkeypatch.setattr("onellm.routing.Router", MagicMock())
        onellm.init_routing({"default": "openai/gpt-4"})
        assert onellm._routing is not None
        onellm.disable_routing()
        assert onellm._routing is None

    def test_routing_stats_without_init(self):
        assert onellm.routing_stats() == {
            "classified": 0,
            "memo_hits": 0,
            "explicit": 0,
            "fell_back_to_default": 0,
            "avg_latency_ms": 0.0,
        }

    def test_explain_route_requires_init(self):
        with pytest.raises(InvalidConfigurationError, match="init_routing"):
            onellm.explain_route(messages=[_user("hi")])

    def test_explain_route_returns_dict(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        info = onellm.explain_route(messages=[_user("fix the frontend")], entry="auto/code")
        assert info["model"] == "google/gemini-pro"
        assert info["source"] == "classified"

    def test_load_routing_map_json(self, tmp_path):
        path = tmp_path / "map.json"
        path.write_text('{"default": "openai/gpt-4"}')
        assert onellm.load_routing_map(str(path)) == {"default": "openai/gpt-4"}

    def test_load_routing_map_yaml(self, tmp_path):
        path = tmp_path / "map.yaml"
        path.write_text("default: openai/gpt-4\ncode:\n  default: anthropic/claude-3\n")
        parsed = onellm.load_routing_map(str(path))
        assert parsed["code"]["default"] == "anthropic/claude-3"

    def test_load_routing_map_unsupported_extension(self, tmp_path):
        path = tmp_path / "map.toml"
        path.write_text("x = 1")
        with pytest.raises(RoutingConfigurationError, match="unsupported file type"):
            onellm.load_routing_map(str(path))

    def test_load_routing_map_non_mapping(self, tmp_path):
        path = tmp_path / "map.json"
        path.write_text('["openai/gpt-4"]')
        with pytest.raises(RoutingConfigurationError, match="mapping"):
            onellm.load_routing_map(str(path))

    def test_is_auto_model(self):
        assert is_auto_model("auto")
        assert is_auto_model("auto/code")
        assert not is_auto_model("autopilot/model")
        assert not is_auto_model("openai/gpt-4")
        assert not is_auto_model(None)


# ---------------------------------------------------------------------------
# API-layer integration (ChatCompletion / Completion)
# ---------------------------------------------------------------------------


def _mock_provider(response):
    provider = MagicMock()
    provider.json_mode_support = True
    provider.streaming_support = True
    provider.vision_support = True
    provider.audio_input_support = True
    provider.create_chat_completion = AsyncMock(return_value=response)
    provider.create_completion = AsyncMock(return_value=response)
    return provider


def _chat_response(model="google/gemini-pro"):
    return ChatCompletionResponse(
        id="resp-1",
        object="chat.completion",
        created=0,
        model=model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    )


class TestApiIntegration:
    @pytest.fixture(autouse=True)
    def _clean_routing(self):
        yield
        onellm.disable_routing()

    def test_auto_without_init_raises(self):
        with pytest.raises(InvalidConfigurationError, match="init_routing"):
            ChatCompletion.create(model="auto", messages=[_user("hi")])

    def test_auto_without_init_raises_completion(self):
        with pytest.raises(InvalidConfigurationError, match="init_routing"):
            Completion.create(model="auto", prompt="hi")

    def test_auto_in_fallback_models_rejected(self):
        with pytest.raises(InvalidRequestError, match="fallback_models"):
            ChatCompletion.create(
                model="openai/gpt-4",
                messages=[_user("hi")],
                fallback_models=["auto/code"],
            )

    def test_routed_call_resolves_model_and_attaches_routing(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        provider = _mock_provider(_chat_response())
        with patch(
            "onellm.chat_completion.get_provider_with_fallbacks",
            return_value=(provider, "gemini-pro"),
        ) as gp:
            response = ChatCompletion.create(
                model="auto/code", messages=[_user("fix the frontend layout")]
            )
        assert gp.call_args.kwargs["primary_model"] == "google/gemini-pro"
        assert response.routing["resolved_path"] == "code/frontend"
        assert response.routing["source"] == "classified"

    def test_resolved_chain_prepends_caller_fallbacks(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        provider = _mock_provider(_chat_response())
        with patch(
            "onellm.chat_completion.get_provider_with_fallbacks",
            return_value=(provider, "claude-3-sonnet"),
        ) as gp:
            ChatCompletion.create(
                model="auto/code/typescript",
                messages=[_user("hi")],
                fallback_models=["openai/gpt-4o-mini"],
            )
        assert gp.call_args.kwargs["primary_model"] == "anthropic/claude-3-sonnet"
        assert gp.call_args.kwargs["fallback_models"] == [
            "openai/gpt-4",
            "openai/gpt-4o-mini",
        ]

    async def test_acreate_routes(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        provider = _mock_provider(_chat_response())
        with patch(
            "onellm.chat_completion.get_provider_with_fallbacks",
            return_value=(provider, "gemini-pro"),
        ):
            response = await ChatCompletion.acreate(
                model="auto/code", messages=[_user("fix the frontend layout")]
            )
        assert response.routing["model"] == "google/gemini-pro"

    def test_completion_prompt_routes(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        provider = _mock_provider(_chat_response())
        with patch(
            "onellm.completion.get_provider_with_fallbacks",
            return_value=(provider, "sonar-pro"),
        ) as gp:
            response = Completion.create(model="auto", prompt="research research this topic")
        assert gp.call_args.kwargs["primary_model"] == "perplexity/sonar-pro"
        assert response.routing["resolved_path"] == "research"

    def test_non_auto_model_untouched(self, monkeypatch):
        # Routing enabled must not affect concrete model calls
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        provider = _mock_provider(_chat_response())
        with patch(
            "onellm.chat_completion.get_provider_with_fallbacks",
            return_value=(provider, "gpt-4"),
        ):
            response = ChatCompletion.create(model="openai/gpt-4", messages=[_user("hi")])
        assert response.routing is None  # declared field, None unless routed

    def test_cached_response_carries_routing(self, monkeypatch):
        router = make_router(monkeypatch)
        monkeypatch.setattr(onellm, "_routing", router)
        cached = _chat_response()
        fake_cache = MagicMock()
        fake_cache.get.return_value = cached
        monkeypatch.setattr(onellm, "_cache", fake_cache)
        try:
            response = ChatCompletion.create(
                model="auto/code", messages=[_user("fix the frontend layout")]
            )
        finally:
            monkeypatch.setattr(onellm, "_cache", None)
        assert response is cached
        assert response.routing["resolved_path"] == "code/frontend"
        # The cache was consulted with the RESOLVED model, not "auto/code"
        assert fake_cache.get.call_args.args[0] == "google/gemini-pro"

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
Auto routing for OneLLM.

The developer registers a routing map at startup via ``onellm.init_routing()``,
then calls ``ChatCompletion.create(model="auto")`` or ``model="auto/code"``.
A local embedding classifier reads the conversation, picks the best-matching
label from the map, and resolves it to a concrete ``provider/model`` string
plus an optional fallback chain.

Design invariants (see onellm-auto-routing-prd.md):

- The map is 100% developer-authored. OneLLM ships no opinions about which
  model is good at what.
- ``default`` is required on every group, so a low-confidence stop always
  resolves locally and the resolution walk always terminates.
- Classification is local (ONNX embedding stack shared with the semantic
  cache) - no network hop, no LLM-based classification.
- Stateless: no session objects, no conversation tracking. The only mutable
  state is a bounded, process-local memo LRU and counters.
- Everything that can be validated at ``init_routing`` is validated there.
"""

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, cast

from .errors import InvalidRequestError, RoutingConfigurationError

logger = logging.getLogger("onellm.routing")

# Reserved keys in the map schema. ``default`` names a group's fallback
# leaf, ``models`` marks an annotated leaf, ``api_keys`` is only legal at
# the top level (extracted before compilation).
_RESERVED_KEYS = {"default", "models", "api_keys"}

# Label names must compose cleanly into a path like ``auto/code/frontend``.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Sibling-distinguishability thresholds. Advisory only - never fatal.
# Absolute cosine values shift when the embedding model changes, and the
# runtime already resolves confusable siblings safely (bunched scores fail
# the margin test and land on the group's mandatory default).
_SIBLING_WARN = 0.90
_SIBLING_STRONG_WARN = 0.98

# Per-section head-truncation budget for the routing slice (tokens).
_SECTION_TOKENS = 200

# Providers that authenticate through config fields other than ``api_key``.
# Credential validation checks these fields instead.
_ALT_CREDENTIAL_FIELDS = {
    "vertexai": "service_account_json",
    "azure": "azure_config_path",
}
# Providers that need no credentials at all.
_NO_CREDENTIAL_PROVIDERS = {"ollama", "llama_cpp", "local", "bedrock"}

_DEBUG_ENV = "ONELLM_ROUTING_DEBUG"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RoutingConfig:
    """Configuration for routing behavior. Mirrors ``CacheConfig``'s role."""

    embedding_model: str = ""  # resolved by init_routing; "local/<hf-repo>"
    min_margin: float = 0.05
    min_score: float | None = None
    max_depth: int = 3
    max_slice_tokens: int = 1500
    recent_user_turns: int = 3
    memo_size: int = 256
    validate_similarity: bool = True


# ---------------------------------------------------------------------------
# Map compilation (pure - no ML, no I/O)
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """A compiled node in the routing map: a leaf, or a group with children.

    A leaf has ``models`` set (the first entry is the model, the rest the
    fallback chain). A group has ``children`` and a mandatory ``default``
    leaf. ``exemplars`` is filled in by the Router after embedding.
    """

    path: str  # e.g. "code/frontend"; "" for root
    label: str  # e.g. "frontend"; "" for root
    models: list[str] | None = None
    description: str | None = None
    examples: list[str] = field(default_factory=list)
    children: "dict[str, _Node]" = field(default_factory=dict)
    default: "_Node | None" = None
    parent: "_Node | None" = None
    exemplars: Any = None  # np.ndarray (n_exemplars, dim), set by Router

    @property
    def is_leaf(self) -> bool:
        return self.models is not None

    def exemplar_texts(self) -> list[str]:
        """Texts embedded as this label's exemplars: name, description, examples.

        Stored individually, never averaged into a centroid - scoring
        max-pools over (chunk, exemplar) pairs.
        """
        # Underscores/hyphens read as word separators to the embedding model.
        texts = [self.label.replace("_", " ").replace("-", " ")]
        if self.description:
            texts.append(self.description)
        texts.extend(self.examples)
        return texts


def _config_error(path: str, message: str) -> RoutingConfigurationError:
    where = f" (at {path!r})" if path else " (at map root)"
    return RoutingConfigurationError(f"Invalid routing map{where}: {message}")


def _compile_leaf_models(value: Any, path: str) -> list[str]:
    """Validate and normalize a leaf's model value (str or list[str])."""
    from .providers.base import parse_model_name

    if isinstance(value, str):
        models = [value]
    elif isinstance(value, list):
        if not value:
            raise _config_error(path, "an empty list is not a valid fallback chain")
        if not all(isinstance(m, str) for m in value):
            raise _config_error(path, "fallback chains must contain only strings")
        models = list(value)
    else:
        raise _config_error(
            path, f"expected a model string, list of models, or dict, got {type(value).__name__}"
        )

    for m in models:
        try:
            parse_model_name(m)
        except ValueError as e:
            raise _config_error(path, str(e))
    return models


def _compile_node(value: Any, path: str, label: str) -> _Node:
    """Recursively compile a map value into a ``_Node``."""
    # Plain leaf: str or list[str]
    if not isinstance(value, dict):
        return _Node(path=path, label=label, models=_compile_leaf_models(value, path))

    # Annotated leaf: a dict containing ``models`` is always a leaf, never a group
    if "models" in value:
        unknown = set(value) - {"models", "description", "examples"}
        if unknown:
            raise _config_error(
                path,
                f"annotated leaf contains unexpected keys {sorted(unknown)}; "
                "allowed: models, description, examples",
            )
        description = value.get("description")
        if description is not None and not isinstance(description, str):
            raise _config_error(path, "description must be a string")
        examples = value.get("examples", [])
        if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
            raise _config_error(path, "examples must be a list of strings")
        return _Node(
            path=path,
            label=label,
            models=_compile_leaf_models(value["models"], path),
            description=description,
            examples=examples,
        )

    # Group
    if not value:
        raise _config_error(path, "empty group; add labels or replace with a model string")
    if "default" not in value:
        raise _config_error(
            path,
            "every group requires a 'default' key so low-confidence requests "
            "resolve locally instead of climbing to an unrelated ancestor",
        )

    node = _Node(path=path, label=label)
    for key, child_value in value.items():
        if key == "default":
            default_path = f"{path}/default" if path else "default"
            default_node = _compile_node(child_value, default_path, "default")
            if not default_node.is_leaf:
                raise _config_error(
                    default_path,
                    "'default' must be a leaf (model string, fallback list, or "
                    "dict with 'models'), not a nested group",
                )
            default_node.parent = node
            node.default = default_node
            continue
        if key in _RESERVED_KEYS:
            raise _config_error(path, f"{key!r} is a reserved key and cannot be a task label")
        if not isinstance(key, str) or not _LABEL_RE.match(key):
            raise _config_error(
                path,
                f"label {key!r} is invalid; labels must match ^[a-z0-9][a-z0-9_-]*$",
            )
        child_path = f"{path}/{key}" if path else key
        child = _compile_node(child_value, child_path, key)
        child.parent = node
        node.children[key] = child

    return node


def _compile_map(routing_map: dict) -> _Node:
    """Compile and validate the developer's routing map into a node tree."""
    if not isinstance(routing_map, dict):
        raise RoutingConfigurationError(
            f"routing_map must be a dict, got {type(routing_map).__name__}"
        )
    if "models" in routing_map:
        raise _config_error("", "the root must be a group, not an annotated leaf")
    return _compile_node(routing_map, "", "")


def _collect_leaves(node: _Node) -> "list[_Node]":
    leaves = []
    if node.is_leaf:
        leaves.append(node)
    if node.default is not None:
        leaves.append(node.default)
    for child in node.children.values():
        leaves.extend(_collect_leaves(child))
    return leaves


def _collect_groups(node: _Node) -> "list[_Node]":
    groups = []
    if not node.is_leaf:
        groups.append(node)
    for child in node.children.values():
        groups.extend(_collect_groups(child))
    return groups


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


def _apply_api_keys(api_keys: dict) -> None:
    """Apply ``init_routing(api_keys=...)`` through the existing config paths.

    There is deliberately no routing-scoped credential store: a routed call
    and a direct ``provider/model`` call must always resolve credentials
    identically. ``set_api_key`` / ``update_provider_config`` also bump
    ``config_version()``, so memoized provider instances invalidate
    transparently.
    """
    from .config import config as _config
    from .config import set_api_key, update_provider_config

    if not isinstance(api_keys, dict):
        raise RoutingConfigurationError(
            f"api_keys must be a dict of provider -> key/config, got {type(api_keys).__name__}"
        )

    providers_config = cast(dict, _config["providers"])
    for provider, value in api_keys.items():
        if provider not in providers_config:
            known = ", ".join(sorted(providers_config))
            raise RoutingConfigurationError(
                f"api_keys references unknown provider {provider!r}. Known providers: {known}"
            )
        if isinstance(value, str):
            key = value
            if key.startswith("env:"):
                var = key[4:]
                resolved = os.environ.get(var)
                if not resolved:
                    raise RoutingConfigurationError(
                        f"api_keys[{provider!r}] references environment variable "
                        f"{var!r}, which is not set. Set it or supply the key directly."
                    )
                key = resolved
            if not key:
                raise RoutingConfigurationError(f"api_keys[{provider!r}] is an empty string")
            set_api_key(key, provider)
        elif isinstance(value, dict):
            update_provider_config(provider, **value)
        else:
            raise RoutingConfigurationError(
                f"api_keys[{provider!r}] must be a key string ('sk-...' or 'env:VAR') "
                f"or a provider config dict, got {type(value).__name__}"
            )


def _validate_credentials(root: _Node) -> None:
    """Fail fast on providers that can't serve requests.

    Registry membership is a hard error for every referenced provider.
    Credential presence is only checked for providers whose config schema
    exposes a credential slot - ``ollama`` / ``llama_cpp`` need none, and
    ``vertexai`` / ``azure`` are checked against their own fields.
    """
    from .config import get_provider_config
    from .providers import list_providers

    known = set(list_providers())
    referenced: dict[str, str] = {}
    for leaf in _collect_leaves(root):
        for model in leaf.models or []:
            provider = model.split("/", 1)[0]
            referenced.setdefault(provider, leaf.path or "default")

    for provider, where in sorted(referenced.items()):
        if provider not in known:
            raise _config_error(
                where,
                f"provider {provider!r} is not supported. "
                f"Supported providers: {', '.join(sorted(known))}",
            )
        if provider in _NO_CREDENTIAL_PROVIDERS:
            continue
        conf = get_provider_config(provider)
        if not conf:
            continue  # provider registered but not in the config schema; nothing to check
        credential_field = _ALT_CREDENTIAL_FIELDS.get(provider, "api_key")
        if credential_field in conf and not conf.get(credential_field):
            raise _config_error(
                where,
                f"provider {provider!r} has no resolvable credentials "
                f"(missing {credential_field}). Set the provider's environment "
                f"variable, call onellm.set_api_key(), or pass api_keys= to "
                f"init_routing().",
            )


# ---------------------------------------------------------------------------
# Slice assembly (deterministic - no ML)
# ---------------------------------------------------------------------------


def _content_text(content: Any) -> str:
    """Extract text from a message's content, ignoring non-text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(p for p in parts if p)
    return ""


def _assemble_slice_sections(
    messages: "list[dict] | None",
    tools: "list[dict] | None",
    prompt: "str | None",
    recent_user_turns: int,
) -> "tuple[list[str], list[str]]":
    """Build the routing-slice sections in priority order.

    Returns ``(fixed_sections, recent_sections)``. The recent window is
    kept separate because the cap trims from the middle of it.
    """
    fixed: list[str] = []

    # 1. Tool names + first-line descriptions. The loudest signal, present
    #    on turn one. Names only - parameter schemas are noise.
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        desc = fn.get("description", "") if isinstance(fn, dict) else ""
        first_line = desc.splitlines()[0] if desc else ""
        if name:
            fixed.append(f"tool: {name} - {first_line}" if first_line else f"tool: {name}")

    if prompt is not None:
        # Completion API path: the prompt is the single user turn.
        return fixed + [prompt], []

    system_texts = []
    user_texts = []
    for msg in messages or []:
        role = msg.get("role")
        text = _content_text(msg.get("content", ""))
        if not text:
            continue
        if role == "system":
            system_texts.append(text)
        elif role == "user":
            user_texts.append(text)
        # assistant turns and tool results are dropped: they are the bulk of
        # the tokens, and classifying on prior output entrenches the
        # previous routing decision.

    # 2. System prompt, head-truncated (role framing lives at the top).
    if system_texts:
        fixed.append(" ".join(system_texts))

    # 3. First user turn; 4. last N user turns, most recent last. Dedupe:
    #    if the first turn is inside the recent window, include it once.
    recent = user_texts[-recent_user_turns:] if recent_user_turns > 0 else []
    if user_texts and user_texts[0] not in recent:
        fixed.append(user_texts[0])

    return fixed, recent


# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """The outcome of one routing decision. Attached to responses as
    ``response.routing`` (dict form) and returned by ``explain_route``."""

    entry: str
    resolved_path: str
    model: str
    fallback_chain: list[str]
    source: str  # classified | explicit | memo | default
    scores: dict[str, float]
    margin: float | None
    slice_tokens: int
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "entry": self.entry,
            "resolved_path": self.resolved_path,
            "model": self.model,
            "fallback_chain": list(self.fallback_chain),
            "source": self.source,
            "scores": dict(self.scores),
            "margin": self.margin,
            "slice_tokens": self.slice_tokens,
            "latency_ms": self.latency_ms,
        }


def is_auto_model(model: Any) -> bool:
    """True when ``model`` requests auto routing (``auto`` or ``auto/...``)."""
    return isinstance(model, str) and (model == "auto" or model.startswith("auto/"))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Owns the compiled map, the embedding backend, exemplars, memo, stats.

    Instances are immutable after ``__init__`` apart from the memo LRU and
    counters, both guarded by a lock, so concurrent ``create`` calls are safe.
    """

    def __init__(
        self,
        routing_map: dict,
        config: RoutingConfig,
        api_keys: "dict | None" = None,
    ) -> None:
        self.config = config

        # A file-based map may carry a reserved top-level api_keys section;
        # the kwarg wins per-provider on conflict.
        map_keys = routing_map.get("api_keys") if isinstance(routing_map, dict) else None
        if map_keys is not None:
            merged = dict(map_keys)
            merged.update(api_keys or {})
            api_keys = merged
            routing_map = {k: v for k, v in routing_map.items() if k != "api_keys"}

        if api_keys:
            _apply_api_keys(api_keys)

        self.root = _compile_map(routing_map)
        _validate_credentials(self.root)

        self._np, self._backend = self._load_backend()
        self._tokenizer = self._resolve_tokenizer(self._backend)
        self._chunk_tokens = self._resolve_chunk_tokens()
        self._embed_exemplars()
        if config.validate_similarity:
            self._warn_confusable_siblings()

        self._memo: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            "classified": 0,
            "memo_hits": 0,
            "explicit": 0,
            "fell_back_to_default": 0,
        }
        self._latency_total_ms = 0.0
        self._latency_count = 0

    # -- backend ---------------------------------------------------------

    def _load_backend(self) -> "tuple[Any, Any]":
        """Load numpy + the embedding backend through ``LocalProvider``.

        Same single code path the semantic cache uses, so the class-level
        ``(repo, revision)`` LRU shares the loaded model between features.
        """
        model = self.config.embedding_model
        if not model.startswith("local/"):
            raise RoutingConfigurationError(
                f"embedding_model must be a local/ model (in-process, no network "
                f"hop), got {model!r}"
            )
        repo = model[len("local/") :]

        try:
            import numpy as np

            from .providers.local import LocalProvider
        except ImportError as e:
            raise RoutingConfigurationError(
                f"Auto routing requires the local embedding stack: {e}. "
                f"Install with: pip install 'onellm[routing]'"
            )

        try:
            provider = LocalProvider()
            backend = provider._load_model(repo, trust_remote_code=False)
        except RoutingConfigurationError:
            raise
        except Exception as e:
            # InvalidConfigurationError (missing backend deps) or normalized
            # HF errors. Unlike the cache there is no degraded mode -
            # classification IS the feature - so surface loudly at init.
            raise RoutingConfigurationError(
                f"Auto routing could not load embedding model {repo!r}: {e}. "
                f"Install the routing stack with: pip install 'onellm[routing]'"
            )
        return np, backend

    @staticmethod
    def _resolve_tokenizer(backend: Any) -> Any:
        """The embedding model's own tokenizer, so the slice cap and the
        chunker never drift. Falls back to word-splitting when absent."""
        tokenizer = getattr(backend, "tokenizer", None)
        if tokenizer is None:
            st_model = getattr(backend, "st_model", None)
            tokenizer = getattr(st_model, "tokenizer", None)
        return tokenizer

    def _resolve_chunk_tokens(self) -> int:
        """Chunk size from the model's effective window, not its accepted max.

        The default multilingual MiniLM-L12 accepts 512 tokens but was
        trained on 128; embedding beyond the training window dilutes the
        vectors. The training window is not discoverable programmatically,
        so we special-case the known default repo and cap everything else
        at 256 (the PRD floor).
        """
        from .cache import _CACHE_MODEL_REPO

        repo = self.config.embedding_model[len("local/") :]
        model_max = getattr(self._backend, "model_max_length", None) or 256
        if repo == _CACHE_MODEL_REPO:
            return min(model_max, 128)
        return min(model_max, 256)

    # -- tokenization helpers ---------------------------------------------

    def _encode_ids(self, text: str) -> list:
        if self._tokenizer is not None:
            try:
                return self._tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return self._tokenizer.encode(text)
        return text.split()

    def _decode_ids(self, ids: list) -> str:
        if self._tokenizer is not None:
            try:
                return self._tokenizer.decode(ids, skip_special_tokens=True)
            except TypeError:
                return self._tokenizer.decode(ids)
        return " ".join(str(i) for i in ids)

    def _truncate(self, text: str, limit: int) -> str:
        ids = self._encode_ids(text)
        if len(ids) <= limit:
            return text
        return self._decode_ids(ids[:limit])

    def _count_tokens(self, text: str) -> int:
        return len(self._encode_ids(text))

    # -- exemplars ---------------------------------------------------------

    def _embed_exemplars(self) -> None:
        """Embed every label's exemplars in one batch at init.

        Exemplars are stored individually per label, never averaged -
        scoring max-pools over (chunk, exemplar) pairs.
        """
        labels: list[_Node] = []
        for group in _collect_groups(self.root):
            labels.extend(group.children.values())

        texts: list[str] = []
        spans: list[tuple[_Node, int, int]] = []
        for node in labels:
            node_texts = node.exemplar_texts()
            spans.append((node, len(texts), len(texts) + len(node_texts)))
            texts.extend(node_texts)

        if not texts:
            return
        embeddings = self._np.asarray(self._backend.encode(texts))
        for node, start, end in spans:
            node.exemplars = embeddings[start:end]

    def _warn_confusable_siblings(self) -> None:
        """Advisory check: warn when two sibling labels are hard to tell apart.

        Never fatal. The margin test already resolves confusable siblings
        to the group's mandatory default at request time, so the map boots
        and routing degrades locally and predictably.
        """
        for group in _collect_groups(self.root):
            siblings = list(group.children.values())
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    a, b = siblings[i], siblings[j]
                    if a.exemplars is None or b.exemplars is None:
                        continue
                    # Backends return L2-normalized vectors, so the inner
                    # product is cosine similarity.
                    sim = float((a.exemplars @ b.exemplars.T).max())
                    if sim >= _SIBLING_STRONG_WARN:
                        logger.warning(
                            "Routing labels %r and %r are effectively "
                            "indistinguishable (max exemplar similarity %.3f). "
                            "They will nearly always fall back to this group's "
                            "default. Merge them or rewrite their descriptions/"
                            "examples.",
                            a.path,
                            b.path,
                            sim,
                        )
                    elif sim >= _SIBLING_WARN:
                        logger.warning(
                            "Routing labels %r and %r are similar (max exemplar "
                            "similarity %.3f) and may be hard to distinguish; "
                            "consider sharper descriptions or examples.",
                            a.path,
                            b.path,
                            sim,
                        )

    # -- slice -------------------------------------------------------------

    def _assemble_slice(
        self,
        messages: "list[dict] | None",
        tools: "list[dict] | None",
        prompt: "str | None",
    ) -> str:
        fixed, recent = _assemble_slice_sections(
            messages, tools, prompt, self.config.recent_user_turns
        )
        fixed = [self._truncate(s, _SECTION_TOKENS) for s in fixed]
        recent = [self._truncate(s, _SECTION_TOKENS) for s in recent]

        def total(parts: list[str]) -> int:
            return sum(self._count_tokens(p) for p in parts)

        # Cap the document, trimming from the middle of the recent window:
        # the first and most recent turns carry the most signal.
        cap = self.config.max_slice_tokens
        while recent and total(fixed + recent) > cap and len(recent) > 1:
            recent.pop(len(recent) // 2)

        doc = "\n".join(fixed + recent).strip()
        if self._count_tokens(doc) > cap:
            doc = self._truncate(doc, cap)
        return doc

    # -- classification ----------------------------------------------------

    def _score_children(self, node: _Node, chunk_embeddings: Any) -> dict[str, float]:
        """score(label) = max over all (chunk, exemplar) cosine pairs.

        Max-pool at both levels: mean-pooling washes a single strongly
        matching chunk out of a long conversation, which is precisely the
        case the feature exists to handle.
        """
        scores = {}
        for label, child in node.children.items():
            if child.exemplars is None or len(child.exemplars) == 0:
                scores[label] = 0.0
                continue
            scores[label] = float((chunk_embeddings @ child.exemplars.T).max())
        return scores

    def _classify(
        self, node: _Node, slice_text: str, depth_budget: int
    ) -> "tuple[_Node, dict[str, float], float | None, bool]":
        """Descend from ``node`` guided by embeddings.

        Returns ``(final_node, last_scores, last_margin, descended)``.
        """
        ids = self._encode_ids(slice_text)
        chunks = [
            self._decode_ids(ids[i : i + self._chunk_tokens])
            for i in range(0, len(ids), self._chunk_tokens)
        ] or [slice_text]
        chunk_embeddings = self._np.asarray(self._backend.encode(chunks))

        scores: dict[str, float] = {}
        margin: float | None = None
        descended = False
        depth = 0
        while node.children and depth < depth_budget:
            if len(node.children) == 1:
                # Single child: no top-2 exists; descend unconditionally.
                node = next(iter(node.children.values()))
                descended = True
                depth += 1
                continue
            scores = self._score_children(node, chunk_embeddings)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top1_label, top1 = ranked[0]
            margin = top1 - ranked[1][1]
            if margin < self.config.min_margin:
                break  # scores bunched - stop and resolve at this node's default
            if self.config.min_score is not None and top1 < self.config.min_score:
                break
            node = node.children[top1_label]
            descended = True
            depth += 1
        return node, scores, margin, descended

    # -- resolution ----------------------------------------------------------

    @staticmethod
    def _resolve_leaf(node: _Node) -> "tuple[_Node, bool]":
        """Walk from ``node`` toward the root; return the first leaf found.

        Every group carries a mandatory default, so in practice this
        terminates at the current node; the walk-up loop is kept as the
        single code path covering all cases. The bool is True when
        resolution used a ``default`` key rather than a labeled leaf.
        """
        current: _Node | None = node
        while current is not None:
            if current.is_leaf:
                return current, current.label == "default"
            if current.default is not None:
                return current.default, True
            current = current.parent
        raise RoutingConfigurationError(
            "routing map has no resolvable default; this should have been " "caught at init_routing"
        )

    def _consume_explicit_segments(self, entry: str) -> "tuple[_Node, list[str]]":
        """Consume path segments after ``auto/`` without inference.

        An unknown segment raises: ``auto/reserch`` is a typo, not a routing
        decision, and silently falling back to default would hide it.
        """
        segments = [s for s in entry.split("/")[1:] if s]
        node = self.root
        consumed: list[str] = []
        for segment in segments:
            if node.is_leaf:
                raise InvalidRequestError(
                    f"Invalid routing path {entry!r}: "
                    f"{'/'.join(consumed)!r} is already a leaf; "
                    f"segment {segment!r} goes deeper than the routing map"
                )
            if segment not in node.children:
                valid = ", ".join(sorted(node.children)) or "(none)"
                raise InvalidRequestError(
                    f"Invalid routing path {entry!r}: unknown label {segment!r}. "
                    f"Valid labels here: {valid}"
                )
            node = node.children[segment]
            consumed.append(segment)
        return node, consumed

    # -- public API ----------------------------------------------------------

    def resolve(
        self,
        entry: str,
        messages: "list[dict] | None" = None,
        tools: "list[dict] | None" = None,
        prompt: "str | None" = None,
    ) -> RouteResult:
        """Resolve an ``auto...`` entry to a concrete model + fallback chain."""
        start = time.perf_counter()
        node, consumed = self._consume_explicit_segments(entry)

        # Fully explicit path (or a group with nothing left to classify):
        # no slice, no embedding.
        if node.is_leaf or not node.children:
            leaf, used_default = self._resolve_leaf(node)
            return self._finish(
                entry=entry,
                leaf=leaf,
                anchor=node,
                source="explicit" if node.is_leaf else "default",
                scores={},
                margin=None,
                slice_tokens=0,
                used_default=used_default,
                start=start,
            )

        slice_text = self._assemble_slice(messages, tools, prompt)
        if os.environ.get(_DEBUG_ENV):
            logger.debug("Routing slice for entry %r:\n%s", entry, slice_text)
        slice_tokens = self._count_tokens(slice_text)

        # Memo check MUST precede any embedding: a cached exact-hash
        # response returns in microseconds today, and routing must not
        # turn that into a multi-millisecond call.
        entry_path = "/".join(consumed)
        memo_key = (hashlib.sha256(slice_text.encode()).hexdigest(), entry_path)
        with self._lock:
            hit = self._memo.get(memo_key)
            if hit is not None:
                self._memo.move_to_end(memo_key)
                self._stats["memo_hits"] += 1
                latency_ms = (time.perf_counter() - start) * 1000
                self._latency_total_ms += latency_ms
                self._latency_count += 1
                return RouteResult(
                    entry=entry,
                    resolved_path=hit.resolved_path,
                    model=hit.model,
                    fallback_chain=list(hit.fallback_chain),
                    source="memo",
                    scores=dict(hit.scores),
                    margin=hit.margin,
                    slice_tokens=slice_tokens,
                    latency_ms=latency_ms,
                )

        if not slice_text:
            # Nothing classifiable (e.g. image-only messages): resolve at
            # the entry node's default.
            leaf, used_default = self._resolve_leaf(node)
            return self._finish(entry, leaf, node, "default", {}, None, 0, used_default, start)

        final_node, scores, margin, descended = self._classify(
            node, slice_text, self.config.max_depth
        )
        leaf, used_default = self._resolve_leaf(final_node)
        source = "classified" if descended else "default"
        result = self._finish(
            entry,
            leaf,
            final_node,
            source,
            scores,
            margin,
            slice_tokens,
            used_default,
            start,
        )
        with self._lock:
            self._memo[memo_key] = result
            self._memo.move_to_end(memo_key)
            while len(self._memo) > self.config.memo_size:
                self._memo.popitem(last=False)
        return result

    async def aresolve(
        self,
        entry: str,
        messages: "list[dict] | None" = None,
        tools: "list[dict] | None" = None,
        prompt: "str | None" = None,
    ) -> RouteResult:
        """Async resolve. ONNX inference runs in a thread executor so the
        event loop is never blocked."""
        return await asyncio.to_thread(
            self.resolve, entry, messages=messages, tools=tools, prompt=prompt
        )

    def _finish(
        self,
        entry: str,
        leaf: _Node,
        anchor: _Node,
        source: str,
        scores: dict[str, float],
        margin: "float | None",
        slice_tokens: int,
        used_default: bool,
        start: float,
    ) -> RouteResult:
        models = list(leaf.models or [])
        resolved_path = leaf.path if leaf.label != "default" else (anchor.path or "default")
        latency_ms = (time.perf_counter() - start) * 1000
        with self._lock:
            if source == "classified":
                self._stats["classified"] += 1
            elif source == "explicit":
                self._stats["explicit"] += 1
            if used_default:
                self._stats["fell_back_to_default"] += 1
            self._latency_total_ms += latency_ms
            self._latency_count += 1
        return RouteResult(
            entry=entry,
            resolved_path=resolved_path,
            model=models[0],
            fallback_chain=models[1:],
            source=source,
            scores=scores,
            margin=margin,
            slice_tokens=slice_tokens,
            latency_ms=latency_ms,
        )

    def stats(self) -> dict:
        with self._lock:
            avg = self._latency_total_ms / self._latency_count if self._latency_count else 0.0
            return {**self._stats, "avg_latency_ms": avg}

---
layout: default
title: Auto Routing
nav_order: 9
---

# Auto Routing

OneLLM can pick the right model for each request. Register a routing map once at startup, then call any completion API with `model="auto"` — a local embedding classifier reads the conversation and resolves it to a concrete `provider/model` plus a fallback chain.

## Overview

Routing is **100% developer-authored**: OneLLM ships no opinions about which model is good at what. You describe your labels (with example phrases), and the classifier matches each conversation against *your* exemplars:

1. **Local classification** - a small multilingual embedding model (the same one the semantic cache uses) scores the conversation against each label's examples. No API calls, no data leaves the process.
2. **Deterministic resolution** - the best-scoring label wins when it clears a score margin; otherwise the group's mandatory `default` is used. There is no "unroutable" error at request time.
3. **Memoization** - repeated calls on an unchanged conversation slice skip inference entirely (~µs lookup).
4. **Full observability** - every non-streaming response carries `response.routing` explaining the decision, and `onellm.explain_route()` dry-runs a decision without a provider call.

**Key properties:**

- 🔒 **Privacy-focused** - classification happens locally, in-process
- 🧭 **Deterministic fallbacks** - every group requires a `default`; low-confidence requests fall back, never fail
- ⚡ **Fast** - a few ms to classify, ~µs on memo hits
- 🌍 **Multilingual** - the default embedding model covers 50+ languages
- ➕ **Strictly additive** - nothing changes unless you call `init_routing()`; concrete `provider/model` strings behave exactly as before

## Installation

```bash
pip install "onellm[routing]"
```

This installs the same local embedding stack as `onellm[cache]` — if you already use the semantic cache, routing adds no new dependencies and shares the loaded model.

## Quick Start

```python
import onellm
from onellm import ChatCompletion

onellm.init_routing({
    "default": "openai/gpt-5-mini",
    "code": {
        "description": "Programming, debugging, code review",
        "examples": ["fix this function", "why does this test fail?"],
        "models": ["anthropic/claude-sonnet-4-5", "openai/gpt-5"],
    },
    "research": {
        "description": "Deep analysis, literature review, long documents",
        "examples": ["compare these papers", "summarize this report in depth"],
        "models": "openai/gpt-5",
    },
})

# The classifier picks the label; the label resolves to a model
response = ChatCompletion.create(
    model="auto",
    messages=[{"role": "user", "content": "Why does this test fail intermittently?"}],
)

print(response.model)    # e.g. "claude-sonnet-4-5"
print(response.routing)  # full decision record (see Observability below)
```

If `model="auto"` is used before `init_routing()` is called, OneLLM raises `InvalidConfigurationError` — routing never activates implicitly.

## The Routing Map

A map is a tree. Each key is a label; each value is either a **leaf** (what to run) or a **group** (a nested dict of labels). Every group — including the root — must have a `default`.

**Leaf forms:**

```python
"summarize": "openai/gpt-5-mini"                          # single model
"code": ["anthropic/claude-sonnet-4-5", "openai/gpt-5"]   # fallback chain
"research": {                                             # annotated leaf
    "description": "Deep analysis and synthesis",
    "examples": ["compare these approaches", "review the literature"],
    "models": ["openai/gpt-5", "anthropic/claude-opus-4-1"],
}
```

**Groups nest** (up to `max_depth`, default 3):

```python
onellm.init_routing({
    "default": "openai/gpt-5-mini",
    "code": {
        "default": "anthropic/claude-sonnet-4-5",
        "frontend": {
            "examples": ["fix this React component", "CSS grid layout issue"],
            "models": "openai/gpt-5",
        },
        "backend": {
            "examples": ["optimize this SQL query", "design this API"],
            "models": "anthropic/claude-sonnet-4-5",
        },
    },
})
```

Classification descends group by group: at each level the best label must beat the runner-up by `min_margin`; otherwise descent stops and that group's `default` applies. A leaf's fallback chain is passed straight to OneLLM's existing [fallback mechanism]({% link advanced-features.md %}).

**Reserved keys:** `default`, `models`, `examples`, `description`, `api_keys` cannot be used as labels. Labels must match `^[a-z0-9][a-z0-9_-]*$`.

**Entry points:** `model="auto"` classifies from the root. `model="auto/code"` skips straight into the `code` subtree (explicit segments are validated at request time; classification continues below them if the target is a group).

### Writing good examples

Labels are matched against your `examples` (each embedded separately, never averaged) plus the `description`. 3-8 short, distinct phrases per label work well. At init, OneLLM warns (never errors) when two sibling labels' examples are nearly identical (cosine ≥ 0.90, strongly ≥ 0.98) — ambiguous siblings just mean more traffic lands on the group's `default`. Pass `validate_similarity=False` to skip the check.

## Loading Maps from Files

Keep the map in version control as YAML or JSON:

```yaml
# routing.yaml
default: openai/gpt-5-mini
code:
  description: Programming, debugging, code review
  examples:
    - fix this function
    - why does this test fail?
  models:
    - anthropic/claude-sonnet-4-5
    - openai/gpt-5
```

```python
onellm.init_routing(onellm.load_routing_map("./routing.yaml"))
```

`load_routing_map()` takes an explicit path only — there is no default location or directory discovery.

## Provider Credentials

`init_routing()` can carry credentials for the providers the map uses, applied through the same paths as `onellm.set_api_key()`:

```python
onellm.init_routing(
    routing_map,
    api_keys={
        "openai": "env:OPENAI_API_KEY",     # read from environment at init
        "anthropic": "sk-ant-...",          # literal key
        "vertexai": {                       # full provider config dict
            "project_id": "my-project",
            "location": "us-central1",
        },
    },
)
```

Validation is fail-fast: every provider referenced in the map must exist, and providers that require an API key must have one available (from `api_keys`, a previous `set_api_key()`, or their environment variable) or `init_routing()` raises `RoutingConfigurationError` at startup — not at 3am when that route first fires. YAML/JSON maps may include a top-level `api_keys` section; use `env:VAR` indirection there rather than literal secrets.

## Configuration

```python
onellm.init_routing(
    routing_map,
    api_keys=None,             # provider credentials (see above)
    embedding_model=None,      # default: same local model as the semantic cache
    min_margin=0.05,           # best label must beat runner-up by this much
    min_score=None,            # optional absolute score floor
    max_depth=3,               # maximum group nesting classified
    max_slice_tokens=1500,     # conversation slice budget
    recent_user_turns=3,       # recent user messages included in the slice
    memo_size=256,             # memoized decisions (LRU)
    validate_similarity=True,  # advisory sibling-similarity warnings at init
)

onellm.disable_routing()       # turn it off (also clears the memo)
```

Raise `min_margin` to send more borderline traffic to defaults; lower it to trust the classifier more.

## What the Classifier Sees

For each request, OneLLM assembles a deterministic slice of the conversation: the first line of any tool definitions, the head of the system message, the first user turn, and the last `recent_user_turns` user turns — deduplicated and trimmed from the middle to fit `max_slice_tokens`. For the `Completion` API, the `prompt` is used directly.

Set `ONELLM_ROUTING_DEBUG=1` to log the assembled slice and per-label scores for every decision.

## Observability

Every non-streaming response carries the decision record; `explain_route()` produces the same record without a provider call:

```python
onellm.explain_route(
    messages=[{"role": "user", "content": "Fix this React component"}],
    entry="auto",
)
# {
#     "entry": "auto",
#     "resolved_path": "code/frontend",
#     "model": "openai/gpt-5",
#     "fallback_chain": [],
#     "source": "classified",      # classified | explicit | memo | default
#     "scores": {"code": 0.71, "research": 0.32, ...},
#     "margin": 0.39,
#     "slice_tokens": 12,
#     "latency_ms": 3.9,
# }
```

Streaming responses are generators and carry no `.routing` attribute — use `explain_route()` when you need the decision for a streamed request.

Aggregate counters:

```python
onellm.routing_stats()
# {"classified": 12, "memo_hits": 40, "explicit": 3,
#  "fell_back_to_default": 1, "avg_latency_ms": 3.9}
```

## Interaction with Other Features

- **Fallbacks** - the resolved chain leads; anything you pass in `fallback_models=` is appended after it. `"auto"` itself is rejected inside `fallback_models` — fallback entries must be concrete.
- **Semantic cache** - routing resolves first, so the cache is keyed on the *resolved* model. Cache hits still carry `response.routing`.
- **Client interface** - `client.chat.completions.create(model="auto", ...)` works; `auto` is exempt from the client's automatic `openai/` prefixing.
- **Async** - `acreate()` runs classification in a thread executor, so the event loop is never blocked on embedding inference.

## Best Practices

1. **Start flat.** A root `default` plus 3-5 top-level labels covers most applications; nest only when a category genuinely splits.
2. **Route on intent, not topic.** "needs deep reasoning" vs "quick answer" separates better than "python" vs "javascript".
3. **Test the map in CI.** Assert `explain_route()` decisions for representative conversations, so map edits can't silently reroute production traffic.
4. **Watch `fell_back_to_default`.** A high ratio in `routing_stats()` means your examples aren't separating — refine them rather than lowering `min_margin` first.

## See Also

- [examples/auto_routing_example.py](https://github.com/muxi-ai/onellm/blob/main/examples/auto_routing_example.py)
- [Semantic Caching]({% link caching.md %}) - shares the same local embedding stack
- [Advanced Features]({% link advanced-features.md %}) - fallback chains and retries

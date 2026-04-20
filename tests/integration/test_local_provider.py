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
Integration tests for the local embedding provider.

These tests download and run real HuggingFace models. They're slow (~30s
cold start per model, ~275 MB for Nomic v1.5) and are skipped unless the
``ONELLM_RUN_LOCAL_INTEGRATION`` env var is set, so CI doesn't pull weights
by accident.

Run with::

    ONELLM_RUN_LOCAL_INTEGRATION=1 pytest tests/integration/test_local_provider.py -m slow
"""

import os

import pytest

RUN_LOCAL = os.environ.get("ONELLM_RUN_LOCAL_INTEGRATION") == "1"

# Skip the whole module unless explicitly opted in - a 275 MB download in CI
# is never what anyone wants.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_LOCAL,
        reason="Set ONELLM_RUN_LOCAL_INTEGRATION=1 to run real model downloads",
    ),
]


class TestNomicV15EndToEnd:
    """Requires HF access + ~275 MB download for Nomic v1.5."""

    async def test_default_tier_produces_768_dim_embeddings(self):
        from onellm.providers.local import LocalProvider

        provider = LocalProvider()
        resp = await provider.create_embedding(
            input=[
                "search_document: The quick brown fox jumps over the lazy dog.",
                "search_document: Hello, world.",
            ],
            model="nomic-ai/nomic-embed-text-v1.5",
            task="search_document",
        )
        assert len(resp.data) == 2
        assert len(resp.data[0].embedding) == 768
        assert len(resp.data[1].embedding) == 768
        assert resp.model == "local/nomic-ai/nomic-embed-text-v1.5"

    async def test_matryoshka_256_dimensions(self):
        from onellm.providers.local import LocalProvider

        provider = LocalProvider()
        resp = await provider.create_embedding(
            input="search_query: capital of france",
            model="nomic-ai/nomic-embed-text-v1.5",
            task="search_query",
            dimensions=256,
        )
        assert len(resp.data[0].embedding) == 256

    async def test_via_public_embedding_api(self):
        """Route through the public Embedding.acreate() surface."""
        import onellm

        resp = await onellm.Embedding.acreate(
            model="local/nomic-ai/nomic-embed-text-v1.5",
            input="search_document: OneLLM is a unified LLM interface.",
            task="search_document",
        )
        assert resp.data[0].embedding
        assert resp.model == "local/nomic-ai/nomic-embed-text-v1.5"

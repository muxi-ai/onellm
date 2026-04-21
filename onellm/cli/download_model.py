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
CLI utility for downloading models from HuggingFace.

Two download modes are supported:

1. **GGUF single-file** (for the ``llama_cpp/`` provider)

   ``onellm download --repo-id <repo> --filename <gguf>``

2. **Full HuggingFace snapshot** (for the ``local/`` embedding provider)

   ``onellm download local/<hf-repo-id>``
   ``onellm download local/nomic-ai/nomic-embed-text-v1.5``
   ``onellm download local/sentence-transformers/all-MiniLM-L6-v2``

The ``local/`` mode uses :func:`huggingface_hub.snapshot_download` to pull the
entire repository (config, tokenizer, weights) into the standard HF cache
(``HF_HOME`` or ``~/.cache/huggingface/hub``), matching sentence-transformers'
expected load location. The model id after ``local/`` is passed straight
through to HuggingFace - there is no alias table.
"""

import argparse
import os
import sys
from pathlib import Path


def download_gguf(repo_id: str, filename: str, output_dir: str | None = None):
    """
    Download a GGUF model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        filename: Model filename to download
        output_dir: Directory to save the model (default: ~/llama_models)

    Returns:
        Path to the downloaded file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface-hub is required but not installed")
        print("This should have been installed with onellm. Try:")
        print("  pip install --upgrade onellm")
        sys.exit(1)

    # Default to ~/llama_models if no output dir specified
    if output_dir is None:
        output_dir = os.path.expanduser("~/llama_models")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")
    print(f"Destination: {output_dir}")

    try:
        # Download with progress bar.
        # Note: `local_dir_use_symlinks` was removed in huggingface_hub>=1.0 -
        # the pin in pyproject.toml (>=1.3.4) always uses the new layout,
        # which writes plain files into `local_dir` without the legacy cache
        # symlink indirection.
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
        )
        print("\n✓ Downloaded successfully!")
        print(f"  File: {file_path}")

        # Show how to use it
        model_name = Path(file_path).name
        print("\nTo use this model with OneLLM:")
        print(f'  model="llama_cpp/{model_name}"')

        return file_path
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        if "404" in str(e):
            print("\nPossible issues:")
            print("  - Check the repository ID is correct")
            print("  - Check the filename exists in the repository")
            print(f"  - Visit https://huggingface.co/{repo_id} to see available files")
        sys.exit(1)


def download_local_model(slug: str, output_dir: str | None = None) -> str:
    """
    Download a full HuggingFace snapshot for use with the ``local/`` provider.

    Args:
        slug: Either a full model id (``local/<org>/<repo>`` or
            ``<org>/<repo>``) or, in rare cases, an unqualified HF repo id.
            Whatever follows ``local/`` is passed directly to HuggingFace
            as the repo id.
        output_dir: Optional explicit cache directory. When omitted,
            ``HF_HOME`` is respected, then ``~/.cache/huggingface/hub``.

    Returns:
        The local filesystem path to the downloaded snapshot.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface-hub is required but not installed")
        print("This should have been installed with onellm. Try:")
        print("  pip install --upgrade onellm")
        sys.exit(1)

    # Strip "local/" prefix if present - what's left is the HF repo id.
    repo_id = slug[len("local/") :] if slug.startswith("local/") else slug

    # Only override ``cache_dir`` when the user explicitly supplied
    # ``--output``. Otherwise let huggingface_hub pick its own default,
    # which resolves to ``$HUGGINGFACE_HUB_CACHE`` (falling back to
    # ``$HF_HOME/hub`` / ``~/.cache/huggingface/hub``) - the same path
    # ``SentenceTransformer`` consults at load time. Passing ``$HF_HOME``
    # directly as ``cache_dir`` places the snapshot at
    # ``$HF_HOME/models--*`` but ``SentenceTransformer`` looks at
    # ``$HF_HOME/hub/models--*``, so the "cached" model is invisible at
    # runtime and inference silently falls back to a fresh download.
    cache_dir = output_dir if output_dir else None

    print(f"Downloading HuggingFace snapshot: {repo_id}")
    if cache_dir:
        print(f"Destination: {cache_dir}")
    else:
        print("Destination: huggingface_hub default cache")

    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - surface provider errors verbatim
        print(f"\n✗ Error downloading snapshot: {exc}")
        if "404" in str(exc):
            print("\nPossible issues:")
            print("  - Check the repository ID is correct")
            print(f"  - Visit https://huggingface.co/{repo_id} to confirm access")
        sys.exit(1)

    print("\n✓ Downloaded successfully!")
    print(f"  Path: {path}")
    print("\nTo use this model with OneLLM:")
    print(f'  await onellm.Embedding.acreate(model="local/{repo_id}", input="...")')

    return path


def _is_local_slug(candidate: str | None) -> bool:
    """Return True if the argument looks like a ``local/...`` slug."""
    if not candidate:
        return False
    return candidate.startswith("local/")


def main():
    """Main entry point for the download command."""
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace for use with OneLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full snapshot for the local/ embedding provider
  onellm download local/nomic-ai/nomic-embed-text-v1.5
  onellm download local/nomic-ai/nomic-embed-text-v2-moe
  onellm download local/sentence-transformers/all-MiniLM-L6-v2

  # Download a single GGUF file for the llama_cpp/ provider
  onellm download -r shinkeonkim/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF \\
                  -f meta-llama-3-8b-instruct-q4_k_m.gguf

  # Download to a custom directory
  onellm download -r TheBloke/Mistral-7B-Instruct-v0.2-GGUF \\
                  -f mistral-7b-instruct-v0.2.Q4_K_M.gguf \\
                  -o /path/to/models

Popular GGUF repositories:
  - TheBloke/* (e.g., TheBloke/Llama-2-7B-GGUF)
  - microsoft/Phi-3-mini-4k-instruct-gguf
  - mistralai/Mistral-7B-Instruct-v0.2-GGUF
        """,
    )

    parser.add_argument(
        "slug",
        nargs="?",
        help="Model slug (e.g. 'local/nomic-ai/nomic-embed-text-v1.5') for full snapshot downloads",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        help="HuggingFace repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')",
    )
    parser.add_argument(
        "--filename",
        "-f",
        help="Model filename for GGUF single-file download (e.g., 'llama-2-7b.Q4_K_M.gguf')",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory (default: ~/llama_models for GGUF, HF cache for local/)",
    )

    args = parser.parse_args()

    # Route to the correct backend based on which arguments were supplied.
    if _is_local_slug(args.slug):
        download_local_model(args.slug, args.output)
        return

    # Treat an unprefixed positional arg as a raw HF repo slug snapshot too -
    # lets callers write `onellm download sentence-transformers/all-MiniLM-L6-v2`.
    if args.slug and not args.repo_id:
        download_local_model(args.slug, args.output)
        return

    if not args.repo_id or not args.filename:
        parser.error(
            "Provide either a slug (e.g. 'local/nomic-ai/nomic-embed-text-v1.5') "
            "or both --repo-id and --filename for GGUF downloads."
        )

    download_gguf(args.repo_id, args.filename, args.output)


if __name__ == "__main__":
    main()

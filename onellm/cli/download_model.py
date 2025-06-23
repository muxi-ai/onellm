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
CLI utility for downloading GGUF models from HuggingFace.
"""

import argparse
import sys
from pathlib import Path
import os


def download_gguf(repo_id: str, filename: str, output_dir: str = None):
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
        # Download with progress bar
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"\n✓ Downloaded successfully!")
        print(f"  File: {file_path}")
        
        # Show how to use it
        model_name = Path(file_path).name
        print(f"\nTo use this model with OneLLM:")
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


def main():
    """Main entry point for the download command."""
    parser = argparse.ArgumentParser(
        description="Download GGUF models for use with OneLLM's llama.cpp provider",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Llama 3 8B model
  onellm download -r shinkeonkim/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF \\
                  -f meta-llama-3-8b-instruct-q4_k_m.gguf

  # Download to custom directory
  onellm download -r TheBloke/Mistral-7B-Instruct-v0.2-GGUF \\
                  -f mistral-7b-instruct-v0.2.Q4_K_M.gguf \\
                  -o /path/to/models

Popular repositories:
  - TheBloke/* (e.g., TheBloke/Llama-2-7B-GGUF)
  - microsoft/Phi-3-mini-4k-instruct-gguf
  - mistralai/Mistral-7B-Instruct-v0.2-GGUF
        """
    )
    
    parser.add_argument(
        "--repo-id", "-r",
        required=True,
        help="HuggingFace repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')"
    )
    parser.add_argument(
        "--filename", "-f",
        required=True,
        help="Model filename (e.g., 'llama-2-7b.Q4_K_M.gguf')"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: ~/llama_models)"
    )
    
    args = parser.parse_args()
    
    # Download the model
    download_gguf(args.repo_id, args.filename, args.output)


if __name__ == "__main__":
    main()
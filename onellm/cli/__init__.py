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


"""OneLLM CLI utilities."""


def main():
    """Main CLI entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        # Remove 'download' from args and call download script
        sys.argv.pop(1)
        from .download_model import main as download_main
        download_main()
    else:
        print("OneLLM CLI")
        print("\nAvailable commands:")
        print("  onellm download - Download GGUF models from HuggingFace")
        print("\nUse 'onellm <command> --help' for more information")


if __name__ == "__main__":
    main()
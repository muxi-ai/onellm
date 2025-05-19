#!/usr/bin/env python3
# Script to verify .version reading

import os

# Version from .version file in the muxi_llm package
with open(os.path.join('muxi_llm', '.version'), 'r') as f:
    version_from_file = f.read().strip()

# Import from muxi_llm to check __version__
import muxi_llm

print(f"Version from muxi_llm/.version file: {version_from_file}")
print(f"Version from muxi_llm.__version__: {muxi_llm.__version__}")

if version_from_file == muxi_llm.__version__:
    print("✅ Versions match! Configuration is correct.")
else:
    print("❌ Versions don't match. Check implementation.")

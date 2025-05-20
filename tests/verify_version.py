#!/usr/bin/env python3
# Script to verify .version reading

import os

# Version from .version file in the onellm package
with open(os.path.join('onellm', '.version'), 'r') as f:
    version_from_file = f.read().strip()

# Import from onellm to check __version__
import onellm

print(f"Version from onellm/.version file: {version_from_file}")
print(f"Version from onellm.__version__: {onellm.__version__}")

if version_from_file == onellm.__version__:
    print("✅ Versions match! Configuration is correct.")
else:
    print("❌ Versions don't match. Check implementation.")

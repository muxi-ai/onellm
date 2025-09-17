#!/usr/bin/env python3
"""
Test OpenAI provider file upload with cross-platform path fix.
"""

import asyncio
from onellm.providers.glm import GLMProvider


async def main():
    provider = GLMProvider()
    resp = await provider.create_chat_completion(
        model="glm-4-plus",
        messages=[{"role": "user", "content": "Quick connectivity test."}],
        max_tokens=16,
    )
    print(resp.choices[0].message["content"])
    print(resp.usage)

asyncio.run(main())

from setuptools import setup, find_packages

setup(
    name="muxi-llm",
    version="0.1.0",
    description="MUXI LLM provides a unified interface for LLM providers using OpenAI format",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "PyYAML>=6.0.0",
        # Optional (but recommended) dependencies
        "tiktoken>=0.3.0; python_version >= '3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "ruff>=0.0.100",
        ],
    },
    python_requires=">=3.10"
)

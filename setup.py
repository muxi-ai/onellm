from setuptools import setup, find_packages

setup(
    name="muxi-llm",
    version="0.1.0",
    description="MUXI LLM provides a unified interface for LLM providers using OpenAI format",
    author="Ran Aroussi",
    author_email="ran@aroussi.com",
    packages=find_packages(),
    install_requires=[
        # Add CLI-specific dependencies here
    ],
    python_requires=">=3.10"
)

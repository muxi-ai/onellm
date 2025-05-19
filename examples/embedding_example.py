#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Unified interface for LLM providers using OpenAI format
# https://github.com/muxi-ai/llm
#
# Copyright (C) 2025 Ran Aroussi
#
# This is free software: You can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License (V3),
# published by the Free Software Foundation (the "License").
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""
# ============================================================================ #
# MUXI-LLM EXAMPLE: Text Embeddings and Semantic Search
# ============================================================================ #
#
# This example demonstrates how to use muxi-llm to generate vector embeddings
# for text and implement semantic search functionality.
# Key features demonstrated:
#
# - Generating embeddings for single or multiple texts
# - Working with embedding vectors and dimensions
# - Computing semantic similarity between texts
# - Implementing a simple semantic search application
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages muxi-llm's support for:
# - Embedding API
# - Vector representation of text
# - Provider-specific embedding models
# - Usage tracking and token counting
#
# RELATED EXAMPLES:
# ----------------
# - chat_completion_example.py: Basic text interactions with LLMs
# - fallback_example.py: Using fallback models for reliability
# - parallel_operation_example.py: Handling multiple requests in parallel
#
# REQUIREMENTS:
# ------------
# - muxi-llm
# - numpy (for vector operations)
# - OpenAI API key with access to embedding models
#
# EXPECTED OUTPUT:
# ---------------
# 1. Embedding vector information for a single text input
# 2. Embedding vector information for multiple text inputs
# 3. A semantic search demonstration showing ranked results by similarity
# ============================================================================ #
"""

import os
import numpy as np
from typing import List

from muxi_llm import Embedding
from muxi_llm.config import set_api_key


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    providing a value between -1 and 1 where 1 means identical direction,
    0 means orthogonal, and -1 means opposite direction.

    Args:
        a: First vector as a list of floats
        b: Second vector as a list of floats

    Returns:
        float: The cosine similarity value between the two vectors
    """
    # Convert to numpy arrays for efficient computation
    a_array = np.array(a)
    b_array = np.array(b)

    # Calculate cosine similarity using the dot product and magnitudes
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))


def get_embeddings(model: str, texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using the specified model.

    This function calls the Embedding API to generate vector representations
    of the provided texts and prints information about the results.

    Args:
        model: The model identifier in format "provider/model"
        texts: List of text strings to generate embeddings for

    Returns:
        List[List[float]]: A list of embedding vectors, one for each input text
    """
    print(f"\n--- Generating Embeddings with {model} ---")

    # Call the Embedding API to generate embeddings for the input texts
    response = Embedding.create(
        model=model,
        input=texts
    )

    # Extract the embeddings from the response object
    embeddings = [data.embedding for data in response.data]

    # Print information about the generated embeddings
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

    # Print token usage information if available in the response
    if response.usage:
        print(f"Token usage: {response.usage.get('total_tokens', 'N/A')} tokens")

    return embeddings


def semantic_search_demo(query_text: str, corpus: List[str], model: str) -> None:
    """
    Demonstrate semantic search using embeddings.

    This function shows how to use embeddings for finding semantically similar
    texts in a corpus based on a query. It computes embeddings for both the
    query and corpus texts, then ranks corpus items by similarity to the query.

    Args:
        query_text: The search query text
        corpus: List of texts to search through
        model: The model identifier to use for generating embeddings

    Returns:
        None: Results are printed to the console
    """
    print(f"\n--- Semantic Search Demo with '{query_text}' ---")

    # Get embeddings for both the query and all corpus texts in a single API call
    all_texts = [query_text] + corpus
    all_embeddings = get_embeddings(model, all_texts)

    # Separate the query embedding from the corpus embeddings
    query_embedding = all_embeddings[0]
    corpus_embeddings = all_embeddings[1:]

    # Calculate similarity scores between the query and each corpus item
    similarities = [
        (text, cosine_similarity(query_embedding, embedding))
        for text, embedding in zip(corpus, corpus_embeddings)
    ]

    # Sort results by similarity score in descending order (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print the ranked results with similarity scores
    print("\nResults (sorted by similarity):")
    for i, (text, score) in enumerate(similarities, 1):
        # Truncate long texts for cleaner display
        display_text = text if len(text) < 70 else text[:67] + "..."
        print(f"{i}. {display_text} (Score: {score:.4f})")


def main() -> None:
    """
    Run the embedding examples.

    This function demonstrates three use cases for embeddings:
    1. Generating an embedding for a single text
    2. Generating embeddings for multiple texts
    3. Using embeddings for semantic search

    The function handles API key setup and runs all the examples sequentially.

    Returns:
        None
    """
    # Set API key from environment variable for authentication
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY is required for this example.")
        return

    # Configure the API key for the OpenAI provider
    set_api_key(openai_api_key, "openai")

    # Specify which embedding model to use
    model = "openai/text-embedding-3-small"

    # Example 1: Generate embedding for a single text
    texts = ["Hello, world!"]
    get_embeddings(model, texts)

    # Example 2: Generate embeddings for multiple texts simultaneously
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process and generate human language.",
        "Embeddings capture semantic meaning in dense vector spaces."
    ]
    get_embeddings(model, texts)

    # Example 3: Demonstrate semantic search using embeddings
    query = "Natural language processing techniques"
    corpus = [
        "Deep learning has revolutionized machine translation.",
        "Quantum computing promises to solve complex problems faster.",
        "Natural language processing models have improved significantly.",
        "Renewable energy sources are becoming more cost-effective.",
        "Neural networks can extract meaning from unstructured text.",
        "Data privacy regulations impact how companies handle information.",
        "Text embeddings allow for semantic similarity calculations.",
        "Cloud computing enables scalable machine learning deployments."
    ]

    # Run the semantic search demonstration
    semantic_search_demo(query, corpus, model)


if __name__ == "__main__":
    main()

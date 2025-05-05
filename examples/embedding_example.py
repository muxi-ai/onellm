"""
Example demonstrating the use of the Embedding module.

This example shows how to use the Embedding class to generate
embeddings from various LLM providers via a unified interface.
"""

import os
import numpy as np
from typing import List

from muxi.llm import Embedding
from muxi.llm.config import set_api_key


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    # Convert to numpy arrays
    a_array = np.array(a)
    b_array = np.array(b)

    # Calculate cosine similarity
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))


def get_embeddings(model: str, texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    print(f"\n--- Generating Embeddings with {model} ---")

    # Call the Embedding API
    response = Embedding.create(
        model=model,
        input=texts
    )

    # Extract the embeddings from the response
    embeddings = [data.embedding for data in response.data]

    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

    # Print usage if available
    if response.usage:
        print(f"Token usage: {response.usage.get('total_tokens', 'N/A')} tokens")

    return embeddings


def semantic_search_demo(query_text: str, corpus: List[str], model: str) -> None:
    """Demonstrate semantic search using embeddings."""
    print(f"\n--- Semantic Search Demo with '{query_text}' ---")

    # Get embeddings for query and corpus
    all_texts = [query_text] + corpus
    all_embeddings = get_embeddings(model, all_texts)

    # The first embedding is our query
    query_embedding = all_embeddings[0]
    corpus_embeddings = all_embeddings[1:]

    # Calculate similarities
    similarities = [
        (text, cosine_similarity(query_embedding, embedding))
        for text, embedding in zip(corpus, corpus_embeddings)
    ]

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\nResults (sorted by similarity):")
    for i, (text, score) in enumerate(similarities, 1):
        # Truncate long texts for display
        display_text = text if len(text) < 70 else text[:67] + "..."
        print(f"{i}. {display_text} (Score: {score:.4f})")


def main() -> None:
    """Run the example."""
    # Set API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY is required for this example.")
        return

    set_api_key(openai_api_key, "openai")

    # Model to use (OpenAI's text-embedding model)
    model = "openai/text-embedding-3-small"

    # Example: Single text embedding
    texts = ["Hello, world!"]
    get_embeddings(model, texts)

    # Example: Multiple text embeddings
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process and generate human language.",
        "Embeddings capture semantic meaning in dense vector spaces."
    ]
    get_embeddings(model, texts)

    # Example: Semantic search
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

    semantic_search_demo(query, corpus, model)


if __name__ == "__main__":
    main()

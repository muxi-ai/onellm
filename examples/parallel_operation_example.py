#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# ============================================================================ #
# MUXI-LLM EXAMPLE: Parallel Operations
# ============================================================================ #
#
# This example demonstrates how to use OneLLM to achieve high throughput by
# processing multiple requests in parallel. This is particularly useful for:
#
# - Batch processing of many prompts/queries
# - Implementing efficient serverless functions or APIs
# - Reducing overall latency when making many independent LLM calls
# - Combining multiple model outputs (ensemble approaches)
#
# CODEBASE RELATIONSHIP:
# ----------------------
# This example leverages OneLLM's support for:
# - Asynchronous API (acreate methods)
# - Thread safety for concurrent usage
# - Provider-agnostic interface across multiple models
# - Usage data for performance tracking
#
# RELATED EXAMPLES:
# ----------------
# - chat_completion_example.py: Basic usage of the chat completion API
# - fallback_example.py: Using fallbacks for reliability
# - retry_example.py: Automatic retries for transient failures
#
# REQUIREMENTS:
# ------------
# - OneLLM
# - multitasking
# ============================================================================ #
"""

import os
import time
import asyncio
import multitasking
import concurrent.futures
from typing import List, Dict, Any

import onellm
from onellm import ChatCompletion

# Configure API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
onellm.api_key = api_key

# Sample data for processing
TOPICS = [
    "Artificial Intelligence",
    "Quantum Computing",
    "Sustainable Energy",
    "Space Exploration",
    "Biotechnology",
    "Neuroscience",
    "Robotics",
    "Blockchain",
    "3D Printing",
    "Virtual Reality"
]

# Example prompts that will be processed in parallel
QUERIES = [
    f"Explain {topic} in simple terms. Keep it under 100 words."
    for topic in TOPICS
]


# ========================== Async Approach ==========================

async def process_single_query_async(
    query: str,
    model: str = "openai/gpt-3.5-turbo"
) -> Dict[str, Any]:
    """
    Process a single query asynchronously.

    Args:
        query: The text prompt to send to the LLM
        model: The model identifier to use

    Returns:
        Dict containing the query and response
    """
    print(f"Processing: {query[:30]}...")

    # Create messages for chat completion
    messages = [{"role": "user", "content": query}]

    # Make the async API call
    response = await ChatCompletion.acreate(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    # Extract the response text
    response_text = response.choices[0].message["content"]

    return {
        "query": query,
        "response": response_text,
        "tokens": response.usage.total_tokens
    }


async def process_all_queries_async(
    queries: List[str],
    model: str = "openai/gpt-3.5-turbo"
) -> List[Dict[str, Any]]:
    """
    Process multiple queries concurrently using asyncio.gather.

    This function demonstrates true concurrency for I/O-bound operations
    by using asyncio.gather to send multiple requests at once.

    Args:
        queries: List of queries to process
        model: Model identifier to use

    Returns:
        List of results from all queries
    """
    # Create tasks for all queries
    tasks = [process_single_query_async(query, model) for query in queries]

    # Run all tasks concurrently and wait for all to complete
    results = await asyncio.gather(*tasks)

    return results


# ========================== Thread Pool Approach ==========================

@multitasking.task  # Decorator from multitasking library
def process_in_thread(
    query: str,
    model: str,
    results: List[Dict[str, Any]],
    index: int
) -> None:
    """
    Process a single query in a separate thread.

    This function runs in a thread and stores the result in the shared results list.

    Args:
        query: The query to process
        model: Model identifier to use
        results: Shared list where results will be stored
        index: Position in the results list for this query
    """
    print(f"Thread processing: {query[:30]}...")

    # Create messages for chat completion
    messages = [{"role": "user", "content": query}]

    try:
        # Use the synchronous API since we're already in a separate thread
        response = ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        # Extract the response text
        response_text = response.choices[0].message["content"]

        # Store result at the specified index
        results[index] = {
            "query": query,
            "response": response_text,
            "tokens": response.usage.total_tokens
        }
    except Exception as e:
        results[index] = {"query": query, "error": str(e)}


def process_all_queries_threaded(
    queries: List[str],
    model: str = "openai/gpt-3.5-turbo"
) -> List[Dict[str, Any]]:
    """
    Process multiple queries concurrently using thread pool via multitasking.

    This function uses the multitasking library to distribute work across threads
    for improved throughput when dealing with multiple independent LLM requests.

    Args:
        queries: List of queries to process
        model: Model identifier to use

    Returns:
        List of results from all queries
    """
    # Set up multitasking to use a thread pool
    multitasking.set_max_threads(min(10, len(queries)))

    # Initialize results list with None placeholders
    results = [None] * len(queries)

    # Start processing each query in its own thread
    for i, query in enumerate(queries):
        process_in_thread(query, model, results, i)

    # Wait for all threads to complete
    multitasking.wait_for_tasks()

    return results


# ========================== Post-Processing in Parallel ==========================

def analyze_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform analysis on a single response (simulated CPU-bound task).

    Args:
        result: The response data to analyze

    Returns:
        Dictionary with the original result and added analysis
    """
    # Simulate CPU-bound work
    time.sleep(0.1)

    # Add analysis to the result
    word_count = len(result["response"].split())
    sentiment = "positive" if "amazing" in result["response"].lower() else "neutral"

    return {
        **result,
        "analysis": {
            "word_count": word_count,
            "sentiment": sentiment,
            "efficiency": word_count / (result["tokens"] or 1)
        }
    }


def parallel_post_process(
    results: List[Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Process results in parallel using thread pool.

    Uses concurrent.futures for CPU-bound tasks parallelized across multiple cores.

    Args:
        results: List of query results to analyze
        max_workers: Maximum number of worker threads

    Returns:
        Processed results with analysis added
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the analyze_response function over all results in parallel
        analyzed_results = list(executor.map(analyze_response, results))

    return analyzed_results


# ========================== Mixed Approach (Advanced) ==========================

async def combined_parallel_processing(
    queries: List[str],
    model: str = "openai/gpt-3.5-turbo",
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Combine async API calls with parallel post-processing.

    This advanced function demonstrates how to combine asyncio for I/O-bound
    tasks (API calls) with thread pools for CPU-bound tasks (post-processing).

    Args:
        queries: List of queries to process
        model: Model identifier to use
        max_workers: Maximum number of worker threads for post-processing

    Returns:
        Fully processed results with analysis
    """
    # First, process all queries concurrently using asyncio
    raw_results = await process_all_queries_async(queries, model)

    # Then, perform CPU-bound post-processing in a thread pool
    # Create a loop to run the post-processing in an executor
    loop = asyncio.get_event_loop()
    processed_results = await loop.run_in_executor(
        None,  # Use default executor
        lambda: parallel_post_process(raw_results, max_workers)
    )

    return processed_results


# ========================== Main Example Code ==========================

async def run_async_example():
    """Run the async example and measure performance."""
    print("\n=== Running Async Example ===")
    start_time = time.time()

    results = await process_all_queries_async(QUERIES)

    duration = time.time() - start_time
    print(f"Processed {len(results)} queries in {duration:.2f} seconds")
    print(f"Average time per query: {duration/len(results):.2f} seconds")

    # Display first result as sample
    print("\nSample response:")
    print(f"Query: {results[0]['query']}")
    print(f"Response: {results[0]['response']}")

    return results


def run_threaded_example():
    """Run the thread pool example and measure performance."""
    print("\n=== Running Thread Pool Example ===")
    start_time = time.time()

    results = process_all_queries_threaded(QUERIES)

    duration = time.time() - start_time
    print(f"Processed {len(results)} queries in {duration:.2f} seconds")
    print(f"Average time per query: {duration/len(results):.2f} seconds")

    # Display first result as sample
    print("\nSample response:")
    print(f"Query: {results[0]['query']}")
    print(f"Response: {results[0]['response']}")

    return results


async def run_combined_example():
    """Run the combined approach example and measure performance."""
    print("\n=== Running Combined Approach Example ===")
    start_time = time.time()

    results = await combined_parallel_processing(QUERIES)

    duration = time.time() - start_time
    print(f"Processed {len(results)} queries in {duration:.2f} seconds")
    print(f"Average time per query: {duration/len(results):.2f} seconds")

    # Display first result as sample
    print("\nSample response with analysis:")
    print(f"Query: {results[0]['query']}")
    print(f"Response: {results[0]['response']}")
    print(f"Analysis: {results[0]['analysis']}")

    return results


async def compare_approaches():
    """Compare all three approaches."""
    print("\n=== Comparing All Approaches ===")

    # Sequential processing for baseline
    print("\nSequential Processing (Baseline):")
    start_time = time.time()
    sequential_results = []
    for query in QUERIES[:3]:  # Use fewer queries for baseline
        result = await process_single_query_async(query)
        sequential_results.append(result)
    sequential_duration = time.time() - start_time
    print(f"Processed 3 queries sequentially in {sequential_duration:.2f} seconds")
    print(f"Average time per query: {sequential_duration/3:.2f} seconds")

    # Async approach
    print("\nAsynchronous Approach:")
    start_time = time.time()
    async_results = await process_all_queries_async(QUERIES)
    async_duration = time.time() - start_time
    print(f"Processed {len(async_results)} queries in {async_duration:.2f} seconds")
    print(f"Average time per query: {async_duration/len(async_results):.2f} seconds")

    speedup_async = (sequential_duration/3)/(async_duration/len(async_results))
    print(f"Speedup vs sequential: {speedup_async:.2f}x")

    # Thread pool approach
    print("\nThread Pool Approach:")
    start_time = time.time()
    threaded_results = process_all_queries_threaded(QUERIES)
    threaded_duration = time.time() - start_time
    print(f"Processed {len(threaded_results)} queries in {threaded_duration:.2f} seconds")
    print(f"Average time per query: {threaded_duration/len(threaded_results):.2f} seconds")

    speedup_thread = (sequential_duration/3)/(threaded_duration/len(threaded_results))
    print(f"Speedup vs sequential: {speedup_thread:.2f}x")

    # Combined approach
    print("\nCombined Approach:")
    start_time = time.time()
    combined_results = await combined_parallel_processing(QUERIES)
    combined_duration = time.time() - start_time
    print(f"Processed {len(combined_results)} queries in {combined_duration:.2f} seconds")
    print(f"Average time per query: {combined_duration/len(combined_results):.2f} seconds")

    speedup_combined = (sequential_duration/3)/(combined_duration/len(combined_results))
    print(f"Speedup vs sequential: {speedup_combined:.2f}x")

    print("\nConclusion:")
    print("The best approach depends on your use case:")
    print("- Async: best for pure I/O-bound workloads")
    print("- Thread pool: simple to use for mixed workloads")
    print("- Combined: most control but more complex")


if __name__ == "__main__":
    # Run the examples
    print("Parallel Operation Example with OneLLM")
    print("======================================")
    print("This example demonstrates different approaches")
    print("to process multiple LLM requests in parallel.")

    # Select which example to run (uncomment one)
    asyncio.run(run_async_example())
    # run_threaded_example()
    # asyncio.run(run_combined_example())
    # asyncio.run(compare_approaches())  # This runs all approaches and compares them

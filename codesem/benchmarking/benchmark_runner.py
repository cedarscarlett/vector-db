# codesem/benchmarking/benchmark_runner.py

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from codesem.embeddings.openai_embedder import OpenAIEmbedder
from codesem.storage.vector_repository import VectorRepository
from codesem.retrieval.hybrid_ranker import tokenize_query


# ============================================================
# Utilities
# ============================================================

def _run_grep(repo_path: str, tokens: List[str]) -> List[str]:
    """
    Run simple grep baseline:
    - case-insensitive
    - OR search across tokens in single subprocess call
    """
    if not tokens:
        return []

    matched_files = set()

    pattern = "|".join(tokens)

    try:
        result = subprocess.run(
            ["grep", "-R", "-l", "-i", "-E", pattern, repo_path],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            for line in result.stdout.splitlines():
                matched_files.add(line.strip())
    except Exception:
        pass

    return list(matched_files)


# ============================================================
# Benchmark
# ============================================================


def run_benchmark(
    queries_path: str,
    repo_path: Optional[str],
    top_k: int = 5,
) -> Dict:
    """
    Run semantic vs grep benchmark.

    Returns:
    {
        "per_query": [...],
        "overall": {...}
    }
    """
    start_time = time.time()

    queries_file = Path(queries_path)
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    with queries_file.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    # Construct dependencies once for benchmark lifecycle
    embedder = OpenAIEmbedder()
    per_query_results = []

    total_vector_recall = 0.0
    total_grep_hits = 0
    total_queries = len(queries)

    with VectorRepository() as repo:
        for item in queries:
            try:
                query = item["query"]
            except KeyError:
                raise ValueError(
                    f"Malformed query entry (missing 'query'): {item}"
                )

            expected_files = set(item.get("expected_files", []))

            # --- Semantic search ---
            query_embedding = embedder.embed_text(query)
            vector_hits = repo.search(query_embedding=query_embedding, top_k=top_k)
            returned_files = {
                Path(hit.file_path).name for hit in vector_hits
            }

            # NOTE: Matching by filename only assumes unique filenames in demo repo.
            hits = len(expected_files.intersection(returned_files))
            recall = hits / len(expected_files) if expected_files else 0.0

            total_vector_recall += recall

            # --- Grep baseline ---
            grep_hit = 0
            if repo_path:
                tokens = tokenize_query(query)
                grep_files = _run_grep(repo_path, tokens)
                grep_file_names = {Path(f).name for f in grep_files}

                if expected_files.intersection(grep_file_names):
                    grep_hit = 1

            total_grep_hits += grep_hit

            per_query_results.append(
                {
                    "query": query,
                    "vector_recall_at_k": recall,
                    "grep_hit": grep_hit,
                    "expected_files": list(expected_files),
                    "returned_files": list(returned_files),
                }
            )

    overall = {
        "vector_avg_recall_at_k": (
            total_vector_recall / total_queries if total_queries else 0.0
        ),
        "grep_hit_rate": (
            total_grep_hits / total_queries if total_queries else 0.0
        ),
        "elapsed_sec": time.time() - start_time,
    }

    return {
        "per_query": per_query_results,
        "overall": overall,
    }

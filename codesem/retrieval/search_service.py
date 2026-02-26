# codesem/retrieval/search_service.py

from __future__ import annotations

from typing import Dict, List

from codesem.config.settings import get_settings
from codesem.embeddings.openai_embedder import OpenAIEmbedder
from codesem.retrieval.hybrid_ranker import HybridCandidate, rank_hybrid
from codesem.storage.vector_repository import VectorRepository


# ============================================================
# Search
# ============================================================


def search_code(
    query: str,
    top_k: int = 5,
    hybrid: bool = False,
) -> List[Dict]:
    """
    Perform semantic (or hybrid) search.

    Returns list of dicts compatible with CLI SearchResultItem contract:
    {
        file_path: str,
        start_line: int,
        end_line: int,
        score: float,
        preview: str
    }
    """
    settings = get_settings()

    embedder = OpenAIEmbedder()
    query_embedding = embedder.embed_text(query)

    with VectorRepository() as repo:
        # Over-retrieve only when hybrid mode is enabled
        candidate_k = top_k if not hybrid else top_k * 3
        vector_hits = repo.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

    if not vector_hits:
        return []

    # Vector-only mode
    vector_scores = [hit.similarity for hit in vector_hits]

    if not hybrid:
        # Repository already returns ordered by similarity (descending via cosine)
        sorted_hits = vector_hits[:top_k]

        return [
            {
                "file_path": h.file_path,
                "start_line": h.start_line,
                "end_line": h.end_line,
                "score": h.similarity,
                "preview": h.content,
            }
            for h in sorted_hits
        ]

    # Hybrid mode
    candidates = [
        HybridCandidate(
            file_path=hit.file_path,
            start_line=hit.start_line,
            end_line=hit.end_line,
            content=hit.content,
            vector_score=hit.similarity,
        )
        for hit in vector_hits
    ]

    ranked = rank_hybrid(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

    return [
        {
            "file_path": r.file_path,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "score": r.final_score,
            "preview": r.content,
        }
        for r in ranked
    ]

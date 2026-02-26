# tests/test_search.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from codesem.retrieval.hybrid_ranker import (
    HybridCandidate,
    rank_hybrid,
    tokenize_query,
    keyword_score,
    min_max_normalize,
)


# ============================================================
# Unit tests for hybrid ranking logic (pure, no DB/OpenAI)
# ============================================================


def test_tokenize_query_removes_stopwords_and_punctuation():
    query = "Where do we handle authentication errors?"
    tokens = tokenize_query(query)

    # Should remove stopwords like "where", "do", "we"
    assert "where" not in tokens
    assert "do" not in tokens
    assert "we" not in tokens

    # Should keep meaningful words
    assert "handle" in tokens
    assert "authentication" in tokens
    assert "errors" in tokens


def test_tokenize_query_empty_input():
    tokens = tokenize_query("")
    assert tokens == []


def test_keyword_score_substring_behavior():
    content = "This module handles logging and logger configuration."
    tokens = ["log"]

    score = keyword_score(content, tokens)

    # "log" should match "logging" and "logger"
    assert score >= 2


def test_min_max_normalize_all_zero():
    values = [0.0, 0.0, 0.0]
    normalized = min_max_normalize(values)

    assert normalized == [0.0, 0.0, 0.0]


def test_min_max_normalize_equal_nonzero():
    values = [5.0, 5.0, 5.0]
    normalized = min_max_normalize(values)

    assert normalized == [1.0, 1.0, 1.0]


def test_min_max_normalize_single_value():
    values = [3.0]
    normalized = min_max_normalize(values)
    assert normalized == [1.0]


def test_rank_hybrid_prefers_stronger_vector_when_no_keyword_signal():
    candidates = [
        HybridCandidate(
            file_path="a.py",
            start_line=1,
            end_line=10,
            content="foo bar baz",
            vector_score=0.9,
        ),
        HybridCandidate(
            file_path="b.py",
            start_line=1,
            end_line=10,
            content="foo bar baz",
            vector_score=0.5,
        ),
    ]

    results = rank_hybrid(
        query="completely unrelated terms",
        candidates=candidates,
        top_k=2,
    )

    # With no keyword matches, ranking should follow vector score
    assert results[0].file_path == "a.py"
    assert results[1].file_path == "b.py"


def test_rank_hybrid_keyword_can_change_order():
    candidates = [
        HybridCandidate(
            file_path="a.py",
            start_line=1,
            end_line=10,
            content="retry retry retry logic",
            vector_score=0.6,
        ),
        HybridCandidate(
            file_path="b.py",
            start_line=1,
            end_line=10,
            content="no relevant terms here",
            vector_score=0.5,
        ),
    ]

    results = rank_hybrid(
        query="retry behavior",
        candidates=candidates,
        top_k=2,
    )

    # With keyword signal, a.py should outrank b.py
    assert results[0].file_path == "a.py"


def test_rank_hybrid_empty_candidates():
    results = rank_hybrid(query="anything", candidates=[], top_k=5)
    assert results == []


def test_rank_hybrid_top_k_truncation():
    candidates = [
        HybridCandidate(
            file_path=f"{i}.py",
            start_line=1,
            end_line=10,
            content="content",
            vector_score=float(i),
        )
        for i in range(5)
    ]

    results = rank_hybrid(
        query="content",
        candidates=candidates,
        top_k=3,
    )

    assert len(results) == 3

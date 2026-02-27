"""Feature extraction for learning-to-rank."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Set, Sequence, Tuple

import numpy as np

from codesem.retrieval.hybrid_ranker import (
    HybridCandidate,
    keyword_score,
    min_max_normalize,
    tokenize_query,
)

FEATURE_NAMES = ["vector_score_norm", "keyword_score_norm", "log_content_length"]


def build_feature_vector(
    vector_score_norm: float,
    keyword_score_norm: float,
    content: str,
) -> np.ndarray:
    """Build a 3-element feature vector from pre-normalized scores and content."""
    log_content_length = math.log1p(len(content))
    return np.array(
        [vector_score_norm, keyword_score_norm, log_content_length],
        dtype=np.float64,
    )


def extract_dataset(
    query: str,
    candidates: List[HybridCandidate],
    expected_files: Set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y from candidates.

    Labels: 1 if Path(candidate.file_path).name is in expected_files, else 0.
    Scores are min-max normalized across candidates.
    """
    if not candidates:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    tokens = tokenize_query(query)

    vector_scores = [c.vector_score for c in candidates]
    kw_scores_raw = [keyword_score(c.content, tokens) for c in candidates]

    vector_norms = min_max_normalize(vector_scores)
    kw_norms = min_max_normalize(kw_scores_raw)

    rows: list[np.ndarray] = []
    labels: list[float] = []

    for i, candidate in enumerate(candidates):
        fv = build_feature_vector(vector_norms[i], kw_norms[i], candidate.content)
        rows.append(fv)

        fname = Path(candidate.file_path).name
        labels.append(1.0 if fname in expected_files else 0.0)

    X = np.vstack(rows)
    y = np.array(labels, dtype=np.float64)
    return X, y

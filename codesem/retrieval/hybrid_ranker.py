"""Hybrid ranking combining vector similarity and keyword scoring."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from codesem.config.settings import get_settings

_log = logging.getLogger(__name__)

# Lazy-loaded model: None = not attempted, False = failed/missing, instance = loaded
_MODEL: Union[Any, None, bool] = None


# ============================================================
# Data Model
# ============================================================


@dataclass(frozen=True)
class HybridCandidate:
    """
    Represents a candidate result for hybrid ranking.
    """
    file_path: str
    start_line: int
    end_line: int
    content: str
    vector_score: float


@dataclass(frozen=True)
class HybridResult:
    file_path: str
    start_line: int
    end_line: int
    content: str
    final_score: float
    vector_score: float
    keyword_score: float


# ============================================================
# Tokenization
# ============================================================


_STOP_WORDS = {
    "the",
    "is",
    "are",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "how",
    "what",
    "where",
    "why",
    "when",
    "do",
    "does",
    "did",
    "we",
    "our",
    "this",
    "that",
}


def tokenize_query(query: str) -> List[str]:
    """
    Lowercase, strip punctuation, remove stop words,
    remove short tokens (< 3 chars).
    """
    query = query.lower()
    query = re.sub(r"[^\w\s]", " ", query)
    tokens = query.split()

    return [
        t
        for t in tokens
        if t not in _STOP_WORDS and len(t) >= 3
    ]


# ============================================================
# Scoring
# ============================================================


def keyword_score(content: str, tokens: Sequence[str]) -> float:
    """
    Substring-based keyword scoring.

    NOTE:
    This intentionally counts substring matches
    (e.g., "log" matches "logging", "logger", etc.).
    """
    if not tokens:
        return 0.0

    content_lower = content.lower()
    total_hits = 0

    for token in tokens:
        total_hits += content_lower.count(token)

    return float(total_hits)


def min_max_normalize(values: Sequence[float]) -> List[float]:
    """
    Normalize scores to [0, 1].

    If all values are equal:
    - If zero, return all zeros
    - Otherwise return all ones
    """
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)

    if math.isclose(v_min, v_max):
        if math.isclose(v_max, 0.0):
            return [0.0 for _ in values]
        return [1.0 for _ in values]

    return [(v - v_min) / (v_max - v_min) for v in values]


# ============================================================
# Hybrid Ranking
# ============================================================


def _get_model() -> Any:
    """Lazy-load the trained LearningRanker model, if available."""
    global _MODEL
    if _MODEL is None:
        model_path = Path.cwd() / ".codesem_ranker.joblib"
        try:
            from codesem.ml.learning_ranker import LearningRanker
            _MODEL = LearningRanker.load(str(model_path))
            _log.info("Loaded LTR model from %s", model_path)
        except Exception:
            _log.debug("No LTR model loaded (path: %s), using static weights", model_path)
            _MODEL = False
    return _MODEL if _MODEL is not False else None


def rank_hybrid(
    query: str,
    candidates: Iterable[HybridCandidate],
    top_k: int,
) -> List[HybridResult]:
    """
    Perform hybrid ranking over vector candidates.

    Uses trained ML model if available, otherwise falls back to static weights.
    """
    settings = get_settings()

    candidate_list = list(candidates)
    if not candidate_list:
        return []

    tokens = tokenize_query(query)

    vector_scores = [c.vector_score for c in candidate_list]
    keyword_scores_raw = [
        keyword_score(c.content, tokens)
        for c in candidate_list
    ]

    vector_scores_norm = min_max_normalize(vector_scores)
    keyword_scores_norm = min_max_normalize(keyword_scores_raw)

    # Try ML-based ranking
    model = _get_model()
    if model is not None:
        try:
            from codesem.ml.feature_extractor import build_feature_vector
            import numpy as np

            rows = [
                build_feature_vector(vector_scores_norm[i], keyword_scores_norm[i], c.content)
                for i, c in enumerate(candidate_list)
            ]
            X = np.vstack(rows)
            probas = model.predict_proba(X)

            scored = sorted(
                zip(candidate_list, probas, keyword_scores_norm),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            return [
                HybridResult(
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content,
                    final_score=float(prob),
                    vector_score=c.vector_score,
                    keyword_score=kw,
                )
                for c, prob, kw in scored
            ]
        except Exception:
            _log.warning("LTR prediction failed, falling back to static weights", exc_info=True)

    # Static weighted fallback
    combined: List[Tuple[HybridCandidate, float, float]] = []

    for i, candidate in enumerate(candidate_list):
        final_score = (
            settings.hybrid_weight_vector * vector_scores_norm[i]
            + settings.hybrid_weight_keyword * keyword_scores_norm[i]
        )

        combined.append(
            (
                candidate,
                final_score,
                keyword_scores_norm[i],
            )
        )

    combined_sorted = sorted(
        combined,
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results: List[HybridResult] = []

    for candidate, final_score, kw_score_norm in combined_sorted:
        results.append(
            HybridResult(
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                content=candidate.content,
                final_score=final_score,
                vector_score=candidate.vector_score,
                keyword_score=kw_score_norm,
            )
        )

    return results

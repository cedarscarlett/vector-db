"""Tests for ML feature extraction — pure synthetic data, no DB or OpenAI."""

from __future__ import annotations

import numpy as np
import pytest

from codesem.ml.feature_extractor import build_feature_vector, extract_dataset
from codesem.retrieval.hybrid_ranker import HybridCandidate


# ============================================================
# build_feature_vector
# ============================================================


def test_feature_vector_shape():
    fv = build_feature_vector(0.8, 0.3, "def hello(): pass")
    assert fv.shape == (3,)


def test_feature_vector_values():
    content = "some code content"
    fv = build_feature_vector(0.5, 0.7, content)
    assert fv[0] == pytest.approx(0.5)
    assert fv[1] == pytest.approx(0.7)
    assert fv[2] == pytest.approx(np.log1p(len(content)))


def test_feature_vector_empty_content():
    fv = build_feature_vector(1.0, 0.0, "")
    assert fv[2] == pytest.approx(0.0)  # log1p(0) == 0


# ============================================================
# extract_dataset
# ============================================================


def _make_candidates() -> list[HybridCandidate]:
    return [
        HybridCandidate(
            file_path="/repo/src/target.py",
            start_line=1,
            end_line=10,
            content="def search(): embedding = embed(query)",
            vector_score=0.95,
        ),
        HybridCandidate(
            file_path="/repo/src/other.py",
            start_line=1,
            end_line=5,
            content="import os\nprint('hello')",
            vector_score=0.60,
        ),
        HybridCandidate(
            file_path="/repo/src/another.py",
            start_line=1,
            end_line=8,
            content="class Config: pass",
            vector_score=0.40,
        ),
    ]


def test_extract_dataset_shape():
    candidates = _make_candidates()
    X, y = extract_dataset("embedding search", candidates, {"target.py"})
    assert X.shape == (3, 3)
    assert y.shape == (3,)


def test_extract_dataset_labels():
    candidates = _make_candidates()
    X, y = extract_dataset("search query", candidates, {"target.py"})
    # First candidate matches target.py
    assert y[0] == 1.0
    assert y[1] == 0.0
    assert y[2] == 0.0


def test_extract_dataset_multiple_expected():
    candidates = _make_candidates()
    X, y = extract_dataset("query", candidates, {"target.py", "other.py"})
    assert y[0] == 1.0
    assert y[1] == 1.0
    assert y[2] == 0.0


def test_extract_dataset_normalization():
    candidates = _make_candidates()
    X, y = extract_dataset("search", candidates, {"target.py"})
    # Vector scores are [0.95, 0.60, 0.40] -> normalized to [1.0, ~0.36, 0.0]
    assert X[0, 0] == pytest.approx(1.0)
    assert X[2, 0] == pytest.approx(0.0)
    # All values in [0, 1] for first two features
    assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 1.0)
    assert np.all(X[:, 1] >= 0.0) and np.all(X[:, 1] <= 1.0)


def test_extract_dataset_empty_candidates():
    X, y = extract_dataset("query", [], {"target.py"})
    assert X.shape == (0, 3)
    assert y.shape == (0,)

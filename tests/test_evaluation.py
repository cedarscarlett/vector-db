"""Tests for evaluation utilities — pure synthetic data, no DB or OpenAI."""

from __future__ import annotations

import numpy as np
import pytest

from codesem.ml.evaluation import auc_score, cross_validate, precision_at_k


# ============================================================
# precision_at_k
# ============================================================


def test_precision_at_k_perfect():
    y_true = [1, 1, 0, 0, 0]
    y_scores = [0.9, 0.8, 0.3, 0.2, 0.1]
    assert precision_at_k(y_true, y_scores, k=2) == pytest.approx(1.0)


def test_precision_at_k_half():
    y_true = [1, 0, 1, 0]
    y_scores = [0.9, 0.8, 0.3, 0.2]
    # Top 2 by score: indices 0 (true=1), 1 (true=0) -> precision = 0.5
    assert precision_at_k(y_true, y_scores, k=2) == pytest.approx(0.5)


def test_precision_at_k_zero():
    y_true = [0, 0, 1, 1]
    y_scores = [0.9, 0.8, 0.3, 0.2]
    assert precision_at_k(y_true, y_scores, k=2) == pytest.approx(0.0)


def test_precision_at_k_zero_k():
    assert precision_at_k([1, 0], [0.9, 0.1], k=0) == pytest.approx(0.0)


# ============================================================
# auc_score
# ============================================================


def test_auc_perfect_separation():
    y_true = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    assert auc_score(y_true, y_scores) == pytest.approx(1.0)


def test_auc_single_class_returns_zero():
    y_true = [0, 0, 0]
    y_scores = [0.5, 0.3, 0.1]
    assert auc_score(y_true, y_scores) == pytest.approx(0.0)


def test_auc_random_baseline():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=1000).tolist()
    y_scores = rng.rand(1000).tolist()
    score = auc_score(y_true, y_scores)
    # Random predictions should give AUC near 0.5
    assert 0.4 < score < 0.6


# ============================================================
# cross_validate — fold selection
# ============================================================


def _make_dataset(n_positive: int, n_negative: int):
    rng = np.random.RandomState(42)
    X_pos = rng.uniform(0.6, 1.0, size=(n_positive, 3))
    X_neg = rng.uniform(0.0, 0.4, size=(n_negative, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * n_positive + [0.0] * n_negative)
    return X, y


def test_cv_5_folds_for_50_plus_positives():
    X, y = _make_dataset(60, 60)
    result = cross_validate(X, y)
    assert result["n_folds"] == 5
    assert "mean_auc" in result
    assert "std_auc" in result


def test_cv_3_folds_for_20_to_49_positives():
    X, y = _make_dataset(30, 30)
    result = cross_validate(X, y)
    assert result["n_folds"] == 3


def test_cv_2_folds_for_under_20_positives():
    X, y = _make_dataset(10, 30)
    result = cross_validate(X, y)
    assert result["n_folds"] == 2


def test_cv_returns_high_auc_on_separable_data():
    X, y = _make_dataset(50, 50)
    result = cross_validate(X, y)
    assert result["mean_auc"] > 0.9

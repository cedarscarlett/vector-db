"""Tests for LearningRanker — pure synthetic data, no DB or OpenAI."""

from __future__ import annotations

import numpy as np
import pytest

from codesem.ml.learning_ranker import LearningRanker
from codesem.ml.evaluation import auc_score


def _separable_dataset(n: int = 200, seed: int = 42):
    """Create a linearly separable synthetic dataset with 3 features."""
    rng = np.random.RandomState(seed)
    X_pos = rng.uniform(0.6, 1.0, size=(n // 2, 3))
    X_neg = rng.uniform(0.0, 0.4, size=(n // 2, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
    return X, y


# ============================================================
# Training
# ============================================================


def test_train_returns_metadata():
    X, y = _separable_dataset()
    ranker = LearningRanker()
    meta = ranker.train(X, y)

    assert meta["n_features"] == 3
    assert meta["n_samples"] == 200
    assert meta["n_positive"] == 100
    assert len(meta["coefficients"]) == 3
    assert "intercept" in meta
    assert "training_date" in meta
    assert meta["feature_names"] == [
        "vector_score_norm",
        "keyword_score_norm",
        "log_content_length",
    ]


def test_train_high_auc_on_separable_data():
    X, y = _separable_dataset()
    ranker = LearningRanker()
    ranker.train(X, y)

    probas = ranker.predict_proba(X)
    score = auc_score(y, probas)
    assert score > 0.9


# ============================================================
# Predict
# ============================================================


def test_predict_proba_shape():
    X, y = _separable_dataset()
    ranker = LearningRanker()
    ranker.train(X, y)

    probas = ranker.predict_proba(X)
    assert probas.shape == (200,)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)


def test_predict_proba_feature_mismatch():
    X, y = _separable_dataset()
    ranker = LearningRanker()
    ranker.train(X, y)

    X_wrong = np.random.rand(10, 5)
    with pytest.raises(ValueError, match="Feature count mismatch"):
        ranker.predict_proba(X_wrong)


def test_predict_before_train_raises():
    ranker = LearningRanker()
    X = np.random.rand(5, 3)
    with pytest.raises(RuntimeError):
        ranker.predict_proba(X)


# ============================================================
# Save / Load
# ============================================================


def test_save_load_preserves_predictions(tmp_path):
    X, y = _separable_dataset()
    ranker = LearningRanker()
    meta = ranker.train(X, y)

    model_path = str(tmp_path / "model.joblib")
    ranker.save(model_path, meta)

    loaded = LearningRanker.load(model_path)
    original_probas = ranker.predict_proba(X)
    loaded_probas = loaded.predict_proba(X)

    np.testing.assert_allclose(original_probas, loaded_probas)


def test_load_validates_feature_mismatch(tmp_path):
    X, y = _separable_dataset()
    ranker = LearningRanker()
    meta = ranker.train(X, y)

    # Corrupt metadata
    meta["n_features"] = 5
    model_path = str(tmp_path / "bad_model.joblib")
    ranker.save(model_path, meta)

    with pytest.raises(ValueError, match="inconsistency"):
        LearningRanker.load(model_path)

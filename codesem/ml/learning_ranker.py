"""Supervised learning-to-rank using logistic regression."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from codesem.ml.feature_extractor import FEATURE_NAMES


class LearningRanker:
    """Logistic regression ranker for hybrid search results."""

    def __init__(self) -> None:
        self._model: Optional[LogisticRegression] = None
        self._metadata: Dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train logistic regression on feature matrix X and labels y."""
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
        )
        model.fit(X, y)
        self._model = model

        self._metadata = {
            "feature_names": list(FEATURE_NAMES),
            "n_features": X.shape[1],
            "n_samples": int(X.shape[0]),
            "n_positive": int(y.sum()),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "training_date": datetime.now(timezone.utc).isoformat(),
        }
        return dict(self._metadata)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class for each row in X."""
        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")

        expected = self._metadata.get("n_features", len(FEATURE_NAMES))
        if X.shape[1] != expected:
            raise ValueError(
                f"Feature count mismatch: expected {expected}, got {X.shape[1]}"
            )

        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Dict[str, Any]) -> None:
        """Save model to joblib and metadata to .meta.json."""
        if self._model is None:
            raise RuntimeError("No model to save.")

        joblib.dump(self._model, path)

        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LearningRanker":
        """Load a saved model and its metadata."""
        instance = cls()
        instance._model = joblib.load(path)

        meta_path = path + ".meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            instance._metadata = json.load(f)

        # Validate feature count consistency
        expected = instance._metadata.get("n_features", len(FEATURE_NAMES))
        actual_coefs = len(instance._metadata.get("coefficients", []))
        if actual_coefs and actual_coefs != expected:
            raise ValueError(
                f"Metadata inconsistency: n_features={expected} but "
                f"coefficients length={actual_coefs}"
            )

        return instance

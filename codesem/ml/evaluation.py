"""Evaluation utilities for learning-to-rank."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def precision_at_k(
    y_true: Sequence[float],
    y_scores: Sequence[float],
    k: int,
) -> float:
    """Precision among the top-k scored items."""
    if k <= 0:
        return 0.0

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    top_k_idx = np.argsort(y_scores_arr)[::-1][:k]
    return float(y_true_arr[top_k_idx].sum() / k)


def auc_score(
    y_true: Sequence[float],
    y_scores: Sequence[float],
) -> float:
    """ROC AUC score. Returns 0.0 if only one class present."""
    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return 0.0
    return float(roc_auc_score(y_true_arr, y_scores))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Stratified cross-validation with adaptive fold count.

    Fold selection:
    - positives >= 50 -> 5 folds
    - positives 20-49 -> 3 folds
    - positives < 20  -> 2 folds
    """
    n_positive = int(y.sum())

    # Guard: too few positives for meaningful stratified CV
    if n_positive < 4:
        return {
            "mean_auc": None,
            "std_auc": None,
            "n_folds": 0,
            "warning": "Too few positive samples for cross-validation.",
        }

    if n_positive >= 50:
        n_folds = 5
    elif n_positive >= 20:
        n_folds = 3
    else:
        n_folds = 2

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        score = auc_score(y_test, y_prob)
        auc_scores.append(score)

    return {
        "mean_auc": float(np.mean(auc_scores)),
        "std_auc": float(np.std(auc_scores)),
        "n_folds": n_folds,
    }

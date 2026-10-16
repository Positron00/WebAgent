"""Drift scoring: pure functions shared by the drift store (OBS-007).

Each function compares a reference sample with a current one and returns ``(score, drift)``:

- ``ks_score`` (numbers) and ``chi2_score`` (categories): score ``1 - p``; drift when p < 0.05,
  i.e. score > 0.95.
- ``centroid_score`` (embedding centroids): score ``1 - cosine similarity``; drift when the
  similarity is below 0.3, i.e. score > 0.7.

The in-memory detector classes that used to live here moved to ``backend/demos/drift_detectors.py``
in 3.6.0; the server computes drift from the shared store in ``app/core/drift_store.py``.
"""
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp

DRIFT_P_VALUE = 0.05
EMBEDDING_MIN_SIMILARITY = 0.3


def ks_score(reference: Sequence[float], current: Sequence[float]) -> Tuple[float, bool]:
    """Two-sample Kolmogorov-Smirnov test; NaNs are replaced by each sample's median."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0 or np.all(np.isnan(ref)) or np.all(np.isnan(cur)):
        return 0.0, False
    ref = np.nan_to_num(ref, nan=float(np.nanmedian(ref)))
    cur = np.nan_to_num(cur, nan=float(np.nanmedian(cur)))
    _, p_value = ks_2samp(ref, cur)
    return float(1.0 - p_value), bool(p_value < DRIFT_P_VALUE)


def chi2_score(reference: Sequence[str], current: Sequence[str]) -> Tuple[float, bool]:
    """Chi-squared test of independence over the union of categories."""
    categories = sorted(set(reference) | set(current))
    if len(categories) < 2 or not reference or not current:
        # One category on both sides: nothing to compare, so no drift.
        return 0.0, False
    table = [
        [sum(1 for value in reference if value == category) for category in categories],
        [sum(1 for value in current if value == category) for category in categories],
    ]
    _, p_value, _, _ = chi2_contingency(table)
    return float(1.0 - p_value), bool(p_value < DRIFT_P_VALUE)


def centroid_score(reference: np.ndarray, current: np.ndarray) -> Tuple[float, bool]:
    """Cosine similarity between two centroids; (0.0, False) if either is empty or all zeros."""
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    if a.shape != b.shape or a.size == 0 or not np.any(a) or not np.any(b):
        return 0.0, False
    similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(1.0 - similarity), bool(similarity < EMBEDDING_MIN_SIMILARITY)

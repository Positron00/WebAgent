"""Drift scoring functions (OBS-007): same distributions aren't drift, shifted ones are."""
import numpy as np

from app.core.drift_detection import centroid_score, chi2_score, ks_score

RNG = np.random.default_rng(7)


def test_ks_same_distribution_is_not_drift():
    score, drift = ks_score(RNG.normal(5, 1, 50), RNG.normal(5, 1, 40))
    assert not drift and 0.0 <= score < 0.95


def test_ks_shifted_distribution_is_drift():
    score, drift = ks_score(RNG.normal(5, 1, 50), RNG.normal(9, 1, 40))
    assert drift and score > 0.95


def test_ks_empty_or_nan_is_not_drift():
    assert ks_score([], [1.0]) == (0.0, False)
    assert ks_score([float("nan")] * 3, [1.0, 2.0]) == (0.0, False)


def test_chi2_same_mix_is_not_drift_and_new_mix_is():
    reference = ["pdf"] * 25 + ["txt"] * 25
    assert not chi2_score(reference, ["pdf"] * 16 + ["txt"] * 14)[1]
    score, drift = chi2_score(reference, ["docx"] * 30)
    assert drift and score > 0.95


def test_chi2_single_category_is_not_drift():
    assert chi2_score(["pdf"] * 50, ["pdf"] * 30) == (0.0, False)


def test_centroid_scores():
    a = np.array([1.0, 0.0, 0.0])
    assert centroid_score(a, a * 2) == (0.0, False)
    score, drift = centroid_score(a, np.array([0.0, 1.0, 0.0]))
    assert drift and abs(score - 1.0) < 1e-9
    assert centroid_score(a, np.zeros(3)) == (0.0, False)
    assert centroid_score(a, np.ones(4)) == (0.0, False)

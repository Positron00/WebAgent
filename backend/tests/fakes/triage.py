"""Fake triage model for unit tests."""
from __future__ import annotations

from typing import Callable, Union

from app.services.triage_model import TriageMetadata


class FakeTriageModel:
    def __init__(
        self,
        p: Union[float, Callable[[str], float]] = 0.5,
        category: str = "test",
        version: str = "fake",
        threshold: float = 0.35,
    ):
        self._p = p
        self._category = category
        self.metadata = TriageMetadata(
            version=version,
            encoder_name="fake",
            threshold=threshold,
            data_hash="fake",
            categories={"test": category},
        )

    def predict_proba(self, text: str) -> float:
        if callable(self._p):
            return float(self._p(text))
        return float(self._p)

    def nearest_category(self, text: str) -> str:
        return self._category


class HashEncoder:
    """Deterministic stand-in for SentenceTransformer: bag of hashed words, normalized."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        import zlib

        import numpy as np

        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


class HashEncoder:
    """Deterministic stand-in for SentenceTransformer: word hashes into a small vector."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        import zlib

        import numpy as np

        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
            if normalize_embeddings:
                out[i] /= np.linalg.norm(out[i]) or 1.0
        return out

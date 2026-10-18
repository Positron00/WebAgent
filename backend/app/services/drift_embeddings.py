"""Embeddings wrapper that feeds embedding drift (OBS-003) without storing single vectors (OBS-004).

Vectors are added to a per-process running sum; each time it holds ``MIN_SAMPLES`` (30) vectors,
one aggregate (sum, count, dimension) is written to the drift store and the sum starts again.
"""
import threading
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core import drift_store
from app.services import run_origin


class DriftRecordingEmbeddings(Embeddings):
    """Passes every call through to ``inner`` and records the vectors it returns."""

    def __init__(self, inner: Embeddings, provider: str):
        self._inner = inner
        self._detector = f"embeddings:{provider}"
        self._lock = threading.Lock()
        self._sum = None
        self._count = 0

    def __getattr__(self, name):  # anything else the vector store asks for
        return getattr(self._inner, name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._inner.embed_documents(texts)
        self._observe(vectors)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vector = self._inner.embed_query(text)
        self._observe([vector])
        return vector

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = await self._inner.aembed_documents(texts)
        self._observe(vectors)
        return vectors

    async def aembed_query(self, text: str) -> List[float]:
        vector = await self._inner.aembed_query(text)
        self._observe([vector])
        return vector

    def _observe(self, vectors: List[List[float]]) -> None:
        if run_origin.topic_run.get():
            return  # scheduled runs repeat the same question; they would skew the drift (REC-108)
        try:
            batch = np.asarray(vectors, dtype=np.float64)
            if batch.ndim != 2 or not len(batch):
                return
            with self._lock:
                if self._sum is not None and self._sum.shape[0] != batch.shape[1]:
                    self._sum, self._count = None, 0  # a different model: start a new sum
                self._sum = batch.sum(axis=0) if self._sum is None else self._sum + batch.sum(axis=0)
                self._count += len(batch)
                if self._count < drift_store.MIN_SAMPLES:
                    return
                total, count = self._sum, self._count
                self._sum, self._count = None, 0
            drift_store.add_aggregate(self._detector, total.tolist(), count, int(total.shape[0]))
        except Exception:  # noqa: BLE001 - drift must never break retrieval or ingestion
            return

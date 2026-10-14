"""IMP-090: vector search runs off the asyncio event loop."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import vectordb as vectordb_module
from main import app


@pytest.mark.asyncio
async def test_slow_vector_search_does_not_block_health(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

    service = vectordb_module.VectorDBService.__new__(vectordb_module.VectorDBService)
    service.initialized = True
    service.vectordb = MagicMock()

    def slow_search(query, k=5):
        time.sleep(0.3)
        return []

    service.vectordb.similarity_search_with_score = slow_search

    async def fake_get():
        return service

    monkeypatch.setattr(vectordb_module, "get_vectordb_service", fake_get)

    client = TestClient(app)
    health_started = time.perf_counter()
    health = await asyncio.to_thread(client.get, "/health")
    health_elapsed = time.perf_counter() - health_started

    search_task = asyncio.create_task(service.search("blocking test", top_k=3))
    await asyncio.sleep(0.05)
    assert health.status_code == 200
    assert health_elapsed < 0.25
    await search_task

"""Samples from real traffic (OBS-001 – OBS-003, OBS-014, OBS-021)."""
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from app.core import drift_store
from app.services import llm_call_metrics, together_llm
from app.services.drift_embeddings import DriftRecordingEmbeddings
from app.services.together_llm import TogetherWithUsage
from tests.test_together_usage import FakeAsyncResponse, FakeResponse, FakeSession

BACKEND = Path(__file__).resolve().parents[1]


def samples():
    return drift_store.read_samples()


def by_detector():
    return Counter(s.detector for s in samples())


def make_llm(label="supervisor"):
    llm = TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=50, together_api_key="test-key")
    llm.call_label = label
    return llm


# --- model calls (OBS-001) ---

def test_a_reply_adds_speed_and_tokens_under_its_label_and_an_ok_outcome(monkeypatch):
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse())
    make_llm("supervisor").invoke("q")
    rows = {s.detector: s.value for s in samples()}
    assert rows["llm_outcome"] == "ok"
    assert rows["llm_tokens:supervisor"] == 17
    assert rows["llm_speed:supervisor"] > 0  # seconds per 1,000 completion tokens


@pytest.mark.parametrize("status,outcome", [(429, "rate_limited"), (500, "error"), (400, "error")])
async def test_failures_add_only_an_outcome(monkeypatch, status, outcome):
    monkeypatch.setattr(together_llm, "ClientSession", lambda **kw: FakeSession(FakeAsyncResponse(status=status)))
    with pytest.raises(Exception):
        await make_llm().ainvoke("q")
    assert [(s.detector, s.value) for s in samples()] == [("llm_outcome", outcome)]


def test_agents_judge_and_triage_label_their_calls():
    from app.services.llm import get_agent_llm, get_trusted_llm

    agent = get_agent_llm("supervisor")
    inner = getattr(agent, "inner", None) or getattr(agent, "_inner", None) or agent
    client = getattr(inner, "llm", inner)
    assert client.call_label == "supervisor"
    assert getattr(get_trusted_llm(call_label="judge"), "llm").call_label == "judge"
    assert getattr(get_trusted_llm(), "llm").call_label == "other"
    assert 'get_trusted_llm(call_label="judge")' in (BACKEND / "app/security/judge.py").read_text()
    assert 'get_trusted_llm(call_label="triage")' in (BACKEND / "app/services/triage.py").read_text()


def test_together_base_url_comes_from_the_environment(monkeypatch):
    assert make_llm().base_url == "https://api.together.xyz/v1/completions"
    monkeypatch.setenv("TOGETHER_API_BASE", "http://127.0.0.1:18200/v1/")
    assert make_llm().base_url == "http://127.0.0.1:18200/v1/completions"


def test_recording_is_cheap():
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    times = []
    for _ in range(1000):
        started = time.perf_counter()
        llm_call_metrics.record("together", "m", 0.4, ok=True, usage=usage, call_label="supervisor")
        times.append(time.perf_counter() - started)
    assert sorted(times)[int(0.95 * len(times))] < 0.005


# --- document loads (OBS-002) ---

@pytest.fixture
def loader(monkeypatch):
    from app.services.document_loader import DocumentLoaderService

    service = DocumentLoaderService.__new__(DocumentLoaderService)
    service.chunk_size, service.chunk_overlap = 500, 50
    service.loader_map = DocumentLoaderService.LOADER_MAP
    service.vector_db = type("FakeVectorDB", (), {"add_documents": lambda self, chunks: None})()
    monkeypatch.setattr(DocumentLoaderService, "_store_artifact", lambda self, path: "artifact-1")
    return service


async def test_a_document_load_adds_one_set_of_samples(loader, tmp_path):
    path = tmp_path / "Quarterly Report.TXT"
    path.write_text("x" * 1234)
    await loader.load_document(path)
    rows = {s.detector: s.value for s in samples()}
    assert rows["doc_size"] == 1234 and rows["doc_type"] == "txt" and rows["doc_outcome"] == "ok"
    assert rows["doc_processing_time"] >= 0
    from app.core.metrics import WEBAGENT_REGISTRY

    assert WEBAGENT_REGISTRY.get_sample_value("document_processing_latency_seconds_count", {"doc_type": "txt"}) >= 1


class BrokenVectorDB:
    def add_documents(self, chunks):
        raise RuntimeError("vector store down")


async def test_a_failed_load_is_an_error_and_a_refused_file_adds_nothing(loader, tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    loader.vector_db = BrokenVectorDB()
    with pytest.raises(RuntimeError):
        await loader.load_document(path)
    assert {s.detector: s.value for s in samples()}["doc_outcome"] == "error"
    before = len(samples())
    unsupported = tmp_path / "image.bmp"
    unsupported.write_bytes(b"0")
    with pytest.raises(ValueError):
        await loader.load_document(unsupported)
    assert len(samples()) == before


async def test_a_directory_adds_one_set_per_file(loader, tmp_path):
    for name in ("a.txt", "b.csv"):
        (tmp_path / name).write_text("a,b\n1,2\n")
    await loader.batch_load_documents(tmp_path)
    assert by_detector()["doc_size"] == 2
    assert [s.value for s in samples() if s.detector == "doc_outcome"] == ["ok", "ok"]


# --- embeddings (OBS-003) ---

class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 2.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 2.0]


def test_embeddings_are_recorded_as_aggregates_of_30():
    emb = DriftRecordingEmbeddings(FakeEmbeddings(), "local")
    emb.embed_documents(["t"] * 29)
    assert not samples()
    emb.embed_query("q")
    [row] = samples()
    assert row.detector == "embeddings:local" and row.agg["n"] == 30 and row.agg["dim"] == 3
    assert np.allclose(row.agg["sum"], [30.0, 0.0, 60.0])


def test_an_upload_through_the_api_adds_one_set_of_samples(loader, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.endpoints import documents
    from app.core.config import settings
    from main import app

    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "ARTIFACTS_PATH", str(tmp_path))
    artifact = tmp_path / "artifact-1"
    artifact.mkdir()
    (artifact / "metadata.json").write_text('{"upload_time": "2026-10-16T00:00:00"}')
    monkeypatch.setattr(documents, "get_document_loader_service", lambda: loader)
    reply = TestClient(app).post(f"{settings.API_V1_STR}/documents/upload",
                                 files={"file": ("notes.csv", b"a,b\n1,2\n", "text/csv")})
    assert reply.status_code == 200, reply.text
    assert by_detector() == Counter({"doc_processing_time": 1, "doc_size": 1, "doc_type": 1, "doc_outcome": 1})
    assert {s.detector: s.value for s in samples()}["doc_type"] == "csv"


def test_an_empty_base_url_means_the_default(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_BASE", "")
    assert make_llm().base_url == "https://api.together.xyz/v1/completions"


# --- the self-hosted model (OBS-001) ---

class FakeSelfHostedResponse:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return {"choices": [{"message": {"content": "Tuples are immutable."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}


def test_the_self_hosted_model_records_its_label_and_a_429(monkeypatch):
    import requests
    from tenacity import RetryError, stop_after_attempt

    from app.services import llm as llm_module

    health = type("Health", (), {"raise_for_status": lambda self: None, "json": lambda self: {"status": "ready"}})
    monkeypatch.setattr(llm_module.requests, "get", lambda *a, **k: health())
    model = llm_module.SelfHostedChatModel(base_url="http://127.0.0.1:9")
    model.call_label = "triage"
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeSelfHostedResponse())
    assert model.invoke("q").content == "Tuples are immutable."
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeSelfHostedResponse(429))
    once = llm_module.SelfHostedChatModel._generate.retry_with(stop=stop_after_attempt(1))
    with pytest.raises((requests.HTTPError, RetryError)):
        once(model, [])
    rows = [(s.detector, s.value) for s in samples()]
    assert ("llm_tokens:triage", 17) in rows and ("llm_outcome", "ok") in rows
    assert rows[-1] == ("llm_outcome", "rate_limited")


def test_the_embeddings_tap_is_cheap():
    """OBS-014 includes embeddings: 1,536-dimension vectors, an aggregate written every 30."""
    vector = [0.01] * 1536

    class Wide:
        def embed_query(self, text):
            return vector

    emb = DriftRecordingEmbeddings(Wide(), "openai")
    times = []
    for _ in range(1000):
        started = time.perf_counter()
        emb.embed_query("q")
        times.append(time.perf_counter() - started)
    assert sorted(times)[int(0.95 * len(times))] < 0.005
    assert by_detector()["embeddings:openai"] == 33

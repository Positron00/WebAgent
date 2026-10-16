"""The drift store keeps numbers, short labels and aggregates of 30+ vectors, never text (OBS-004)."""
from app.core import drift_store
from app.services import together_llm
from app.services.drift_embeddings import DriftRecordingEmbeddings
from app.services.together_llm import TogetherWithUsage
from tests.test_together_usage import FakeResponse

MARKER = "zq-private-marker-7781"


class MarkerEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


async def test_no_text_reaches_the_drift_folder(monkeypatch, tmp_path):
    reply = {"choices": [{"text": f"answer {MARKER}"}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
    monkeypatch.setattr(together_llm.requests, "post", lambda *a, **k: FakeResponse(data=reply))
    llm = TogetherWithUsage(model="meta-models/Muse-Glimmer-30B", max_tokens=50, together_api_key="test-key")
    llm.invoke(f"question {MARKER}")

    emb = DriftRecordingEmbeddings(MarkerEmbeddings(), "local")
    for _ in range(3):
        emb.embed_documents([f"chunk {MARKER}"] * 20)
        emb.embed_query(f"query {MARKER}")

    from app.services.document_loader import DocumentLoaderService

    loader = DocumentLoaderService.__new__(DocumentLoaderService)
    loader.chunk_size, loader.chunk_overlap, loader.loader_map = 500, 50, DocumentLoaderService.LOADER_MAP
    loader.vector_db = type("FakeVectorDB", (), {"add_documents": lambda self, chunks: None})()
    loader._store_artifact = lambda path: "artifact-1"
    path = tmp_path / f"{MARKER}.txt"
    path.write_text(f"body {MARKER}")
    await loader.load_document(path)
    drift_store.compute_once()

    files = list(drift_store.drift_dir().iterdir())
    assert files
    for file in files:
        assert MARKER not in file.read_text(encoding="utf-8", errors="ignore"), file.name
    aggregates = [s for s in drift_store.read_samples() if s.agg]
    assert aggregates and all(s.agg["n"] >= drift_store.MIN_SAMPLES for s in aggregates)

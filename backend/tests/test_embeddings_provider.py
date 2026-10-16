"""IMP-095: embedding provider selection in get_embeddings()."""
from app.core.config import settings
from app.services import vectordb as vectordb_module
from app.services.vectordb import get_embeddings


def test_openai_embeddings_by_default(monkeypatch):
    get_embeddings.cache_clear()
    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setattr(
        vectordb_module,
        "OpenAIEmbeddings",
        lambda **kwargs: ("openai", kwargs),
    )
    emb = get_embeddings()
    # 3.6.0: wrapped to record drift aggregates (OBS-003); the provider's object is inside.
    assert type(emb).__name__ == "DriftRecordingEmbeddings"
    assert emb._inner[0] == "openai"
    get_embeddings.cache_clear()


def test_local_embeddings_provider(monkeypatch):
    get_embeddings.cache_clear()
    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "local")

    class FakeHF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", FakeHF)
    emb = get_embeddings()
    assert emb.kwargs["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.kwargs["cache_folder"] == "/opt/models"
    get_embeddings.cache_clear()


def test_no_wrapper_when_drift_is_off(monkeypatch):
    get_embeddings.cache_clear()
    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setattr(settings, "DRIFT_ENABLED", False)
    monkeypatch.setattr(vectordb_module, "OpenAIEmbeddings", lambda **kwargs: ("openai", kwargs))
    assert get_embeddings()[0] == "openai"
    get_embeddings.cache_clear()

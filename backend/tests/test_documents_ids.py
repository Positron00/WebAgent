"""Document IDs, bounds and upload type check (IMP-021, IMP-022); /api/task query limit."""
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.endpoints import documents
from app.core.config import settings
from main import app

DOCS = f"{settings.API_V1_STR}/documents"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "API_KEY_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "ARTIFACTS_PATH", str(tmp_path))
    return TestClient(app)


def _make_artifact(root: Path) -> str:
    doc_id = str(uuid.uuid4())
    folder = Path(root) / doc_id
    folder.mkdir()
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps({"original_filename": "notes.txt", "upload_time": "2026-01-01T00:00:00", "content_type": "txt"}),
        encoding="utf-8",
    )
    return doc_id


@pytest.mark.parametrize("bad_id", ["..", "%2e%2e", "", "not-a-uuid", "../../etc/passwd", None])
def test_artifact_dir_rejects_non_uuids_without_touching_disk(monkeypatch, bad_id):
    touched = []
    monkeypatch.setattr(Path, "exists", lambda self: touched.append(self) or True)

    with pytest.raises(HTTPException) as err:
        documents._artifact_dir(bad_id)

    assert err.value.status_code == 404
    assert touched == []


@pytest.mark.parametrize("path", ["not-a-uuid", "%2e%2e", "..%2f..%2fetc", ""])
def test_get_document_with_bad_id_is_404(client, path):
    assert client.get(f"{DOCS}/document/{path}").status_code == 404


def test_get_document_with_real_id_works(client, tmp_path):
    doc_id = _make_artifact(tmp_path)

    reply = client.get(f"{DOCS}/document/{doc_id}")

    assert reply.status_code == 200
    assert reply.json()["id"] == doc_id


def test_unknown_uuid_is_404(client):
    assert client.get(f"{DOCS}/document/{uuid.uuid4()}").status_code == 404
    assert client.get(f"{DOCS}/related/{uuid.uuid4()}").status_code == 404


@pytest.mark.parametrize("path", ["not-a-uuid", "%2e%2e"])
def test_related_with_bad_id_is_404(client, path):
    with patch.object(documents, "get_graph_db_service") as graph:
        assert client.get(f"{DOCS}/related/{path}").status_code == 404
    graph.assert_not_called()


@pytest.mark.parametrize("query,status", [
    ("depth=0", 422), ("depth=1", 200), ("depth=3", 200), ("depth=4", 422),
    ("limit=0", 422), ("limit=1", 200), ("limit=50", 200), ("limit=51", 422),
])
def test_related_bounds(client, tmp_path, query, status):
    doc_id = _make_artifact(tmp_path)
    graph = MagicMock()
    graph.find_related_documents.return_value = [{"id": doc_id}, {"id": "../escape"}]
    with patch.object(documents, "get_graph_db_service", return_value=graph):
        reply = client.get(f"{DOCS}/related/{doc_id}?{query}")

    assert reply.status_code == status
    if status == 200:
        assert [d["id"] for d in reply.json()] == [doc_id]


@pytest.mark.parametrize("limit,status", [(0, 422), (1, 200), (50, 200), (51, 422)])
def test_search_limit_bounds(client, limit, status):
    vector_db = MagicMock()

    async def search(query, limit):
        return []

    vector_db.search = search
    with patch.object(documents, "get_vectordb_service", return_value=vector_db):
        reply = client.post(f"{DOCS}/search", json={"query": "q", "limit": limit})

    assert reply.status_code == status


@pytest.mark.parametrize("filename", ["evil.exe", "script.sh", "noextension"])
def test_unsupported_upload_is_415_before_writing(client, filename):
    with patch.object(documents.tempfile, "NamedTemporaryFile") as temp, \
            patch.object(documents, "get_document_loader_service") as loader:
        reply = client.post(f"{DOCS}/upload", files={"file": (filename, b"data")})

    assert reply.status_code == 415
    temp.assert_not_called()
    loader.assert_not_called()


def test_supported_upload_reaches_the_loader(client, tmp_path):
    doc_id = _make_artifact(tmp_path)
    loader = MagicMock()

    async def load_document(path, metadata):
        return doc_id

    loader.load_document = load_document
    with patch.object(documents, "get_document_loader_service", return_value=loader):
        reply = client.post(f"{DOCS}/upload", files={"file": ("Notes.TXT", b"hello")})

    assert reply.status_code == 200
    assert reply.json()["id"] == doc_id


def test_supported_extensions_match_the_loader_table():
    from app.services.document_loader import DocumentLoaderService

    assert DocumentLoaderService.supported_extensions() == set(DocumentLoaderService.LOADER_MAP)
    assert ".pdf" in DocumentLoaderService.supported_extensions()


@pytest.mark.parametrize("length,status", [(19999, 200), (20000, 200), (20001, 422)])
def test_task_query_limit(client, monkeypatch, length, status):
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(update={"enabled": False}))
    with patch("app.api.endpoints.task.task_manager"):
        reply = client.post("/api/task", json={"query": "x" * length, "mode": "off"})

    assert reply.status_code == status

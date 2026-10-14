"""The image must ship the data and models the agents load: uat and prod don't download at runtime."""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _requested() -> set:
    names = set()
    for path in (BACKEND / "app").rglob("*.py"):
        for call in re.findall(r"ensure_nltk_data\(([^)]*)\)", path.read_text(encoding="utf-8")):
            names.update(re.findall(r"['\"](\w+)['\"]", call))
    return names


def test_dockerfile_downloads_every_requested_resource():
    # Seen in the 3.2.4 shadow-week simulation: NLTK 3.9 tokenizers need punkt_tab, the image only
    # had punkt, and every document extraction failed.
    dockerfile = (BACKEND / "Dockerfile").read_text(encoding="utf-8")
    line = re.search(r"nltk\.downloader[^\n]*\\\n([^\n]*)", dockerfile).group(1)
    requested = _requested()
    assert "punkt_tab" in requested
    assert requested <= set(line.split())


def test_dockerfile_bakes_the_default_reranker():
    # Before 3.2.6 the reranker downloaded on the first research task into /app/data/hf-cache:
    # the pod needed Hugging Face access, and it failed where that folder wasn't writable.
    source = (BACKEND / "app" / "agents" / "internal_research.py").read_text(encoding="utf-8")
    default = re.search(r'get\("reranker_model",\s*"([^"]+)"\)', source).group(1)
    dockerfile = (BACKEND / "Dockerfile").read_text(encoding="utf-8")
    assert f"CrossEncoder('{default}')" in dockerfile
    assert "HF_HUB_CACHE=/opt/hf-hub" in dockerfile

"""The image must ship every NLTK resource the agents ask for: uat and prod don't download at startup."""
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

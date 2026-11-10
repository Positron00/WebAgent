#!/usr/bin/env python3
"""Label a review sheet of web pages in the browser: Benign, Malicious or Skip, one page at a time.

Run from backend/ (trafilatura on the path, as for import_reviewed_pages.py):
    python scripts/label_pages.py ~/webagent-shadow-feed/labels/to_check2.csv [--port 8765]

Opens http://127.0.0.1:<port>/. For each unlabelled row it shows what the guard scored: the page's
main content (fetched again, cut with `clip`), split into the model's chunks, highest injection
score first. "Open page" shows the live page in a second window that follows along.
Keys: b benign, m malicious (written as `injection`, the sheet's label), s skip, left arrow back,
o open page. Every answer is written to the sheet at once (a .bak copy is kept from the start), so
the page can be closed and reopened at any time. Skip leaves the label blank.

Serves on localhost only: the page text is third-party content and is never written to disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

SCRIPTS = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # pages are scored from worker threads
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPTS.parent))

from build_security_dataset import clip  # noqa: E402
from import_reviewed_pages import EXTRACTOR, page_text  # noqa: E402

LABELS = {"benign": "benign", "malicious": "injection", "skip": ""}


class Sheet:
    """The CSV, rewritten in place (temp file and rename) after every answer."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.fields = list(reader.fieldnames or [])
            self.rows = list(reader)
        for col in ("label", "note"):
            if col not in self.fields:
                self.fields.append(col)

    def save(self, index: int, label: str, note: str) -> None:
        with self.lock:
            self.rows[index]["label"] = label
            self.rows[index]["note"] = note
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp, self.path)


class Pages:
    """Fetches and scores pages in the background, in sheet order, so the next one is ready."""

    def __init__(self, sheet: Sheet, workers: int):
        from app.security import context_defender

        self.model = context_defender.get_model()
        self.chunk_text = context_defender.chunk_text
        self.cache: Dict[int, dict] = {}
        self.ready: Dict[int, threading.Event] = {i: threading.Event() for i in range(len(sheet.rows))}
        self.sheet = sheet
        pool = ThreadPoolExecutor(workers)
        todo = [i for i, r in enumerate(sheet.rows) if not (r.get("label") or "").strip()]
        rest = [i for i in range(len(sheet.rows)) if i not in todo]
        for i in todo + rest:
            pool.submit(self._load, i)

    def _load(self, i: int) -> None:
        url = self.sheet.rows[i]["ref"]
        try:
            text, status = page_text(url)
            chunks: List[dict] = []
            if text:
                text = clip(text)
                if self.model is not None:
                    pieces = self.chunk_text(text, self.model.tokenizer) or [text]
                    _, per_chunk, _ = self.model.score_text(text, "web_result")
                    chunks = [{"n": n, "text": p, "injection": s["injection"], "jailbreak": s["jailbreak"]}
                              for n, (p, s) in enumerate(zip(pieces, per_chunk))]
                    chunks.sort(key=lambda c: -c["injection"])
                else:
                    chunks = [{"n": 0, "text": text, "injection": None, "jailbreak": None}]
            self.cache[i] = {"status": status, "chunks": chunks, "extractor": EXTRACTOR}
        except Exception as exc:  # noqa: BLE001 - shown on the page; the row can still be labelled
            self.cache[i] = {"status": f"error: {type(exc).__name__}", "chunks": [], "extractor": EXTRACTOR}
        self.ready[i].set()

    def get(self, i: int, wait: float = 60.0) -> dict:
        self.ready[i].wait(wait)
        return self.cache.get(i, {"status": "still loading", "chunks": [], "extractor": EXTRACTOR})


def make_handler(sheet: Sheet, pages: Pages):
    page_html = (SCRIPTS / "label_pages.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _json(self, body: dict, code: int = 200) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page_html)))
                self.end_headers()
                self.wfile.write(page_html)
            elif self.path == "/api/sheet":
                self._json({"name": sheet.path.name,
                            "rows": [{"ref": r["ref"], "label": r.get("label", ""), "note": r.get("note", ""),
                                      "injection": r.get("injection"), "jailbreak": r.get("jailbreak"),
                                      "pattern_hits": r.get("pattern_hits", "")} for r in sheet.rows]})
            elif self.path.startswith("/api/page/"):
                try:
                    i = int(self.path.rsplit("/", 1)[1])
                    sheet.rows[i]
                except (ValueError, IndexError):
                    return self._json({"error": "no such row"}, 404)
                self._json(pages.get(i))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            i, choice = body.get("index"), body.get("choice")
            if not isinstance(i, int) or not 0 <= i < len(sheet.rows) or choice not in LABELS:
                return self._json({"error": "bad request"}, 400)
            if sheet.rows[i]["ref"] != body.get("ref"):
                return self._json({"error": "row changed; reload the page"}, 409)
            sheet.save(i, LABELS[choice], str(body.get("note") or "")[:500])
            self._json({"ok": True, "label": LABELS[choice]})

    return Handler


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    path = args.sheet.expanduser()
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    sheet = Sheet(path)
    open_rows = sum(1 for r in sheet.rows if not (r.get("label") or "").strip())
    print(f"{len(sheet.rows)} rows, {open_rows} unlabelled; extractor {EXTRACTOR}; backup {backup.name}")
    pages = Pages(sheet, args.workers)
    if pages.model is None:
        print("ContextDefender model not loaded: pages are shown whole, without chunk scores")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sheet, pages))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Labelling at {url} (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    labelled = sum(1 for r in sheet.rows if (r.get("label") or "").strip())
    print(f"Stopped; {labelled}/{len(sheet.rows)} rows labelled in {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

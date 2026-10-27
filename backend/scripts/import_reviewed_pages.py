#!/usr/bin/env python3
"""Turn a reviewed shadow week into ContextDefender rows (task 5.3, reviewed import only).

Run from backend/:
    python scripts/import_reviewed_pages.py --dir <data>/security/collected --since 2026-10-17
        --labels review_labelled.csv [more_labels.csv ...] [--out data/security/local/web_pages.jsonl]

The collection keeps web-result URLs, not text, so each page is fetched again. The guard scored
Tavily's extract of the page's main content, not the whole page, so the main content is taken with
trafilatura when it is installed (`pip install trafilatura`; menus, cookie banners and footers on a
raw page made the 2026-10-13 model decline 46% of ordinary pages) and plain visible text otherwise,
then cut like the other long texts (`clip`: first and last 1,500 characters). Rows:
- `shadow:web_reviewed`: pages on the labelled sheet marked benign (the guard would have acted on
  them); split by URL hash, about two thirds `train` and one third `test`.
- `shadow:web_unreviewed`: every other web page of the week; `test` only, never trained on, because
  nobody read them.
Pages labelled injection are counted and not imported: a page needs the injection's span first.

The page text is third-party content: the output folder is git-ignored and must stay out of the repo.
Only counts are printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_security_dataset import clip  # noqa: E402
from review_security_collection import load_rows  # noqa: E402

UA = "Mozilla/5.0 (compatible; WebAgent-review/1.0; one-off labelling of pages a research run cited)"
NO_LABELS = {"injection": "none", "harm": "none", "sensitive_data": "none", "jailbreak": "none"}
MAX_BYTES = 3_000_000
try:
    import trafilatura  # noqa: F401
    EXTRACTOR = "trafilatura"
except ImportError:
    EXTRACTOR = "visible-text"


def page_split(url: str) -> str:
    return "test" if int(hashlib.sha256(url.encode("utf-8")).hexdigest(), 16) % 3 == 0 else "train"


def page_text(url: str, timeout: int = 20) -> Tuple[Optional[str], str]:
    """Visible text of an HTML page (scripts, styles and SVG dropped), or None and why not."""
    from bs4 import BeautifulSoup

    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            kind = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_BYTES)
    except Exception as exc:  # noqa: BLE001 - a page that won't load is skipped and counted
        return None, f"fetch error: {type(exc).__name__}"
    if "html" not in kind and "text" not in kind:
        return None, "not html"
    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    if trafilatura is not None:
        text = trafilatura.extract(raw.decode("utf-8", errors="replace"), include_comments=False,
                                   include_tables=True) or ""
    else:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(" ")
    text = " ".join(text.split())
    return (text, "ok") if len(text) >= 200 else (None, "too short")


def web_pages(rows: List[dict]) -> List[str]:
    return sorted({r["item"]["ref"] for r in rows
                   if r.get("checkpoint") == 1 and r.get("item") and r["item"].get("source") == "web_result"
                   and str(r["item"].get("ref", "")).startswith("http")})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="the week's security/collected folder")
    parser.add_argument("--since", default=None)
    parser.add_argument("--labels", type=Path, nargs="+", required=True,
                        help="labelled review sheets (columns source, ref, label); later rounds add sheets")
    parser.add_argument("--out", type=Path, default=BACKEND / "data/security/local/web_pages.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests per worker")
    args = parser.parse_args(argv)

    sheet = []
    for path in args.labels:
        with open(path, newline="", encoding="utf-8") as handle:
            sheet += [r for r in csv.DictReader(handle) if r.get("source") == "web_result" and r.get("ref")]
    label_of = {}
    for r in sheet:
        label = (r.get("label") or "").strip().lower()
        if label:
            label_of[r["ref"]] = "injection" if "injection" in (label, label_of.get(r["ref"])) else label
    flagged = {r["ref"] for r in sheet}
    urls = web_pages(load_rows(args.dir.expanduser(), args.since))

    plan = []
    counts: Counter = Counter()
    for url in urls:
        label = label_of.get(url)
        if label == "injection":
            counts["injection (not imported)"] += 1
        elif label == "benign":
            plan.append((url, "shadow:web_reviewed", page_split(url)))
        elif url in flagged:
            counts["flagged, unlabelled (not imported)"] += 1
        else:
            plan.append((url, "shadow:web_unreviewed", "test"))

    def fetch(job):
        url, origin, split = job
        text, status = page_text(url)
        time.sleep(args.pause)
        return url, origin, split, text, status

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for i, (url, origin, split, text, status) in enumerate(pool.map(fetch, plan), 1):
            counts[f"{origin} {status.split(':')[0]}"] += 1
            if text:
                counts[f"{origin} {split}"] += 1
                out.write(json.dumps({"text": clip(text), "source": "web_result", "labels": dict(NO_LABELS),
                                      "origin": origin, "ref": url, "split": split, "extractor": EXTRACTOR}) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(plan)} pages", flush=True)
    print(json.dumps({"web_pages": len(urls), **dict(sorted(counts.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

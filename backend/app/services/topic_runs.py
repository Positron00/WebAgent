"""Run records of recurring research: URL normalisation, the source diff and pruning
(recurring-research spec REC-111, REC-115).

The scheduler is the only writer of run records (REC-101). This module imports nothing heavy, since
the scheduler process uses it (RNFR-02).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.services import topic_files as files

DIFF_WINDOW = 3            # new sources are those not seen in the last 3 runs (REC-111)
OPEN_STATUSES = ("starting", "running")
_TRACKING = {"fbclid", "gclid"}


# --- URLs -----------------------------------------------------------------------------------------

def normalise_url(url: Any) -> Optional[str]:
    """A comparable form of a URL, or None when there isn't one.

    Lower-cases the scheme and host, strips `www.`, drops the fragment, the `utm_*`, `fbclid` and
    `gclid` parameters and a trailing `/`. The path and the other parameters keep their case.
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.netloc:
        return url.rstrip("/") or None
    host = parts.netloc.lower()
    if "@" in host:                       # never keep credentials in a stored URL
        host = host.rsplit("@", 1)[1]
    if host.startswith("www."):
        host = host[4:]
    query = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
    ]
    path = parts.path.rstrip("/")
    out = urlunsplit((parts.scheme.lower(), host, path, urlencode(query, doseq=True), ""))
    return out.rstrip("/") or None


def source_url(source: Any) -> Optional[str]:
    """The URL of one source of a result: a dict with `url` (or `link`/`href`), or a string."""
    if isinstance(source, dict):
        for key in ("url", "link", "href"):
            if isinstance(source.get(key), str) and source[key].strip():
                return source[key]
        return None
    if isinstance(source, str):
        return source
    return None


def clean_sources(sources: Any) -> List[Dict[str, Any]]:
    """The result's sources as `{url, title}` dicts, without duplicates (by normalised URL).

    Sources without a URL are kept (with `url: None`) so the report's citations still line up.
    """
    if not isinstance(sources, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for source in sources:
        url = source_url(source)
        key = normalise_url(url)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        title = source.get("title") if isinstance(source, dict) else None
        out.append({"url": url, "title": title if isinstance(title, str) else None})
    return out


def url_set(sources: Iterable[Any]) -> List[str]:
    """Normalised URLs of `sources`, in order and without duplicates."""
    out: List[str] = []
    for source in sources or []:
        key = normalise_url(source_url(source))
        if key is not None and key not in out:
            out.append(key)
    return out


# --- the diff -------------------------------------------------------------------------------------

def diff(sources: List[Any], previous_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`change` of a run (REC-111).

    `previous_runs` are the topic's earlier completed runs, newest first. `new_sources` are this
    run's URLs not in the last 3 of them, `dropped_sources` those of the previous run missing now.
    A first run (no earlier completed run) marks all its sources new, with `first: true`.
    """
    current = url_set(sources)
    earlier = [r for r in previous_runs if r.get("status") == "completed"]
    if not earlier:
        return {"first": True, "new_sources": current, "dropped_sources": [], "changed": True}
    seen = set()
    for record in earlier[:DIFF_WINDOW]:
        seen.update(url_set(record.get("sources") or []))
    new = [u for u in current if u not in seen]
    previous = url_set(earlier[0].get("sources") or [])
    dropped = [u for u in previous if u not in set(current)]
    return {"first": False, "new_sources": new, "dropped_sources": dropped, "changed": bool(new)}


NO_CHANGE = "No material change"     # the change summary's answer when nothing material changed


def previous_completed(records: List[Dict[str, Any]], run_id: str) -> Optional[Dict[str, Any]]:
    """The newest completed run other than `run_id` (records newest first), or None."""
    for record in records:
        if record.get("run_id") != run_id and record.get("status") == "completed":
            return record
    return None


def apply_summary(change: Dict[str, Any], summary: Optional[str], summary_changed: bool,
                  error: Optional[str]) -> bool:
    """Put the change summary (REC-130) on a run's `change`; True when the summary itself reports
    a material change. `change.changed` becomes true also for a summary other than "No material
    change" (REC-131). A failed call keeps the source diff, with `summary_error`."""
    if error or not isinstance(summary, str):
        change["summary"] = None
        change["summary_error"] = str(error or "no_summary")[:100]
        return False
    change["summary"] = summary
    change.pop("summary_error", None)
    material = bool(summary_changed) and summary.strip() != NO_CHANGE
    if material:
        change["changed"] = True
    return material


def flagged_refs(result: Any) -> List[str]:
    """Refs checkpoint 1 flagged in a task result: `security.dropped` (act mode) and
    `security.shadow_flagged` (shadow mode), in order and without duplicates (REC-135)."""
    security = result.get("security") if isinstance(result, dict) else None
    if not isinstance(security, dict):
        return []
    out: List[str] = []
    for key in ("dropped", "shadow_flagged"):
        rows = security.get(key)
        for row in rows if isinstance(rows, list) else []:
            ref = row.get("ref") if isinstance(row, dict) else None
            if isinstance(ref, str) and ref.strip() and ref not in out:
                out.append(ref)
    return out


# --- records --------------------------------------------------------------------------------------

def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def new_record(topic_id: str, run_id: str, started_at: datetime, trigger: str) -> Dict[str, Any]:
    """A run record in `starting`, written before the task is requested (REC-106)."""
    return {
        "run_id": run_id,
        "topic_id": topic_id,
        "task_id": files.task_id_for_run(run_id),
        "status": "starting",
        "trigger": trigger,
        "error_code": None,
        "started_at": _iso(started_at),
        "completed_at": None,
        "duration_seconds": None,
        "usage": None,
        "title": None,
        "report": None,
        "sources": [],
        "change": None,
        "flagged_refs": [],
    }


def save(record: Dict[str, Any]) -> None:
    files.write_json_atomic(files.run_path(record["topic_id"], record["run_id"]), record)


def load(topic_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    return files.read_json(files.run_path(topic_id, run_id))


def _sort_key(record: Dict[str, Any]) -> datetime:
    return (files.parse_time(record.get("started_at")) or files.parse_time(record.get("completed_at"))
            or datetime.min.replace(tzinfo=timezone.utc))


def list_records(topic_id: str) -> List[Dict[str, Any]]:
    """The topic's run records, newest first."""
    folder = files.runs_dir(topic_id)
    if not folder.is_dir():
        return []
    records = []
    for path in folder.glob("*.json"):
        if not files.is_uuid(path.stem):
            continue
        record = files.read_json(path)
        if record is not None:
            record.setdefault("run_id", path.stem)
            record.setdefault("topic_id", topic_id)
            records.append(record)
    records.sort(key=_sort_key, reverse=True)
    return records


def task_id_of(record: Dict[str, Any]) -> Optional[str]:
    task_id = record.get("task_id")
    if files.is_uuid(task_id):
        return str(task_id)
    if files.is_uuid(record.get("run_id")):
        return files.task_id_for_run(record["run_id"])
    return None


def prune(topic_id: str, keep: int) -> List[Dict[str, Any]]:
    """Remove all but the newest `keep` finished records. Returns the removed records.

    Records still `starting` or `running` are never removed. The caller removes the removed runs'
    task results.
    """
    removed = []
    finished = [r for r in list_records(topic_id) if r.get("status") not in OPEN_STATUSES]
    for record in finished[max(keep, 0):]:
        try:
            files.run_path(topic_id, record["run_id"]).unlink(missing_ok=True)
        except (OSError, ValueError):
            continue
        removed.append(record)
    return removed

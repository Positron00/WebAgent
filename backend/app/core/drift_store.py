"""Shared drift store (OBS-005 – OBS-011): samples and drift/alert state under data/drift/.

Every worker appends samples; one worker per tick (the one holding ``compute.lock``) computes the
detectors and alerts and writes ``state.json``; every worker sets its gauges from that file, so all
workers and restarts give the same answer. Only numbers, short labels and embedding aggregates of at
least 30 vectors are stored (OBS-004).
"""
import asyncio
import fcntl
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core import drift_alerts
from app.core.config import settings
from app.core.drift_detection import centroid_score, chi2_score, ks_score
from app.core.drift_reference import detector_kind, group_of, update_detector

logger = logging.getLogger(__name__)
MIN_SAMPLES = 30
_last_state: Dict[str, Any] = {}


def drift_dir() -> Path:
    return Path(settings.DATA_DIR) / "drift"


def _empty_state() -> Dict[str, Any]:
    return {"computed_at": None, "resets": {}, "detectors": {}, "alerts": [], "figures": {}, "last_cleanup": None}


def _append(record: Dict[str, Any], lock: bool = False) -> None:
    folder = drift_dir()
    folder.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    path = folder / f"samples-{datetime.now():%Y-%m-%d}.jsonl"
    with _locked(folder / "append.lock") if lock else _nullcontext():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)  # one write per line: O_APPEND keeps workers' lines whole
        finally:
            os.close(fd)


def add(detector: str, value: Any) -> None:
    """Record one sample (a number or a short label). Never raises."""
    if not settings.DRIFT_ENABLED:
        return
    try:
        _append({"ts": datetime.now().isoformat(timespec="milliseconds"), "d": detector, "v": value})
        from app.core.observability import DRIFT_SAMPLES

        DRIFT_SAMPLES.labels(detector=detector).inc()
    except Exception as exc:  # noqa: BLE001 - drift must never break a model call or an upload
        logger.debug("Drift sample not recorded: %s", exc)


def add_aggregate(detector: str, vector_sum: List[float], count: int, dim: int) -> None:
    """Record one embedding aggregate (the caller passes count >= MIN_SAMPLES). Never raises."""
    if not settings.DRIFT_ENABLED:
        return
    try:
        _append({"ts": datetime.now().isoformat(timespec="milliseconds"), "d": detector,
                 "agg": {"sum": vector_sum, "n": count, "dim": dim}}, lock=True)
        from app.core.observability import DRIFT_SAMPLES

        DRIFT_SAMPLES.labels(detector=detector).inc(count)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Drift aggregate not recorded: %s", exc)


@contextmanager
def _nullcontext():
    yield


@contextmanager
def _locked(path: Path, blocking: bool = True):
    """flock on `path`; yields True if held (always when blocking), False if busy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_state() -> Dict[str, Any]:
    """The last computed state; the last good copy (or an empty state) if the file can't be read."""
    global _last_state
    path = drift_dir() / "state.json"
    try:
        _last_state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _last_state = _empty_state()
    except (OSError, ValueError) as exc:
        logger.warning("Drift state unreadable (%s); using the last good copy", exc)
    return _last_state or _empty_state()


def _write_state(state: Dict[str, Any]) -> None:
    folder = drift_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"state.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, folder / "state.json")


def read_samples() -> List[drift_alerts.Sample]:
    samples: List[drift_alerts.Sample] = []
    for path in sorted(drift_dir().glob("samples-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                samples.append(drift_alerts.Sample(datetime.fromisoformat(row["ts"]), row["d"], row.get("v"), row.get("agg")))
            except (ValueError, KeyError, TypeError):
                continue  # a torn or foreign line
    samples.sort(key=lambda s: s.ts)
    return samples


def reset(group: str, now: Optional[datetime] = None) -> None:
    """Start new references for a group (`llm` or `doc`); waits for a running computation."""
    with _locked(drift_dir() / "compute.lock"):
        state = load_state()
        state.setdefault("resets", {})[group] = (now or datetime.now()).isoformat(timespec="seconds")
        state["detectors"] = {n: d for n, d in state.get("detectors", {}).items() if group_of(n) != group}
        _write_state(state)


def compute_once(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Freeze references, score detectors, evaluate alerts and write the state (caller holds the lock)."""
    now = now or datetime.now()
    state = load_state()
    samples = read_samples()
    resets = state.get("resets", {})
    previous = state.get("detectors", {})
    names = {s.detector for s in samples if detector_kind(s.detector)} | set(previous)
    scorers = {"ks": ks_score, "chi2": chi2_score, "centroid": centroid_score}
    detectors = {}
    for name in sorted(names):
        detectors[name] = update_detector(name, previous.get(name, {}), samples, resets.get(group_of(name)),
                                          settings.DRIFT_WINDOW, MIN_SAMPLES, scorers, now)
        if detectors[name].get("drift") and not previous.get(name, {}).get("drift"):
            logger.warning("Drift detected in %s (score %.2f)", name, detectors[name]["score"])
            _log_drift_to_mlflow(name, detectors[name]["score"])
    figures = drift_alerts.figures(samples, now)
    since = {a["name"]: a.get("since") for a in state.get("alerts", [])}
    alerts = []
    for alert in drift_alerts.evaluate(figures, detectors, settings.TOKEN_ALERT_DAILY):
        if alert["name"] not in since:
            logger.warning("Alert %s: %s", alert["name"], alert["message"])
        alerts.append({**alert, "since": since.get(alert["name"]) or now.isoformat(timespec="seconds")})
    state.update(computed_at=now.isoformat(timespec="seconds"), detectors=detectors, alerts=alerts, figures=figures)
    last = state.get("last_cleanup")
    if not last or now - datetime.fromisoformat(last) > timedelta(days=1):
        cleanup(settings.RESULTS_RETENTION_DAYS, now)
        state["last_cleanup"] = now.isoformat(timespec="seconds")
    _write_state(state)
    return state


def _log_drift_to_mlflow(name: str, score: float) -> None:
    try:
        from app.core.observability import log_to_mlflow

        log_to_mlflow(metrics={f"{name.replace(':', '_')}_drift_score": score}, params={"detector": name},
                      experiment_name="drift_detection")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Drift not logged to MLflow: %s", exc)


def apply_gauges(state: Dict[str, Any]) -> None:
    from app.core.observability import (ALERT_ACTIVE, DATA_DRIFT_SCORE, DRIFT_CURRENT_SAMPLES,
                                        DRIFT_DETECTED, DRIFT_ENOUGH_DATA)

    active = {a["name"] for a in state.get("alerts", [])}
    for name, (severity, _) in drift_alerts.ALERTS.items():
        ALERT_ACTIVE.labels(alert=name, severity=severity).set(1 if name in active else 0)
    for name, d in state.get("detectors", {}).items():
        DRIFT_DETECTED.labels(detector=name).set(1 if d.get("drift") else 0)
        DRIFT_ENOUGH_DATA.labels(detector=name).set(1 if d.get("enough_data") else 0)
        DRIFT_CURRENT_SAMPLES.labels(detector=name).set(d.get("current_count", 0))
        DATA_DRIFT_SCORE.labels(component=name).set(d.get("score") or 0.0)


async def compute_loop(interval_sec: Optional[float] = None) -> None:
    """Forever: compute if this worker gets the lock, then set gauges from the shared state."""
    interval = interval_sec or settings.DRIFT_INTERVAL_SECONDS
    while True:
        try:
            with _locked(drift_dir() / "compute.lock", blocking=False) as held:
                if held:
                    await asyncio.to_thread(compute_once)
            apply_gauges(load_state())
        except Exception as exc:  # noqa: BLE001 - the loop must keep running
            logger.warning("Drift computation failed: %s", exc)
        await asyncio.sleep(interval)


def cleanup(retention_days: int, now: Optional[datetime] = None) -> int:
    """Delete sample files older than the retention period (by the date in the name)."""
    cutoff = ((now or datetime.now()) - timedelta(days=retention_days)).date()
    removed = 0
    for path in drift_dir().glob("samples-*.jsonl"):
        try:
            if datetime.strptime(path.stem[len("samples-"):], "%Y-%m-%d").date() < cutoff:
                path.unlink()
                removed += 1
        except (ValueError, OSError):
            continue
    return removed

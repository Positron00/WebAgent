"""Detector table and per-detector reference / current logic for the drift store (OBS-006 – OBS-008).

A reference is the first ``window`` samples after the group's last reset (for embeddings: the first
aggregates reaching ``window`` vectors), frozen in the state so retention can't delete it. Current is
the latest samples after the reference, up to ``window``. A detector scores once it has a frozen
reference and at least ``min_samples`` current samples.
"""
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# (name or prefix ending in ":", kind, group)
DETECTORS: Tuple[Tuple[str, str, str], ...] = (
    ("llm_speed:", "ks", "llm"),
    ("llm_tokens:", "ks", "llm"),
    ("embeddings:", "centroid", "llm"),
    ("doc_processing_time", "ks", "doc"),
    ("doc_size", "ks", "doc"),
    ("doc_type", "chi2", "doc"),
)
PRIVATE_FIELDS = ("reference", "reference_centroid", "reference_end", "reference_after")


def _match(name: str) -> Optional[Tuple[str, str]]:
    for key, kind, group in DETECTORS:
        if name == key or (key.endswith(":") and name.startswith(key)):
            return kind, group
    return None


def detector_kind(name: str) -> Optional[str]:
    match = _match(name)
    return match[0] if match else None


def group_of(name: str) -> Optional[str]:
    match = _match(name)
    return match[1] if match else None


def public(detector: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the API shows (no stored reference values or centroids)."""
    return {k: v for k, v in detector.items() if k not in PRIVATE_FIELDS}


def _ts(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _centroid(aggregates: List[Any]) -> Tuple[np.ndarray, int]:
    total = np.sum([np.asarray(a.agg["sum"], dtype=float) for a in aggregates], axis=0)
    count = sum(int(a.agg["n"]) for a in aggregates)
    return total / count, count


def update_detector(name: str, previous: Dict[str, Any], samples: List[Any], reset_at: Optional[str],
                    window: int, min_samples: int, scorers: Dict[str, Callable], now: datetime) -> Dict[str, Any]:
    kind, group = _match(name)
    det = {k: previous[k] for k in ("reference", "reference_centroid", "reference_dim", "reference_start",
                                    "reference_end", "reference_count", "reference_after", "last_drift_at")
           if k in previous}
    det.update(group=group, kind=kind)
    starts = [t for t in (_ts(reset_at), _ts(det.get("reference_after"))) if t]
    own = [s for s in samples if s.detector == name and (not starts or s.ts > max(starts))]
    frozen = "reference" in det or "reference_centroid" in det
    if not frozen:
        det["reference_count"] = 0

    if not frozen and own:
        det["reference_start"] = _iso(own[0].ts)
        if kind == "centroid":
            used, count = [], 0
            for s in own:
                if s.agg and count < window:
                    used.append(s)
                    count += int(s.agg["n"])
            if count >= window:
                centroid, count = _centroid(used)
                det.update(reference_centroid=centroid.tolist(), reference_dim=int(used[0].agg["dim"]),
                           reference_end=_iso(used[-1].ts), reference_count=count)
                frozen = True
            else:
                det["reference_count"] = count
        elif len(own) >= window:
            det.update(reference=[s.value for s in own[:window]], reference_end=_iso(own[window - 1].ts),
                       reference_count=window)
            frozen = True
        else:
            det["reference_count"] = len(own)

    current = [s for s in own if frozen and s.ts > _ts(det["reference_end"])]
    score, drift, current_count = None, False, 0
    if kind == "centroid" and current:
        if any(int(s.agg["dim"]) != det.get("reference_dim") for s in current if s.agg):
            # A new embeddings provider: start a new reference from its first aggregate.
            first_new = next(s for s in current if s.agg and int(s.agg["dim"]) != det.get("reference_dim"))
            fresh = {"reference_after": _iso(first_new.ts - timedelta(milliseconds=1))}
            return update_detector(name, fresh, samples, reset_at, window, min_samples, scorers, now)
        picked, count = [], 0
        for s in reversed(current):
            if count >= window:
                break
            picked.append(s)
            count += int(s.agg["n"])
        current_count = count
        if count >= min_samples:
            centroid, _ = _centroid(picked)
            score, drift = scorers["centroid"](np.asarray(det["reference_centroid"]), centroid)
    elif current:
        values = [s.value for s in current][-window:]
        current_count = len(values)
        if current_count >= min_samples:
            score, drift = scorers[kind](det["reference"], values)

    enough = frozen and current_count >= min_samples
    det.update(current_count=current_count, enough_data=enough, score=score if enough else None,
               drift=bool(enough and drift))
    if det["drift"] and not previous.get("drift"):
        det["last_drift_at"] = _iso(now)
    return det

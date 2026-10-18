"""Topic metrics, read from the scheduler's files on each scrape (recurring-research spec REC-117).

The scheduler is a separate process without an HTTP server, so the backend exports its numbers
through this custom collector and there's one scrape target. Reads are cached for 15 s, so a burst
of scrapes (two workers, several Prometheus servers) costs one read of each file.

- Counters come from `counters.json`, which the scheduler only ever increases, so they never go
  down, also after run records are pruned.
- `webagent_topic_run_seconds` is a real histogram: the scheduler keeps per-bucket counts (not
  cumulative), with the sum and count, in `counters.json["run_seconds"]`; this collector makes them
  cumulative and exports `_bucket`, `_sum` and `_count`. Only completed runs are observed.
- Every family is yielded even with no data, so dashboards always find the names.

Labels carry no topic text: only outcomes, kinds, reasons and states (RNFR-03).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily

from app.core.config import settings
from app.services import topic_files as files

CACHE_SECONDS = 15.0
RUN_OUTCOMES = ("completed", "failed", "skipped")
CHANGE_KINDS = ("new_sources", "summary_changed")
HOLD_REASONS = ("failures", "budget")
TOPIC_STATES = ("active", "paused", "held_failures", "held_budget")
BUDGET_KINDS = ("credits", "dollars")
RUN_SECONDS_BUCKETS = ("30", "60", "120", "180", "300", "600", "900", "1800")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _topic_states() -> Dict[str, int]:
    """Counts of non-deleted topics by effective state (the definition's status, then the hold)."""
    counts = {state: 0 for state in TOPIC_STATES}
    for topic_id in files.list_topic_ids():
        defn = files.read_json(files.definition_path(topic_id)) or {}
        status = defn.get("status", "active")
        if status == "deleted":
            continue
        if status == "paused":
            counts["paused"] += 1
            continue
        hold = (files.read_json(files.state_path(topic_id)) or {}).get("hold")
        key = {"failures": "held_failures", "budget": "held_budget"}.get(hold, "active")
        counts[key] += 1
    return counts


def _snapshot(now: datetime) -> Dict[str, Any]:
    counters = files.read_json(files.counters_path()) or {}
    usage = files.read_json(files.usage_path(now.strftime("%Y-%m"))) or {}
    lock = files.read_json(files.scheduler_lock_path()) or {}
    schedule = getattr(settings, "SCHEDULE", None)
    beat = files.parse_time(lock.get("heartbeat")) if lock.get("owner", "scheduler") == "scheduler" else None
    return {
        "counters": counters,
        "states": _topic_states(),
        "used": {"credits": _number(usage.get("credits")), "dollars": _number(usage.get("dollars"))},
        "limit": {
            "credits": _number(getattr(schedule, "monthly_tavily_credits", 0)),
            "dollars": _number(getattr(schedule, "monthly_together_usd", 0)),
        },
        "heartbeat": beat.timestamp() if beat else None,
        "enabled": bool(lock.get("enabled")) if beat else None,
    }


def _histogram(raw: Any) -> Tuple[List[Tuple[str, float]], float]:
    raw = raw if isinstance(raw, dict) else {}
    per_bucket = raw.get("buckets") if isinstance(raw.get("buckets"), dict) else {}
    running = 0.0
    buckets = []
    for edge in RUN_SECONDS_BUCKETS:
        running += _number(per_bucket.get(edge))
        buckets.append((str(float(edge)), running))     # "30.0", as prometheus_client writes `le`
    running += _number(per_bucket.get("+Inf"))
    buckets.append(("+Inf", max(running, _number(raw.get("count")))))
    return buckets, _number(raw.get("sum"))


class TopicCollector:
    """Prometheus collector of the scheduler's files, cached for 15 s."""

    def __init__(self, cache_seconds: float = CACHE_SECONDS, clock=time.monotonic):
        self.cache_seconds = cache_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._cached: Optional[Dict[str, Any]] = None
        self._at = 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            moment = self.clock()
            if self._cached is None or moment - self._at >= self.cache_seconds:
                try:
                    self._cached = _snapshot(datetime.now(timezone.utc))
                except Exception:  # noqa: BLE001 - a scrape must never fail because of a file
                    self._cached = self._cached or {}
                self._at = moment
            return self._cached

    def describe(self) -> Iterable[Any]:
        # Registering without describe() would call collect() at once, reading files at import.
        return list(self._families({}))

    def collect(self) -> Iterable[Any]:
        return self._families(self.snapshot())

    def _families(self, snap: Dict[str, Any]) -> List[Any]:
        counters = snap.get("counters") or {}

        runs = CounterMetricFamily("webagent_topic_runs", "Scheduled topic runs by outcome", labels=["outcome"])
        raw_runs = counters.get("runs") if isinstance(counters.get("runs"), dict) else {}
        for outcome in sorted(set(RUN_OUTCOMES) | set(raw_runs)):
            if snap:
                runs.add_metric([outcome], _number(raw_runs.get(outcome)))

        changes = CounterMetricFamily("webagent_topic_changes", "Topic runs that changed, by kind",
                                      labels=["kind"])
        raw_changes = counters.get("changes") if isinstance(counters.get("changes"), dict) else {}
        for kind in CHANGE_KINDS:
            if snap:
                changes.add_metric([kind], _number(raw_changes.get(kind)))

        held = CounterMetricFamily("webagent_topic_held", "Topics put on hold, by reason", labels=["reason"])
        raw_holds = counters.get("holds") if isinstance(counters.get("holds"), dict) else {}
        for reason in HOLD_REASONS:
            if snap:
                held.add_metric([reason], _number(raw_holds.get(reason)))

        missed = CounterMetricFamily("webagent_scheduler_missed_slots",
                                     "Schedule slots missed while the scheduler was down")
        if snap:
            missed.add_metric([], _number(counters.get("missed_slots")))

        states = GaugeMetricFamily("webagent_topics", "Topics by effective state", labels=["state"])
        for state, count in (snap.get("states") or {}).items():
            states.add_metric([state], count)

        used = GaugeMetricFamily("webagent_topic_budget_used",
                                 "Scheduled spending this month (credits: Tavily, dollars: Together)",
                                 labels=["kind"])
        limit = GaugeMetricFamily("webagent_topic_budget_limit", "Monthly allowance of scheduled runs",
                                  labels=["kind"])
        for kind in BUDGET_KINDS:
            if snap:
                used.add_metric([kind], (snap.get("used") or {}).get(kind, 0.0))
                limit.add_metric([kind], (snap.get("limit") or {}).get(kind, 0.0))

        heartbeat = GaugeMetricFamily("webagent_scheduler_heartbeat_timestamp",
                                      "Unix time of the scheduler's last heartbeat")
        enabled = GaugeMetricFamily("webagent_scheduler_enabled",
                                    "1 if the scheduler's last heartbeat said scheduling is enabled")
        if snap.get("heartbeat") is not None:
            heartbeat.add_metric([], snap["heartbeat"])
            enabled.add_metric([], 1.0 if snap.get("enabled") else 0.0)

        seconds = HistogramMetricFamily("webagent_topic_run_seconds", "Duration of completed topic runs")
        if snap:
            buckets, total = _histogram(counters.get("run_seconds"))
            seconds.add_metric([], buckets, total)

        return [runs, changes, held, missed, states, used, limit, heartbeat, enabled, seconds]


def register_topic_collector(registry) -> TopicCollector:
    """Register one `TopicCollector` on `registry` (e.g. WEBAGENT_REGISTRY) and return it."""
    collector = TopicCollector()
    registry.register(collector)
    return collector

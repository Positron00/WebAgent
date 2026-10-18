"""The recurring-research scheduler: `python -m app.scheduler` (recurring-research spec REC-104 –
REC-106, REC-111 – REC-113, REC-115, RNFR-01, RNFR-02).

One process. Every 60 s it ticks: it reads the topic definitions (never writing them), keeps each
topic's `state.json`, and starts due runs as ordinary `POST /api/task` calls with a fixed task ID.
Each started run is followed in its own thread, which polls `GET /api/task/{id}` every 15 s for up
to 10 minutes and then hands the result back to `finish_run`. A finished run wakes the main loop at
once, so a freed slot is used without waiting for the next minute (RNFR-01).

`python -m app.scheduler --health` only reads `.scheduler.lock` and exits 0 when its heartbeat is
under 180 s old (the container health check); it takes no lock and starts no loop.

Imports stay light: config, the topic file helpers, `topics` (models and reads), `schedule_math`,
`topic_runs`, `result_store`, `httpx` and the standard library. Never agents, the security guard or
an ML library (RNFR-02). Logs carry topic and run IDs only, never a query, name or report (RNFR-03).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.core.config import settings
from app.services import result_store, schedule_math, topic_runs, topics
from app.services import topic_files as files

logger = logging.getLogger("app.scheduler")

TICK_SECONDS = 60
POLL_SECONDS = 15
POLL_LIMIT_SECONDS = 600
POST_TIMEOUT_SECONDS = 30
GET_TIMEOUT_SECONDS = 10
STALE_SECONDS = 180
SUMMARY_TOKENS = 8000          # phase B's change summary, reserved in the worst case (REC-113)
CREDITS_PER_PASS = 3           # two searches per pass, at most one advanced (REC-113)
DELETE_RETRY = timedelta(hours=1)
RUN_SECONDS_BUCKETS = (30, 60, 120, 180, 300, 600, 900, 1800)


# --- small helpers --------------------------------------------------------------------------------

def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _month(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m")


def _day(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _error_code(response: httpx.Response) -> Optional[str]:
    """The `code` of an error body: top level, under `error` or under `detail`."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    for holder in (body, body.get("error"), body.get("detail")):
        if isinstance(holder, dict) and isinstance(holder.get("code"), str):
            return holder["code"]
    return None


def _usage_numbers(usage: Any) -> Dict[str, int]:
    usage = usage if isinstance(usage, dict) else {}
    out = {}
    for key in ("input_tokens", "output_tokens", "searches", "tavily_credits"):
        try:
            out[key] = max(int(usage.get(key) or 0), 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


def _bump(counter: Dict[str, Any], *path: str, by: float = 1) -> None:
    node = counter
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = node.get(path[-1], 0) + by


# --- the lock (REC-105) ---------------------------------------------------------------------------

def heartbeat_age(lock: Optional[Dict[str, Any]], now: datetime) -> Optional[float]:
    beat = files.parse_time((lock or {}).get("heartbeat"))
    return None if beat is None else (now - beat).total_seconds()


def _process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    except (TypeError, ValueError, OSError):
        return False
    return True


class LockHeld(RuntimeError):
    """Another scheduler (or the backend, REC-114) holds a fresh heartbeat."""


class SchedulerLock:
    """`data/topics/.scheduler.lock`: `{pid, heartbeat, enabled, owner, host, instance}`.

    Taken when there's no fresh heartbeat (under 180 s), or when the fresh one is our own previous
    life (same host and PID, or same host and a dead PID). Writes go under the file's flock, which
    the backend also takes when it stands in for a missing scheduler (REC-114).
    """

    def __init__(self, path=None, now: Callable[[], datetime] = files.utcnow,
                 wait_for_backend: float = STALE_SECONDS, sleep: Callable[[float], None] = time.sleep,
                 wait_for_holder: float = 0.0):
        self.path = path or files.scheduler_lock_path()
        self.now = now
        self.instance = str(uuid.uuid4())
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.wait_for_backend = wait_for_backend
        self.wait_for_holder = wait_for_holder
        self.sleep = sleep
        self.held = False

    def _mine(self, enabled: bool) -> Dict[str, Any]:
        return {"pid": self.pid, "heartbeat": _iso(self.now()), "enabled": bool(enabled),
                "owner": "scheduler", "host": self.host, "instance": self.instance}

    def _ours_from_before(self, lock: Dict[str, Any]) -> bool:
        if lock.get("owner") != "scheduler" or lock.get("host") != self.host:
            return False
        return lock.get("pid") == self.pid or not _process_alive(lock.get("pid"))

    def acquire(self, enabled: bool) -> None:
        waited = 0.0
        while True:
            with files.locked(self.path):
                lock = files.read_json(self.path)
                age = heartbeat_age(lock, self.now())
                fresh = (lock is not None and age is not None and age < STALE_SECONDS
                         and not lock.get("released"))
                if not fresh or self._ours_from_before(lock):
                    if lock is not None and not fresh:
                        logger.info("Scheduler lock is stale (%s s); taking it over",
                                    "?" if age is None else int(age))
                    files.write_json_atomic(self.path, self._mine(enabled))
                    self.held = True
                    return
                owner = lock.get("owner")
            if owner == "backend" and waited < self.wait_for_backend:
                # The backend is finishing a delete for a missing scheduler; it releases soon.
                self.sleep(5)
                waited += 5
                continue
            if owner != "backend" and waited < self.wait_for_holder:
                # After a crash or a redeploy the previous scheduler's heartbeat is still fresh. Wait
                # for it to go stale instead of exiting, so a restart doesn't become a crash loop;
                # a live holder keeps beating, and this one gives up after wait_for_holder.
                if waited == 0:
                    logger.info("Scheduler lock held (heartbeat %s s old); waiting for it to go stale",
                                "?" if age is None else int(age))
                self.sleep(15)
                waited += 15
                continue
            raise LockHeld(f"another {owner or 'scheduler'} holds the lock (heartbeat {int(age)} s old)")

    def beat(self, enabled: bool) -> bool:
        """Refresh the heartbeat. False if someone else has taken the lock meanwhile."""
        with files.locked(self.path):
            lock = files.read_json(self.path)
            if lock is not None and lock.get("instance") != self.instance:
                self.held = False
                return False
            files.write_json_atomic(self.path, self._mine(enabled))
            return True

    def release(self) -> None:
        if not self.held:
            return
        with files.locked(self.path):
            lock = files.read_json(self.path) or {}
            if lock.get("instance") == self.instance:
                # Kept, not removed: the heartbeat ages, so the "Scheduler down" alert still sees a
                # stopped scheduler, while `released` lets the next one start at once.
                lock["released"] = True
                files.write_json_atomic(self.path, lock)
        self.held = False

    def __enter__(self) -> "SchedulerLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def health(now: Optional[datetime] = None, path=None) -> int:
    """`--health`: 0 if the lock's heartbeat is under 180 s old, else 1. Prints one line."""
    now = now or files.utcnow()
    lock = files.read_json(path or files.scheduler_lock_path())
    if lock is None:
        print("unhealthy: no scheduler lock")
        return 1
    age = heartbeat_age(lock, now)
    if age is None:
        print("unhealthy: no heartbeat in the scheduler lock")
        return 1
    if age >= STALE_SECONDS:
        print(f"unhealthy: heartbeat {int(age)} s old")
        return 1
    if lock.get("released"):
        print("unhealthy: the scheduler has stopped")
        return 1
    state = "enabled" if lock.get("enabled") else "disabled"
    print(f"ok: heartbeat {int(age)} s old ({state})")
    return 0


# --- runs in flight -------------------------------------------------------------------------------

@dataclass
class Job:
    topic_id: str
    run_id: str
    task_id: str
    query: str
    started_at: datetime
    deadline: datetime
    phase: str = "post"             # post -> poll
    reconciled: bool = False
    next_poll: Optional[datetime] = None


@dataclass
class Outcome:
    status: str                     # completed | failed
    error_code: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    late_poll: bool = False


class Scheduler:
    """The scheduler's logic. `now`, the HTTP client, threading and sleeping are injectable.

    With `threaded=False` (tests), a started run makes its POST at once in the caller's thread, and
    `pump(now)` performs the polls that are due, so a fake clock drives everything deterministically.
    """

    def __init__(self, client: Optional[httpx.Client] = None, now: Callable[[], datetime] = files.utcnow,
                 config: Any = None, threaded: bool = True, sleep: Optional[Callable[[float], None]] = None,
                 max_follow_up_questions: Optional[int] = None, search_cap: Optional[int] = None,
                 price_input: Optional[float] = None, price_output: Optional[float] = None):
        self.cfg = config or settings.SCHEDULE
        self.client = client or httpx.Client(base_url=settings.BACKEND_URL.rstrip("/"),
                                             timeout=GET_TIMEOUT_SECONDS)
        self.now = now
        self.threaded = threaded
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.sleep = sleep or self._stop.wait
        self._mu = threading.RLock()
        self.jobs: Dict[str, Job] = {}            # run_id -> job, polled now
        self.late: Dict[str, Dict[str, Any]] = {}  # run_id -> {topic_id, task_id, after}: one more poll
        if max_follow_up_questions is None or search_cap is None:
            agents = _agents_config()
            if max_follow_up_questions is None:
                max_follow_up_questions = agents[0]
            if search_cap is None:
                search_cap = agents[1]
        self.max_follow_up_questions = max_follow_up_questions
        self.search_cap = search_cap
        self.price_input = settings.PRICE_INPUT_PER_MILLION if price_input is None else price_input
        self.price_output = settings.PRICE_OUTPUT_PER_MILLION if price_output is None else price_output

    # -- HTTP ----------------------------------------------------------------------------------

    def _headers(self, topic_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"X-WebAgent-Scheduler": settings.SCHEDULER_TOKEN or ""}
        api_key = os.getenv("WEBAGENT_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key
        if topic_id:
            headers["X-WebAgent-Client"] = f"schedule:{topic_id}"
        return headers

    # -- allowance (REC-113) -------------------------------------------------------------------

    def worst_case(self) -> Dict[str, float]:
        credits = self.max_follow_up_questions * CREDITS_PER_PASS
        if self.search_cap and self.search_cap > 0:
            credits = min(credits, self.search_cap * 2)     # at most 2 credits a search
        dollars = (self.cfg.token_budget_default + SUMMARY_TOKENS) * self.price_output / 1e6
        return {"credits": float(credits), "dollars": dollars}

    def load_usage(self, now: datetime) -> Dict[str, Any]:
        usage = files.read_json(files.usage_path(_month(now))) or {}
        usage.setdefault("month", _month(now))
        usage.setdefault("credits", 0)
        usage.setdefault("dollars", 0.0)
        usage.setdefault("runs", 0)
        usage.setdefault("runs_by_day", {})
        return usage

    def save_usage(self, usage: Dict[str, Any]) -> None:
        files.write_json_atomic(files.usage_path(usage["month"]), usage)

    def remaining(self, now: datetime) -> Dict[str, float]:
        usage = self.load_usage(now)
        return {"credits": self.cfg.monthly_tavily_credits - float(usage["credits"]),
                "dollars": self.cfg.monthly_together_usd - float(usage["dollars"])}

    def reserved(self) -> int:
        """Runs whose spending isn't recorded yet: polled now, or waiting for their late poll."""
        return len(self.jobs) + len(self.late)

    def allowance_covers(self, now: datetime, runs: int) -> bool:
        worst, left = self.worst_case(), self.remaining(now)
        return all(left[k] + 1e-9 >= worst[k] * runs for k in ("credits", "dollars"))

    def _add_spending(self, now: datetime, usage: Dict[str, int]) -> None:
        record = self.load_usage(now)
        record["credits"] = int(record["credits"]) + usage["tavily_credits"]
        record["dollars"] = round(float(record["dollars"])
                                  + usage["input_tokens"] * self.price_input / 1e6
                                  + usage["output_tokens"] * self.price_output / 1e6, 6)
        self.save_usage(record)

    # -- counters ------------------------------------------------------------------------------

    def _counters(self) -> Dict[str, Any]:
        return files.read_json(files.counters_path()) or {}

    def _count(self, *path: str, by: float = 1) -> None:
        counters = self._counters()
        _bump(counters, *path, by=by)
        files.write_json_atomic(files.counters_path(), counters)

    def _count_run_seconds(self, seconds: float) -> None:
        counters = self._counters()
        hist = counters.setdefault("run_seconds", {"buckets": {}, "sum": 0.0, "count": 0})
        for edge in RUN_SECONDS_BUCKETS:
            if seconds <= edge:
                _bump(hist, "buckets", str(edge))
                break
        else:
            _bump(hist, "buckets", "+Inf")
        hist["sum"] = round(float(hist.get("sum", 0.0)) + seconds, 3)
        hist["count"] = int(hist.get("count", 0)) + 1
        files.write_json_atomic(files.counters_path(), counters)

    # -- state ---------------------------------------------------------------------------------

    @staticmethod
    def _next(defn: topics.TopicDefinition, after: datetime) -> datetime:
        s = defn.schedule
        return schedule_math.next_slot(s.every, s.time, s.tz, after, s.weekday)

    def _save_state(self, topic_id: str, state: Dict[str, Any]) -> None:
        files.write_json_atomic(files.state_path(topic_id), state)

    def _new_state(self, defn: topics.TopicDefinition, now: datetime) -> Dict[str, Any]:
        return {
            "topic_id": defn.topic_id,
            "next_due": _iso(self._next(defn, now)),
            "last_run": None,
            "failures_in_a_row": 0,
            "hold": None,
            "hold_month": None,
            "skipped": None,
            "run_requests_seen": 0,
            "resumes_seen": defn.resumes,
            "paused_since": None,
            "schedule_seen": defn.schedule.model_dump(),
            "current_run": None,
        }

    def _advance(self, defn: topics.TopicDefinition, state: Dict[str, Any], now: datetime) -> None:
        """Move `next_due` past now when due (no backfill), counting slots missed beyond one."""
        due = files.parse_time(state.get("next_due"))
        if due is None or due > now:
            return
        span = schedule_math.period(defn.schedule.every)
        late = now - due
        if late > span:
            self._count("missed_slots", by=math.floor(late / span))
        state["next_due"] = _iso(self._next(defn, now))

    def _set_hold(self, topic_id: str, state: Dict[str, Any], reason: str, now: datetime) -> None:
        state["hold"] = reason
        state["hold_month"] = _month(now) if reason == "budget" else None
        self._count("holds", reason)
        logger.info("Topic %s held: %s", topic_id, reason)

    def _clear_hold(self, defn: topics.TopicDefinition, state: Dict[str, Any], now: datetime) -> None:
        state["hold"] = None
        state["hold_month"] = None
        state["next_due"] = _iso(self._next(defn, now))   # no burst of old slots after a hold

    # -- deletion (REC-114, REC-115) -----------------------------------------------------------

    def remove_topic_files(self, topic_id: str) -> None:
        """Delete the results of the topic's runs, then its folder (state and run records)."""
        for record in topic_runs.list_records(topic_id):
            task_id = topic_runs.task_id_of(record)
            if task_id:
                result_store.delete_result(task_id)
        shutil.rmtree(files.topic_dir(topic_id), ignore_errors=True)
        logger.info("Removed the files of deleted topic %s", topic_id)

    def request_deletion(self, topic_id: str) -> bool:
        """Ask the backend to delete a topic paused too long (REC-115). True on 204 or 404."""
        try:
            response = self.client.delete(f"/api/topics/{topic_id}", headers=self._headers(),
                                          timeout=GET_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.warning("Deleting long-paused topic %s failed: %s", topic_id, type(exc).__name__)
            return False
        if response.status_code in (200, 202, 204, 404):
            logger.info("Requested deletion of topic %s, paused for %d days", topic_id,
                        self.cfg.paused_topic_days)
            return True
        logger.warning("Deleting long-paused topic %s answered %d", topic_id, response.status_code)
        return False

    # -- the tick (design §9) ------------------------------------------------------------------

    def tick(self, now: Optional[datetime] = None) -> None:
        """One pass of design §9 steps 1–8 and 10 (step 9 happens as runs finish)."""
        now = now or self.now()
        deletions: List[str] = []
        late_polls: List[str] = []
        with self._mu:
            try:
                deletions, late_polls = self._plan_and_start(now)
            except OSError as exc:              # disk full and the like: skip this tick (design §4)
                logger.error("Scheduler tick failed: %s", type(exc).__name__)
                return
        for run_id in late_polls:
            self._late_poll(run_id, now)
        for topic_id in deletions:
            if self.request_deletion(topic_id):
                with self._mu:
                    state = topics.load_state(topic_id)
                    if state:
                        state["deletion_requested_at"] = _iso(now)
                        self._save_state(topic_id, state)

    def _plan_and_start(self, now: datetime):
        definitions = topics.list_definitions(include_deleted=True)
        busy = {job.topic_id for job in self.jobs.values()}
        candidates = []
        deletions = []
        for defn in definitions:
            topic_id = defn.topic_id
            # 2. tombstones
            if defn.status == "deleted":
                if files.topic_dir(topic_id).exists():
                    self.remove_topic_files(topic_id)
                continue
            # 1. state
            state = topics.load_state(topic_id)
            changed = False
            if not state:
                state = self._new_state(defn, now)
                changed = True
            # 3. paused
            if defn.status == "paused":
                if not state.get("paused_since"):
                    state["paused_since"] = _iso(now)
                    changed = True
                since = files.parse_time(state.get("paused_since"))
                asked = files.parse_time(state.get("deletion_requested_at"))
                if since and now - since >= timedelta(days=self.cfg.paused_topic_days) \
                        and (asked is None or now - asked >= DELETE_RETRY):
                    deletions.append(topic_id)
                if changed:
                    self._save_state(topic_id, state)
                continue
            if state.get("paused_since"):
                state["paused_since"] = None
                state.pop("deletion_requested_at", None)
                state["next_due"] = _iso(self._next(defn, now))   # resume from the next slot
                changed = True
            # 4. resume
            if defn.resumes > int(state.get("resumes_seen") or 0):
                if state.get("hold") == "failures":
                    self._clear_hold(defn, state, now)
                state["failures_in_a_row"] = 0
                state["resumes_seen"] = defn.resumes
                changed = True
            # 5. schedule changes
            schedule = defn.schedule.model_dump()
            if state.get("schedule_seen") != schedule:
                state["schedule_seen"] = schedule
                state["next_due"] = _iso(self._next(defn, now))
                changed = True
            # 6. budget hold
            if state.get("hold") == "budget" and (
                    state.get("hold_month") != _month(now)
                    or self.allowance_covers(now, self.reserved() + 1)):
                self._clear_hold(defn, state, now)
                logger.info("Topic %s budget hold cleared", topic_id)
                changed = True
            if changed:
                self._save_state(topic_id, state)
            # 7. candidates
            due = files.parse_time(state.get("next_due"))
            requested = defn.run_requests > int(state.get("run_requests_seen") or 0)
            is_due = due is not None and due <= now
            if state.get("hold") or topic_id in busy or not (is_due or requested):
                continue
            candidates.append((due if is_due else now, defn, state))

        candidates.sort(key=lambda c: c[0])
        # 8. start
        for _, defn, state in candidates:
            if len(self.jobs) >= self.cfg.max_concurrent_runs:
                break
            usage = self.load_usage(now)
            if int(usage["runs_by_day"].get(_day(now), 0)) >= self.cfg.max_runs_per_day:
                state["skipped"] = {"reason": "daily_cap", "at": _iso(now)}
                self._advance(defn, state, now)
                state["run_requests_seen"] = defn.run_requests
                self._save_state(defn.topic_id, state)
                self._count("runs", "skipped")
                logger.info("Topic %s skipped: daily cap", defn.topic_id)
                continue
            if not self.allowance_covers(now, self.reserved() + 1):
                self._hold_everything_for_budget(definitions, now)
                break
            self._start(defn, state, now, usage)
        late = [run_id for run_id, info in self.late.items() if info["after"] <= now]
        return deletions, late

    def _hold_everything_for_budget(self, definitions, now: datetime) -> None:
        logger.warning("Scheduled-run allowance can't cover another run; holding every topic")
        for defn in definitions:
            if defn.status == "deleted":
                continue
            state = topics.load_state(defn.topic_id) or self._new_state(defn, now)
            if state.get("hold"):
                continue
            self._set_hold(defn.topic_id, state, "budget", now)
            self._save_state(defn.topic_id, state)

    def _start(self, defn: topics.TopicDefinition, state: Dict[str, Any], now: datetime,
               usage: Dict[str, Any]) -> None:
        run_id = str(uuid.uuid4())
        due = files.parse_time(state.get("next_due"))
        trigger = "slot" if due is not None and due <= now else "request"
        # The state first: a crash after this loses a slot, never runs it twice (REC-106).
        self._advance(defn, state, now)
        state["run_requests_seen"] = defn.run_requests
        state["skipped"] = None
        state["current_run"] = run_id
        self._save_state(defn.topic_id, state)
        record = topic_runs.new_record(defn.topic_id, run_id, now, trigger)
        topic_runs.save(record)
        usage["runs"] = int(usage["runs"]) + 1
        _bump(usage["runs_by_day"], _day(now))
        self.save_usage(usage)
        job = Job(topic_id=defn.topic_id, run_id=run_id, task_id=record["task_id"], query=defn.query,
                  started_at=now, deadline=now + timedelta(seconds=POLL_LIMIT_SECONDS))
        self.jobs[run_id] = job
        logger.info("Starting run %s of topic %s (%s)", run_id, defn.topic_id, trigger)
        self._launch(job)

    # -- running a job -------------------------------------------------------------------------

    def _launch(self, job: Job) -> None:
        if self.threaded:
            threading.Thread(target=self._worker, args=(job,), name=f"run-{job.run_id[:8]}",
                             daemon=True).start()
        else:
            self.step(job)

    def _worker(self, job: Job) -> None:
        try:
            while not self.step(job):
                if self._stop.is_set():
                    return                 # left `running`; reconciled on the next start
                self.sleep(POLL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - a run thread must never die silently
            logger.error("Run %s thread failed: %s", job.run_id, type(exc).__name__)
            self.finish_run(job, Outcome("failed", "scheduler_error"))

    def pump(self, now: Optional[datetime] = None) -> int:
        """Unthreaded mode: perform the polls that are due. Returns how many runs finished."""
        now = now or self.now()
        finished = 0
        for job in list(self.jobs.values()):
            if job.next_poll is not None and job.next_poll <= now and self.step(job):
                finished += 1
        return finished

    def step(self, job: Job) -> bool:
        """The job's next action: the POST, or one poll. True when the job has finished."""
        now = self.now()
        if job.phase == "post":
            outcome = self._post(job)
            if outcome is None:
                job.phase = "poll"
                job.next_poll = now + timedelta(seconds=POLL_SECONDS)
                return False
            self.finish_run(job, outcome)
            return True
        outcome = self._poll(job, now)
        if outcome is None:
            if now >= job.deadline:
                self.finish_run(job, Outcome("failed", "poll_timeout", late_poll=True))
                return True
            job.next_poll = now + timedelta(seconds=POLL_SECONDS)
            return False
        self.finish_run(job, outcome)
        return True

    def _post(self, job: Job) -> Optional[Outcome]:
        body = {"query": job.query, "mode": "always", "origin": "schedule", "topic_id": job.topic_id,
                "task_id": job.task_id, "token_budget": self.cfg.token_budget_default}
        try:
            response = self.client.post("/api/task", json=body, headers=self._headers(job.topic_id),
                                        timeout=POST_TIMEOUT_SECONDS)
        except httpx.TimeoutException:
            # It may have been accepted: the first poll tells (404 means it never started).
            logger.warning("Run %s: starting the task timed out; polling", job.run_id)
            return None
        except httpx.HTTPError as exc:
            logger.warning("Run %s: backend unreachable (%s)", job.run_id, type(exc).__name__)
            return Outcome("failed", "backend_unreachable")
        status = response.status_code
        if status in (200, 201, 202):
            self._mark_running(job)
            return None
        code = _error_code(response)
        if status == 409 and code == "task_exists":
            self._mark_running(job)
            return None
        if status in (403, 409):
            return Outcome("failed", "intake_refused")
        if status == 429:
            return Outcome("failed", "rate_limited")
        if status >= 500:
            return Outcome("failed", "backend_unreachable")
        logger.warning("Run %s: the backend refused the task (%d)", job.run_id, status)
        return Outcome("failed", "rejected")

    def _mark_running(self, job: Job) -> None:
        with self._mu:
            record = topic_runs.load(job.topic_id, job.run_id)
            if record is not None and record.get("status") == "starting":
                record["status"] = "running"
                topic_runs.save(record)

    def _get_task(self, task_id: str):
        """('done', body) | ('processing', body) | ('unknown', None) | ('error', None)."""
        try:
            response = self.client.get(f"/api/task/{task_id}", headers=self._headers(),
                                       timeout=GET_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return "error", None
        if response.status_code == 404:
            return "unknown", None
        if response.status_code != 200:
            return "error", None
        try:
            body = response.json()
        except ValueError:
            return "error", None
        if not isinstance(body, dict):
            return "error", None
        status = body.get("status")
        if status == "failed" and body.get("result") is None and \
                str(body.get("error") or "").lower() == "task not found":
            return "unknown", None          # the backend's older not-found answer
        if status in ("completed", "failed"):
            return "done", body
        return "processing", body

    def _poll(self, job: Job, now: datetime) -> Optional[Outcome]:
        kind, body = self._get_task(job.task_id)
        if kind == "unknown":
            return Outcome("failed", "not_started")
        if kind != "done":
            return None                    # processing, or a transient error: poll again
        result = body.get("result") if isinstance(body.get("result"), dict) else None
        if body.get("status") == "completed":
            return Outcome("completed", None, result)
        code = body.get("error_code") or (result or {}).get("error_code") or "task_failed"
        return Outcome("failed", str(code), result)

    # -- step 9: a finished run ----------------------------------------------------------------

    def finish_run(self, job: Job, outcome: Outcome) -> None:
        now = self.now()
        with self._mu:
            self.jobs.pop(job.run_id, None)
            try:
                self._finish(job, outcome, now)
            except Exception as exc:  # noqa: BLE001 - never leave the job half-finished twice
                logger.error("Recording run %s failed: %s", job.run_id, type(exc).__name__)
        self._wake.set()

    def _finish(self, job: Job, outcome: Outcome, now: datetime) -> None:
        result = outcome.result or {}
        usage = _usage_numbers(result.get("usage"))
        self._add_spending(now, usage)       # always counted, even for a deleted topic (REC-106)
        seconds = max((now - job.started_at).total_seconds(), 0.0)
        self._count("runs", outcome.status)
        if outcome.status == "completed":
            self._count_run_seconds(seconds)

        defn = topics.load_definition(job.topic_id)
        if defn is None or defn.status == "deleted":
            result_store.delete_result(job.task_id)
            logger.info("Run %s finished after topic %s was deleted", job.run_id, job.topic_id)
            return

        record = topic_runs.load(job.topic_id, job.run_id) or topic_runs.new_record(
            job.topic_id, job.run_id, job.started_at, "slot")
        record["status"] = outcome.status
        record["error_code"] = outcome.error_code
        record["completed_at"] = _iso(now)
        record["duration_seconds"] = round(seconds, 3)
        record["usage"] = usage if outcome.result is not None else None
        if outcome.late_poll:
            record["late_poll"] = True
            self.late[job.run_id] = {"topic_id": job.topic_id, "task_id": job.task_id,
                                     "after": now + timedelta(seconds=TICK_SECONDS)}
        if outcome.status == "completed":
            record["title"] = result.get("title") if isinstance(result.get("title"), str) else None
            content = result.get("content")
            record["report"] = content if isinstance(content, str) else None
            record["sources"] = topic_runs.clean_sources(result.get("sources"))
            earlier = [r for r in topic_runs.list_records(job.topic_id) if r.get("run_id") != job.run_id]
            record["change"] = topic_runs.diff(record["sources"], earlier)
            if record["change"]["new_sources"] and not record["change"]["first"]:
                self._count("changes", "new_sources")
        topic_runs.save(record)

        state = topics.load_state(job.topic_id) or self._new_state(defn, now)
        if state.get("current_run") == job.run_id:
            state["current_run"] = None
        state["last_run"] = {
            "run_id": job.run_id, "status": outcome.status, "error_code": outcome.error_code,
            "completed_at": record["completed_at"],
            "new_sources": len((record.get("change") or {}).get("new_sources") or []),
        }
        if outcome.status == "completed":
            state["failures_in_a_row"] = 0
        else:
            state["failures_in_a_row"] = int(state.get("failures_in_a_row") or 0) + 1
            if state["failures_in_a_row"] >= self.cfg.pause_after_failures \
                    and state.get("hold") != "failures":
                self._set_hold(job.topic_id, state, "failures", now)
        self._save_state(job.topic_id, state)
        logger.info("Run %s of topic %s %s%s", job.run_id, job.topic_id, outcome.status,
                    f" ({outcome.error_code})" if outcome.error_code else "")

        for old in topic_runs.prune(job.topic_id, self.cfg.history_runs_kept):
            task_id = topic_runs.task_id_of(old)
            if task_id:
                result_store.delete_result(task_id)
            self.late.pop(old.get("run_id"), None)

    def _late_poll(self, run_id: str, now: datetime) -> None:
        """One more poll of a run that timed out, so the spending of a task that ran is counted."""
        info = self.late.get(run_id)
        if info is None:
            return
        kind, body = self._get_task(info["task_id"])
        with self._mu:
            self.late.pop(run_id, None)
            if kind != "done":
                logger.info("Run %s still unfinished at its late poll", run_id)
                return
            result = body.get("result") if isinstance(body.get("result"), dict) else {}
            usage = _usage_numbers(result.get("usage"))
            self._add_spending(now, usage)
            record = topic_runs.load(info["topic_id"], run_id)
            if record is not None:
                record["usage"] = usage
                record["late_poll"] = False
                record["late_status"] = body.get("status")
                topic_runs.save(record)

    # -- restart (REC-106) ---------------------------------------------------------------------

    def reconcile(self) -> int:
        """Pick up runs left `starting` or `running` by a previous process. Returns how many."""
        now = self.now()
        found = 0
        with self._mu:
            for defn in topics.list_definitions(include_deleted=True):
                topic_id = defn.topic_id
                if not files.topic_dir(topic_id).is_dir():
                    continue
                state = topics.load_state(topic_id)
                open_ids = set()
                for record in topic_runs.list_records(topic_id):
                    if record.get("status") not in topic_runs.OPEN_STATUSES:
                        continue
                    run_id = record["run_id"]
                    open_ids.add(run_id)
                    if not files.is_uuid(run_id) or topic_runs.task_id_of(record) is None:
                        continue
                    started = files.parse_time(record.get("started_at")) or now
                    job = Job(topic_id=topic_id, run_id=run_id, task_id=topic_runs.task_id_of(record),
                              query="", started_at=started,
                              deadline=now + timedelta(seconds=POLL_LIMIT_SECONDS),
                              phase="poll", reconciled=True, next_poll=now)
                    self.jobs[run_id] = job
                    found += 1
                current = (state or {}).get("current_run")
                if current and current not in open_ids and topic_runs.load(topic_id, current) is None \
                        and files.is_uuid(current):
                    # Crashed between the state and the record: the task was never requested.
                    record = topic_runs.new_record(topic_id, current, now, "slot")
                    topic_runs.save(record)
                    job = Job(topic_id=topic_id, run_id=current, task_id=record["task_id"], query="",
                              started_at=now, deadline=now, phase="poll", reconciled=True)
                    self.jobs[current] = job
                    self.finish_run(job, Outcome("failed", "not_started"))
        if found:
            logger.info("Reconciling %d unfinished runs", found)
        for job in list(self.jobs.values()):
            if job.reconciled:
                if self.threaded:
                    threading.Thread(target=self._worker, args=(job,), daemon=True).start()
                else:
                    self.step(job)
        return found

    # -- the process ---------------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def serve(self, lock: SchedulerLock) -> None:
        enabled = bool(self.cfg.enabled)
        if enabled:
            self.reconcile()
        while not self._stop.is_set():
            if not lock.beat(enabled):
                logger.error("Another process took the scheduler lock; stopping")
                return
            if enabled:
                self.tick()
            self._wake.wait(timeout=TICK_SECONDS - (time.time() % TICK_SECONDS) or TICK_SECONDS)
            self._wake.clear()


def _agents_config():
    """(max_follow_up_questions, max_searches_per_task) from the agents config."""
    try:
        from app.core.loadEnvYAML import get_agents_config

        agents = get_agents_config()
        return (int(agents.senior_research.max_follow_up_questions),
                int(agents.web_research.max_searches_per_task))
    except Exception:  # noqa: BLE001 - fall back to the config defaults
        return 3, 5


# --- startup checks and main ----------------------------------------------------------------------

def checks(client: httpx.Client, sleep: Callable[[float], None] = time.sleep, tries: int = 3) -> None:
    """When enabled: the token is set and the backend's /health answers. Exits non-zero otherwise."""
    if not settings.SCHEDULE.enabled:
        logger.info("Scheduling is disabled (schedule.enabled is false); idling with a heartbeat")
        return
    if not settings.SCHEDULER_TOKEN:
        logger.error("Scheduling is enabled but WEBAGENT_SCHEDULER_TOKEN is not set")
        raise SystemExit(2)
    for attempt in range(1, tries + 1):
        try:
            if client.get("/health", timeout=GET_TIMEOUT_SECONDS).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if attempt < tries:
            sleep(5)
    logger.error("The backend at WEBAGENT_BACKEND_URL is not reachable (/health, %d tries)", tries)
    raise SystemExit(3)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.scheduler", description=__doc__.split("\n")[0])
    parser.add_argument("--health", action="store_true", help="check the heartbeat and exit")
    args = parser.parse_args(argv)
    if args.health:
        return health()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = httpx.Client(base_url=settings.BACKEND_URL.rstrip("/"), timeout=GET_TIMEOUT_SECONDS)
    checks(client)
    enabled = bool(settings.SCHEDULE.enabled)
    # Wait up to the stale age plus a minute: long enough for a dead holder's heartbeat to expire.
    lock = SchedulerLock(wait_for_holder=STALE_SECONDS + 60)
    try:
        lock.acquire(enabled)
    except LockHeld as exc:
        logger.error("Not starting: %s", exc)
        return 1
    scheduler = Scheduler(client=client)
    try:
        with lock:
            scheduler.serve(lock)
    except KeyboardInterrupt:
        scheduler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Topic API (recurring-research spec REC-100 – REC-103, REC-114).

Definitions are written only here, always under the definition's flock (two workers, REC-101). The
scheduler's files (state, run records) are only read. Logs carry IDs and lengths, never the query,
name or addresses (RNFR-03). Phase B adds the scheduler-only `summarize` and `notify` routes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.api.intake import refusal_body
from app.core.api_errors import error_body
from app.core.config import settings
from app.core.log_context import describe_text
from app.models.task import WorkflowState
from app.security.errors import GuardStop
from app.security.hop_guard import check_hop
from app.security.types import ContextItem
from app.services import notify_email
from app.services import topic_files as files
from app.services import topics
from app.services.topics import Schedule, TopicDefinition

logger = logging.getLogger(__name__)
router = APIRouter()


# --- request bodies --------------------------------------------------------------------------------

class NotifySettings(BaseModel):
    """Notify settings a caller may set. Subscribers are added one by one in phase B."""

    model_config = ConfigDict(extra="forbid")
    when: Literal["changes", "always"] = "changes"


class TopicCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=2000)
    schedule: Schedule
    notify: NotifySettings = Field(default_factory=NotifySettings)
    created_from_task: Optional[str] = None

    @field_validator("name", "query")
    @classmethod
    def _text(cls, value: str, info) -> str:
        value = topics._no_control(value, info.field_name)
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("created_from_task")
    @classmethod
    def _task(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not files.is_uuid(value):
            raise ValueError("must be a UUID")
        return value


class TopicPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    schedule: Optional[Schedule] = None
    notify: Optional[NotifySettings] = None
    status: Optional[Literal["active", "paused"]] = None
    add_subscriber: Optional[str] = None       # phase B (REC-133): an unconfirmed entry and one confirmation
    remove_subscriber: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = topics._no_control(value, "name")
        if not value:
            raise ValueError("name must not be empty")
        return value

    @field_validator("add_subscriber", "remove_subscriber")
    @classmethod
    def _address(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        # The same rules as stored subscribers: one @, a domain with a dot, at most 254 characters.
        value = value.strip().lower()
        if len(value) > 254 or not topics._EMAIL.match(value) or topics._CONTROL.search(value):
            raise ValueError("not a valid email address")
        return value


# --- helpers ---------------------------------------------------------------------------------------

class _Gone(Exception):
    """The topic is missing or deleted."""


class _TopicLimit(Exception):
    """Activating the topic would pass `schedule.max_active_topics`."""


class _SubscriberLimit(Exception):
    """Adding the address would pass `notifications.email.max_subscribers_per_topic`."""


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Topic not found")


def _topic_limit_response() -> JSONResponse:
    limit = settings.SCHEDULE.max_active_topics
    body = error_body("topic_limit", f"At most {limit} topics can be active")
    return JSONResponse(status_code=409, content={**body, "code": "topic_limit", "limit": limit})


def _cap_lock():
    """One lock for every change that can raise the active count, so two workers can't both pass the cap."""
    return files.locked(files.topics_dir() / ".active")


def _live(topic_id: str) -> TopicDefinition:
    if not files.is_uuid(topic_id):
        raise _not_found()
    defn = topics.load_definition(topic_id)
    if defn is None or defn.status == "deleted":
        raise _not_found()
    return defn


def _update(topic_id: str, change, *, activates: bool = False) -> TopicDefinition:
    """`topics.update_definition` that refuses deleted topics and, when `activates`, checks the cap."""
    def guarded(defn: TopicDefinition) -> None:
        if defn.status == "deleted":
            raise _Gone()
        change(defn)

    def write() -> Optional[TopicDefinition]:
        return topics.update_definition(topic_id, guarded)

    try:
        if activates:
            with _cap_lock():
                if topics.active_count(exclude=topic_id) >= settings.SCHEDULE.max_active_topics:
                    current = topics.load_definition(topic_id)
                    if current is not None and current.status != "deleted":
                        raise _TopicLimit()
                updated = write()
        else:
            updated = write()
    except _Gone:
        updated = None
    if updated is None:
        raise _not_found()
    return updated


def _run_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    change = record.get("change") or {}
    summary: Dict[str, Any] = {
        "first": bool(change.get("first", False)),
        "new_sources": topics.new_source_count(record) or 0,
        "changed": change.get("changed"),
    }
    if "summary" in change:
        summary["summary"] = change.get("summary")
    return {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "error_code": record.get("error_code"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "change": summary,
    }


def _last_run(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    return {key: record.get(key) for key in ("run_id", "status", "error_code", "started_at", "completed_at")}


def _schedule_view(defn: TopicDefinition) -> Dict[str, Any]:
    """Effective state and schedule fields read from the scheduler's files."""
    state = topics.load_state(defn.topic_id)
    runs = topics.list_runs(defn.topic_id, limit=1)
    latest = runs[0] if runs else None
    return {
        "state": topics.effective_state(defn, state),
        "next_due": state.get("next_due"),
        "last_run": _last_run(latest),
        "last_new_sources": topics.new_source_count(latest),
        "skipped": state.get("skipped"),
    }


def _list_row(defn: TopicDefinition) -> Dict[str, Any]:
    return {
        "topic_id": defn.topic_id,
        "name": defn.name,
        "query": defn.query,   # instance-wide topics (D2)
        "schedule": defn.schedule.model_dump(mode="json"),
        "notify": {"when": defn.notify.when},
        "subscriber_count": len(defn.notify.subscribers),
        "status": defn.status,
        "created_at": defn.created_at.isoformat(),
        **_schedule_view(defn),
    }


def _intake_check(query: str, topic_id: str) -> Optional[JSONResponse]:
    """Checkpoint 1 on the topic's query, as POST /api/guard/input does (REC-103).

    In act mode, consent, decline and escalate refuse the topic with 403 and the refusal body: a
    scheduled run has nobody to give consent. Shadow mode records the decision and allows.
    """
    decision_id = f"topic-{topic_id}"
    state = WorkflowState(
        query=query,
        context={"task_id": decision_id, "origin": "topic_create", "topic_id": topic_id},
    )
    try:
        check_hop("topic_create", [ContextItem("user_turn", query)], state, chat_id=decision_id)
    except GuardStop as exc:
        decision = exc.decision
        logger.info("Topic %s refused at intake (%s, query %s)", topic_id, decision.outcome, describe_text(query))
        body = {**refusal_body(decision), "decision_id": decision_id}
        return JSONResponse(status_code=403, content=body)
    return None


# --- routes ----------------------------------------------------------------------------------------

@router.get("")
async def list_topics() -> Dict[str, Any]:
    def rows():
        return [_list_row(d) for d in topics.list_definitions()]

    return {"topics": await run_in_threadpool(rows)}


@router.post("", status_code=201)
async def create_topic(req: TopicCreate):
    if topics.active_count() >= settings.SCHEDULE.max_active_topics:
        return _topic_limit_response()

    topic_id = topics.new_topic_id()
    refused = await run_in_threadpool(_intake_check, req.query, topic_id)
    if refused is not None:
        return refused

    defn = TopicDefinition(
        topic_id=topic_id,
        name=req.name,
        query=req.query,
        schedule=req.schedule,
        notify=topics.Notify(when=req.notify.when),
        status="active",
        created_at=files.utcnow(),
        created_from_task=req.created_from_task,
    )

    def save() -> bool:
        with _cap_lock():   # checked again: the guard took time and another worker may have created one
            if topics.active_count() >= settings.SCHEDULE.max_active_topics:
                return False
            topics.save_new_definition(defn)
            return True

    if not await run_in_threadpool(save):
        return _topic_limit_response()
    logger.info("Topic %s created (name %s, query %s)", topic_id, describe_text(req.name), describe_text(req.query))
    return defn.model_dump(mode="json")


@router.get("/{topic_id}")
async def get_topic(topic_id: str) -> Dict[str, Any]:
    defn = _live(topic_id)
    return {**defn.model_dump(mode="json"), **await run_in_threadpool(_schedule_view, defn)}


def _email_error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={**error_body(code, message), "code": code, **extra})


@router.patch("/{topic_id}")
async def patch_topic(topic_id: str, req: TopicPatch):
    _live(topic_id)
    fields = req.model_dump(exclude_unset=True)
    adding = req.add_subscriber
    if adding is not None and not notify_email.email_enabled():
        return _email_error(409, "email_disabled", "Email notifications are turned off on this instance")
    limit = settings.NOTIFICATIONS.email.max_subscribers_per_topic
    confirm_needed: List[bool] = []

    def change(defn: TopicDefinition) -> None:
        if req.name is not None:
            defn.name = req.name
        if req.schedule is not None:
            defn.schedule = req.schedule
        if req.notify is not None and "when" in fields.get("notify", {}):
            defn.notify.when = req.notify.when
        if req.status is not None:
            defn.status = req.status
        if req.remove_subscriber is not None:
            defn.notify.subscribers = [s for s in defn.notify.subscribers if s.email != req.remove_subscriber]
        if adding is not None:
            now = files.utcnow()
            # Expired unconfirmed entries don't hold a place (REC-115).
            defn.notify.subscribers = [
                s for s in defn.notify.subscribers if not notify_email.expired_unconfirmed(s, now)
            ]
            existing = next((s for s in defn.notify.subscribers if s.email == adding), None)
            if existing is not None:
                if existing.confirmed_at is None:   # asked again: one more confirmation (capped per day)
                    existing.added_at = now
                    confirm_needed.append(True)
                return
            if len(defn.notify.subscribers) >= limit:
                raise _SubscriberLimit()
            defn.notify.subscribers.append(topics.Subscriber(email=adding, added_at=now))
            confirm_needed.append(True)

    try:
        updated = await run_in_threadpool(_update, topic_id, change, activates=req.status == "active")
    except _TopicLimit:
        return _topic_limit_response()
    except _SubscriberLimit:
        return _email_error(409, "subscriber_limit", f"At most {limit} addresses per topic", limit=limit)
    logger.info("Topic %s updated (%s)", topic_id, ",".join(sorted(fields)) or "nothing")
    body = {**updated.model_dump(mode="json"), **await run_in_threadpool(_schedule_view, updated)}
    if adding is not None:
        body["confirmation"] = (
            await run_in_threadpool(notify_email.send_confirmation, topic_id, updated.name, adding)
            if confirm_needed else "not_needed"
        )
    return body


@router.delete("/{topic_id}", status_code=204)
async def delete_topic(topic_id: str) -> Response:
    _live(topic_id)

    def tombstone(defn: TopicDefinition) -> None:
        # The tombstone keeps only what the cleanup needs: no question, name or addresses stay on disk
        # while the files are removed (RNFR-03).
        defn.status = "deleted"
        defn.name = "deleted"
        defn.query = "deleted"
        defn.notify.subscribers = []

    await run_in_threadpool(_update, topic_id, tombstone)
    logger.info("Topic %s deleted (tombstone)", topic_id)
    # Finish at once when there's no live scheduler; otherwise the scheduler and the daily cleanup do.
    await run_in_threadpool(topics.cleanup_deleted, None, [topic_id])
    return Response(status_code=204)


@router.post("/{topic_id}/run", status_code=202)
async def run_topic(topic_id: str) -> Dict[str, Any]:
    _live(topic_id)

    def bump(defn: TopicDefinition) -> None:
        defn.run_requests += 1

    updated = await run_in_threadpool(_update, topic_id, bump)
    logger.info("Topic %s run requested (%d)", topic_id, updated.run_requests)
    return {"run_requests": updated.run_requests}


@router.post("/{topic_id}/resume", status_code=202)
async def resume_topic(topic_id: str):
    _live(topic_id)

    def resume(defn: TopicDefinition) -> None:
        defn.status = "active"
        defn.resumes += 1

    try:
        updated = await run_in_threadpool(_update, topic_id, resume, activates=True)
    except _TopicLimit:
        return _topic_limit_response()
    logger.info("Topic %s resumed (%d)", topic_id, updated.resumes)
    return {"status": updated.status, "resumes": updated.resumes}


@router.get("/{topic_id}/runs")
async def list_topic_runs(topic_id: str) -> Dict[str, Any]:
    _live(topic_id)
    records = await run_in_threadpool(topics.list_runs, topic_id, settings.SCHEDULE.history_runs_kept)
    return {"runs": [_run_summary(r) for r in records]}


@router.get("/{topic_id}/runs/{run_id}")
async def get_topic_run(topic_id: str, run_id: str) -> Dict[str, Any]:
    _live(topic_id)
    record = await run_in_threadpool(topics.load_run, topic_id, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    record.setdefault("run_id", run_id)
    return record

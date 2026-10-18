"""Topics: models, validation, reading and the backend's definition writes (recurring-research spec
REC-100 – REC-102, REC-114).

The scheduler imports this module for the models and the read helpers only; it never calls the
`*_definition` writers (REC-101). Keep imports light (RNFR-02).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services import topic_files as files

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
EffectiveState = Literal["active", "paused", "held_failures", "held_budget", "deleted"]


def _no_control(value: str, field: str) -> str:
    value = value.strip()
    if _CONTROL.search(value):
        raise ValueError(f"{field} must not contain control characters")
    return value


class Schedule(BaseModel):
    every: Literal["day", "week"]
    weekday: Optional[Weekday] = None
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    tz: str

    @field_validator("tz")
    @classmethod
    def _zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("unknown time zone") from None
        return value

    @model_validator(mode="after")
    def _weekday(self) -> "Schedule":
        if self.every == "week" and not self.weekday:
            raise ValueError("a weekly schedule needs a weekday")
        if self.every == "day":
            self.weekday = None
        return self


class Subscriber(BaseModel):
    email: str
    added_at: datetime
    confirmed_at: Optional[datetime] = None

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) > 254 or not _EMAIL.match(value) or _CONTROL.search(value):
            raise ValueError("not a valid email address")
        return value


class Notify(BaseModel):
    when: Literal["changes", "always"] = "changes"
    subscribers: List[Subscriber] = Field(default_factory=list)


class TopicDefinition(BaseModel):
    topic_id: str
    name: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=2000)
    schedule: Schedule
    notify: Notify = Field(default_factory=Notify)
    status: Literal["active", "paused", "deleted"] = "active"
    created_at: datetime
    created_from_task: Optional[str] = None
    run_requests: int = 0
    resumes: int = 0

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _no_control(value, "name")

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        return _no_control(value, "query")

    @field_validator("topic_id", "created_from_task")
    @classmethod
    def _uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not files.is_uuid(value):
            raise ValueError("must be a UUID")
        return value


def new_topic_id() -> str:
    return str(uuid.uuid4())


# --- reading (backend and scheduler) ------------------------------------------------------------

def load_definition(topic_id: str) -> Optional[TopicDefinition]:
    data = files.read_json(files.definition_path(topic_id))
    if data is None:
        return None
    try:
        return TopicDefinition.model_validate(data)
    except ValueError:
        return None


def list_definitions(include_deleted: bool = False) -> List[TopicDefinition]:
    out = []
    for topic_id in files.list_topic_ids():
        defn = load_definition(topic_id)
        if defn is not None and (include_deleted or defn.status != "deleted"):
            out.append(defn)
    return out


def load_state(topic_id: str) -> Dict[str, Any]:
    return files.read_json(files.state_path(topic_id)) or {}


def effective_state(defn: TopicDefinition, state: Optional[Dict[str, Any]] = None) -> EffectiveState:
    """The state shown to people: the definition's status first, then the scheduler's hold."""
    if defn.status in ("deleted", "paused"):
        return defn.status
    hold = (state or {}).get("hold")
    if hold == "failures":
        return "held_failures"
    if hold == "budget":
        return "held_budget"
    return "active"


def active_count(exclude: Optional[str] = None) -> int:
    return sum(1 for d in list_definitions() if d.status == "active" and d.topic_id != exclude)


# --- writing (backend only, REC-101) -------------------------------------------------------------

def save_new_definition(defn: TopicDefinition) -> None:
    path = files.definition_path(defn.topic_id)
    with files.locked(path):
        files.write_json_atomic(path, defn.model_dump(mode="json"))


def update_definition(topic_id: str, change: Callable[[TopicDefinition], None]) -> Optional[TopicDefinition]:
    """Read, apply `change`, validate and write, under the definition's lock. None if missing."""
    path = files.definition_path(topic_id)
    with files.locked(path):
        defn = load_definition(topic_id)
        if defn is None:
            return None
        change(defn)
        defn = TopicDefinition.model_validate(defn.model_dump())
        files.write_json_atomic(path, defn.model_dump(mode="json"))
        return defn

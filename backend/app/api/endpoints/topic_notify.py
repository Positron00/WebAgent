"""`POST /api/topics/{id}/notify`, called by the scheduler only (recurring-research spec REC-131, REC-135,
REC-137, REC-138; the contract is in the low-level design §12a).

Body `{kind: "run" | "hold" | "deletion", run_id?}`; response `{sent, skipped, reason?}`. `run` sends the
run's notice (or summary, REC-136) to confirmed subscribers; `hold` and `deletion` send those notices.
Unconfirmed addresses never get anything (REC-133). Any backend-side problem is a 200 with `reason`:
a run never fails because of email.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.concurrency import run_in_threadpool

from app.core.scheduler_auth import require_scheduler
from app.services import notify_email, topics
from app.services import topic_files as files

logger = logging.getLogger(__name__)
router = APIRouter()


class NotifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["run", "hold", "deletion"]
    run_id: Optional[str] = None

    @field_validator("run_id")
    @classmethod
    def _uuid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not files.is_uuid(value):
            raise ValueError("must be a UUID")
        return value


def _notify(topic_id: str, req: NotifyRequest) -> Dict[str, Any]:
    defn = topics.load_definition(topic_id)
    if defn is None or defn.status == "deleted":
        return {"sent": 0, "skipped": 0, "reason": "topic_not_found"}
    addresses = [s.email for s in defn.notify.subscribers if s.confirmed_at is not None]
    if not notify_email.email_enabled():
        return {"sent": 0, "skipped": len(addresses), "reason": "email_disabled"}
    if not addresses:
        return {"sent": 0, "skipped": 0, "reason": "no_subscribers"}

    if req.kind == "run":
        record = topics.load_run(topic_id, req.run_id) if req.run_id else None
        if record is None:
            return {"sent": 0, "skipped": len(addresses), "reason": "run_not_found"}
        if record.get("status") != "completed":
            return {"sent": 0, "skipped": len(addresses), "reason": "run_not_completed"}
        kind = notify_email.run_kind()
        return notify_email.send_topic_email(topic_id, defn.name, addresses, kind,
                                             record=record, run_id=req.run_id)
    return notify_email.send_topic_email(topic_id, defn.name, addresses, req.kind)


@router.post("/{topic_id}/notify", dependencies=[Depends(require_scheduler)])
async def notify_topic(topic_id: str, req: NotifyRequest) -> Dict[str, Any]:
    if not files.is_uuid(topic_id):
        raise HTTPException(status_code=404, detail="Topic not found")
    if req.kind == "run" and req.run_id is None:
        raise HTTPException(status_code=422, detail="run_id is required for kind run")
    try:
        result = await run_in_threadpool(_notify, topic_id, req)
    except Exception as exc:  # noqa: BLE001 - email never fails a run
        logger.warning("Notify %s for topic %s failed (%s)", req.kind, topic_id, type(exc).__name__)
        result = {"sent": 0, "skipped": 0, "reason": "error"}
    logger.info("Notify %s for topic %s: %s", req.kind, topic_id,
                {k: v for k, v in result.items() if k in ("sent", "skipped", "reason")})
    return result

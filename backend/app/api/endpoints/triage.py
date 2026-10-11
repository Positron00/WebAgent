"""POST /api/triage — classify whether a question needs the agentic workflow."""
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.triage import classify

router = APIRouter()


class TriageRequest(BaseModel):
    query: str = Field(default="", max_length=8000)
    previous_user_question: Optional[str] = Field(default=None, max_length=8000)
    has_image: bool = False


class TriageResponse(BaseModel):
    route: Literal["direct", "agentic"]
    p_agentic: Optional[float]
    decided_by: Literal["rule", "model", "fallback", "config"]
    reason: str
    classifier_version: str


@router.post("", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    d = classify(req.query, req.previous_user_question, req.has_image)
    body = d.to_dict()
    return TriageResponse(
        route=body["route"],
        p_agentic=body["p_agentic"],
        decided_by=body["decided_by"],
        reason=body["reason"],
        classifier_version=body["classifier_version"],
    )

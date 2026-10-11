"""Route questions to direct LLM vs agentic workflow (rules + local classifier)."""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

Route = Literal["direct", "agentic"]
DecidedBy = Literal["rule", "model", "fallback", "config", "mode", "override"]

GREETINGS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "bye",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
    }
)

RESEARCH_PREFIX = re.compile(
    r"^\s*(please\s+)?(research|find sources|find studies|write a report|do a deep dive|"
    r"give me a literature review)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TriageDecision:
    route: Route
    p_agentic: Optional[float]
    decided_by: DecidedBy
    reason: str
    classifier_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "p_agentic": round(self.p_agentic, 3) if self.p_agentic is not None else None,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "classifier_version": self.classifier_version,
            "latency_ms": round(self.latency_ms, 2),
        }


def build_model_input(query: str, previous_user_question: Optional[str]) -> str:
    q = query.strip()[:2000]
    cfg = _triage_cfg()
    short_words = cfg.short_query_words if cfg else 6
    if previous_user_question and len(q.split()) < short_words:
        prev = previous_user_question.strip()[:1000]
        return f"{prev} → {q}"
    return q


def match_rule(query: str, has_image: bool) -> Optional[tuple[Route, str]]:
    if has_image:
        return "direct", "Has an image"
    normalized = re.sub(r"[^\w\s']", "", query).strip().lower()
    if not normalized or normalized in GREETINGS:
        return "direct", "Greeting or short acknowledgement"
    if RESEARCH_PREFIX.match(query):
        return "agentic", "Explicit research request"
    return None


def _triage_cfg():
    return getattr(settings, "TRIAGE", None)


def classify(
    query: str,
    previous_user_question: Optional[str] = None,
    has_image: bool = False,
) -> TriageDecision:
    """Decide the route for one question. Never raises."""
    start = time.perf_counter()
    version = "none"

    def finish(
        route: Route,
        decided_by: DecidedBy,
        reason: str,
        p: Optional[float] = None,
        clf_version: str = "none",
        threshold: Optional[float] = None,
    ) -> TriageDecision:
        from app.core.metrics import record_triage_decision

        latency_ms = (time.perf_counter() - start) * 1000
        decision = TriageDecision(route, p, decided_by, reason, clf_version, latency_ms)
        record_triage_decision(decision, threshold)
        _log_decision(query, decision)
        return decision

    try:
        cfg = _triage_cfg()
        if cfg is not None and not cfg.enabled:
            return finish("agentic", "config", "Triage disabled")

        rule = match_rule(query, has_image)
        if rule:
            route, reason = rule
            return finish(route, "rule", reason)

        from app.services import triage_model

        model = triage_model.get_model()
        if model is None:
            return finish("agentic", "fallback", "Classifier unavailable")

        version = model.metadata.version
        text = build_model_input(query, previous_user_question)
        try:
            p = model.predict_proba(text)
        except Exception as exc:
            logger.warning("Triage model error: %s", exc)
            return finish("agentic", "fallback", "Classifier error", clf_version=version)

        threshold = cfg.threshold if cfg and cfg.threshold is not None else model.metadata.threshold
        route: Route = "agentic" if p >= threshold else "direct"

        if cfg and cfg.llm_tiebreak and threshold <= p < cfg.tiebreak_upper:
            route = _llm_tiebreak(query, route)

        reason = f"Looks like: {model.nearest_category(text)}"
        return finish(route, "model", reason, p=p, clf_version=version, threshold=threshold)
    except Exception as exc:
        logger.exception("Triage classify failed: %s", exc)
        return finish("agentic", "fallback", "Classifier error", clf_version=version)


TIEBREAK_TIMEOUT_SEC = 3.0
TIEBREAK_PROMPT = (
    "Decide how a chat assistant should answer the question below.\n"
    "Reply with exactly one word: RESEARCH if it needs web research, current data or several "
    "sources; DIRECT if one quick answer from general knowledge is enough.\n\n"
    "Question: {query}\n\nAnswer:"
)
_tiebreak_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triage-tiebreak")


def _ask_llm(prompt: str) -> str:
    """One text LLM call. Replaced in tests."""
    from app.services.llm import get_llm

    reply = get_llm().invoke(prompt)
    return reply if isinstance(reply, str) else str(getattr(reply, "content", reply))


def _llm_tiebreak(query: str, default_route: Route) -> Route:
    """Let one LLM call settle a borderline case; keep `default_route` on error or after 3 s."""
    future = _tiebreak_pool.submit(_ask_llm, TIEBREAK_PROMPT.format(query=query[:2000]))
    try:
        answer = future.result(timeout=TIEBREAK_TIMEOUT_SEC).strip().upper()
    except Exception as exc:
        logger.warning("Triage tie-breaker failed, keeping the model's route: %s", exc)
        return default_route
    if answer.startswith("RESEARCH"):
        return "agentic"
    if answer.startswith("DIRECT"):
        return "direct"
    logger.warning("Triage tie-breaker gave an unclear answer; keeping the model's route")
    return default_route


def _log_decision(query: str, decision: TriageDecision) -> None:
    extra = ""
    if settings.WEBAGENT_ENV == "dev":
        extra = f' query="{query[:200]}"'
    logger.info(
        'triage route=%s p=%s by=%s version=%s latency_ms=%.1f reason="%s"%s',
        decision.route,
        decision.p_agentic,
        decision.decided_by,
        decision.classifier_version,
        decision.latency_ms,
        decision.reason,
        extra,
    )

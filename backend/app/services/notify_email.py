"""Topic email: the ledger, caps, messages and transports (recurring-research spec REC-132 – REC-138).

Only the backend sends email. Every send goes through `_send`, which checks the caps and records the
outcome in the day's ledger (`data/notify/ledger-<YYYY-MM-DD>.json`, written under flock because the
backend runs several workers). The ledger holds counts by kind and outcome, addresses only as HMAC
hashes (for the per-address confirmation cap), the SHA-256 of used confirm tokens, and which topic
emails went out (by topic ID, removed when the topic is deleted). Ledgers are kept 30 days.

Privacy (RNFR-03): logs carry topic IDs, kinds and lengths, never an address, a topic name or text.
Notice mode puts no model text in a message: the date, the new-source count, the domains of new
sources checkpoint 1 didn't flag, and a link. Summary mode (only with `input_mode: act`, REC-136)
adds the change summary's bullets and domain-and-path of the new sources.
"""
from __future__ import annotations

import logging
import re
import smtplib
import ssl
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional
from urllib.parse import urlsplit

from prometheus_client import Counter

from app.core.config import settings
from app.core.log_context import describe_text
from app.core.metrics import WEBAGENT_REGISTRY
from app.services import notify_tokens
from app.services import topic_files as files

logger = logging.getLogger(__name__)

Kind = Literal["confirm", "notice", "summary", "hold", "deletion"]
KINDS = ("confirm", "notice", "summary", "hold", "deletion")
OUTCOMES = ("sent", "failed", "capped")

CONFIRMATIONS_PER_ADDRESS_PER_DAY = 3
SUBJECT_NAME_CHARS = 60
SOURCE_PATH_CHARS = 100
MAX_LISTED_SOURCES = 20
SUMMARY_CHARS = 3000
UNCONFIRMED_TTL = timedelta(hours=48)
FOOTER_AI = "Written by an AI research assistant."

EMAIL_SENT = Counter(
    "webagent_email_sent_total",
    "Topic emails by kind and outcome (sent, failed, capped)",
    ["kind", "outcome"],
    registry=WEBAGENT_REGISTRY,
)

_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def email_config():
    notifications = getattr(settings, "NOTIFICATIONS", None)
    return notifications.email if notifications is not None else None


def email_enabled() -> bool:
    cfg = email_config()
    return bool(cfg and cfg.enabled)


def summary_mode_allowed() -> bool:
    """Summary emails carry model text, so they need checkpoint 1 acting (REC-136)."""
    guard = getattr(settings, "SECURITY_GUARD", None)
    return bool(guard is not None and getattr(guard, "input_mode", None) == "act")


def run_kind() -> Kind:
    """`summary` when configured and allowed, else `notice`."""
    cfg = email_config()
    if cfg is not None and cfg.mode == "summary" and summary_mode_allowed():
        return "summary"
    return "notice"


# --- the ledger -------------------------------------------------------------------------------------

def _today(now: Optional[datetime] = None) -> date:
    return (now or files.utcnow()).astimezone(timezone.utc).date()


def ledger_path(day: date) -> Path:
    return files.notify_dir() / f"ledger-{day.isoformat()}.json"


def _empty_ledger(day: date) -> Dict[str, Any]:
    return {"date": day.isoformat(), "counts": {}, "attempts": 0, "addresses": {},
            "used_confirm_tokens": [], "topics": {}}


def _read_ledger(day: date) -> Dict[str, Any]:
    data = files.read_json(ledger_path(day)) or {}
    ledger = _empty_ledger(day)
    for key, value in data.items():
        if key in ledger and isinstance(value, type(ledger[key])):
            ledger[key] = value
    return ledger


def _update_ledger(change, now: Optional[datetime] = None) -> Any:
    """Apply `change(ledger)` to today's ledger under its lock; returns what `change` returns."""
    day = _today(now)
    path = ledger_path(day)
    with files.locked(path):
        ledger = _read_ledger(day)
        out = change(ledger)
        files.write_json_atomic(path, ledger)
    return out


def address_hash(address: str) -> str:
    """The address as the ledger stores it: an HMAC with the secret key, so it can't be looked up."""
    return notify_tokens.address_ref(address)


def _count(ledger: Dict[str, Any], kind: str, outcome: str) -> None:
    by_kind = ledger["counts"].setdefault(kind, {})
    by_kind[outcome] = int(by_kind.get(outcome, 0)) + 1
    EMAIL_SENT.labels(kind=kind, outcome=outcome).inc()


def _recent_days(now: Optional[datetime], days: int) -> List[date]:
    today = _today(now)
    return [today - timedelta(days=n) for n in range(days)]


def confirm_token_used(token: str, now: Optional[datetime] = None) -> bool:
    """True if the confirm token was used (looked up in the ledgers of its 48 h lifetime)."""
    digest = notify_tokens.token_hash(token)
    return any(digest in _read_ledger(day)["used_confirm_tokens"] for day in _recent_days(now, 3))


def use_confirm_token(token: str, now: Optional[datetime] = None) -> bool:
    """Mark a confirm token used. False if it already was (single use, REC-133).

    The check of today's ledger and the mark are one locked step, so two workers can't both use it;
    earlier days are only read (a token is never marked on a day other than the one it was used).
    """
    digest = notify_tokens.token_hash(token)
    if any(digest in _read_ledger(day)["used_confirm_tokens"] for day in _recent_days(now, 3)[1:]):
        return False

    def mark(ledger: Dict[str, Any]) -> bool:
        if digest in ledger["used_confirm_tokens"]:
            return False
        ledger["used_confirm_tokens"].append(digest)
        return True

    return _update_ledger(mark, now)


def forget_topic(topic_id: str) -> None:
    """Remove a deleted topic's entries from every ledger (REC-114)."""
    folder = files.notify_dir()
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("ledger-*.json")):
        try:
            day = date.fromisoformat(path.stem[len("ledger-"):])
        except ValueError:
            continue
        with files.locked(path):
            ledger = _read_ledger(day)
            if topic_id in ledger["topics"]:
                ledger["topics"].pop(topic_id, None)
                files.write_json_atomic(path, ledger)


def cleanup_ledgers(now: Optional[datetime] = None, keep_days: Optional[int] = None) -> int:
    """Delete ledgers older than the retention (30 days, `RESULTS_RETENTION_DAYS`). Never raises."""
    keep = keep_days if keep_days is not None else int(getattr(settings, "RESULTS_RETENTION_DAYS", 30))
    oldest = _today(now) - timedelta(days=keep)
    removed = 0
    folder = files.notify_dir()
    if not folder.is_dir():
        return 0
    for path in folder.glob("ledger-*.json"):
        try:
            day = date.fromisoformat(path.stem[len("ledger-"):])
        except ValueError:
            continue
        if day < oldest:
            try:
                path.unlink()
                path.with_name(path.name + ".lock").unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


# --- message building -------------------------------------------------------------------------------

def clean_text(value: Any, limit: Optional[int] = None) -> str:
    """Text without control, format or line/paragraph-separator characters (headers, names)."""
    text = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Zl", "Zp") else ch
        for ch in str(value or "")
        if unicodedata.category(ch) not in ("Cf", "Cs", "Co")
    )
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _clean_line(value: Any) -> str:
    """One body line of model text: control characters removed, whitespace collapsed."""
    return clean_text(value)


def subject_for(name: str, day: date) -> str:
    return f"WebAgent: {clean_text(name, SUBJECT_NAME_CHARS)} — {day.isoformat()}"


def _public_url() -> str:
    return (getattr(settings, "PUBLIC_URL", None) or "").rstrip("/")


def topic_link(topic_id: str) -> str:
    return f"{_public_url()}/topics/{topic_id}"


def confirm_link(token: str) -> str:
    return f"{_public_url()}/notify/confirm?token={token}"


def unsubscribe_page_link(token: str) -> str:
    return f"{_public_url()}/notify/unsubscribe?token={token}"


def one_click_link(token: str) -> str:
    return f"{_public_url()}/api/notify/unsubscribe?token={token}"


def _host(url: Any) -> Optional[str]:
    """The URL's host as plain lower-case ASCII (IDNA), without `www.`, port or credentials; else None."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
    except ValueError:
        return None
    if not host:
        return None
    try:
        host = host.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None
    if host.startswith("www."):
        host = host[4:]
    return host if len(host) <= 253 and _DOMAIN.match(host) else None


def _key(url: Any) -> Optional[str]:
    from app.services.topic_runs import normalise_url

    return normalise_url(url) if isinstance(url, str) else None


def unflagged_new_sources(record: Dict[str, Any]) -> List[str]:
    """The run's new source URLs, minus every URL in `flagged_refs` and every URL on a flagged domain.

    Checkpoint 1 flagged those (act: dropped; shadow: would have dropped), so even their domain names
    stay out of the email (REC-135, T10).
    """
    change = record.get("change") or {}
    new = change.get("new_sources")
    if not isinstance(new, list):
        new = [s.get("url") if isinstance(s, dict) else s for s in (record.get("sources") or [])]
    flagged = [r for r in (record.get("flagged_refs") or []) if isinstance(r, str)]
    flagged_keys = {_key(r) for r in flagged} - {None}
    flagged_hosts = {_host(r) for r in flagged} - {None}
    out: List[str] = []
    for url in new:
        if not isinstance(url, str):
            continue
        if _key(url) in flagged_keys or url in flagged or _host(url) in flagged_hosts:
            continue
        out.append(url)
    return out


def _domains(urls: Iterable[str]) -> List[str]:
    out: List[str] = []
    for url in urls:
        host = _host(url)
        if host and host not in out:
            out.append(host)
    return out


def _domain_and_path(url: str) -> Optional[str]:
    host = _host(url)
    if host is None:
        return None
    try:
        path = urlsplit(url.strip()).path or ""
    except ValueError:
        path = ""
    text = clean_text(host + path.rstrip("/"))
    return text[:SOURCE_PATH_CHARS]


def _listed(lines: List[str]) -> List[str]:
    shown = [f"- {line}" for line in lines[:MAX_LISTED_SOURCES]]
    if len(lines) > MAX_LISTED_SOURCES:
        shown.append(f"- and {len(lines) - MAX_LISTED_SOURCES} more")
    return shown


def _summary_bullets(summary: Any) -> List[str]:
    if not isinstance(summary, str) or not summary.strip():
        return []
    text = summary[:SUMMARY_CHARS]
    bullets = []
    for raw in text.splitlines():
        line = _clean_line(raw).lstrip("-*• ").strip()
        if line:
            bullets.append(f"- {line}")
    return bullets[:10]


def _footer(name: str, reason: str, unsubscribe_token: str) -> List[str]:
    return [
        "",
        "--",
        reason.format(name=clean_text(name, 80)),
        f"Unsubscribe: {unsubscribe_page_link(unsubscribe_token)}",
        FOOTER_AI,
    ]


_WHY_CONFIRM = 'You got this email because someone entered this address for the WebAgent topic "{name}".'
_WHY_TOPIC = 'You get this email because this address subscribed to the WebAgent topic "{name}" and confirmed it.'


def _run_date(record: Optional[Dict[str, Any]], now: Optional[datetime]) -> date:
    moment = None
    if record:
        moment = files.parse_time(record.get("completed_at")) or files.parse_time(record.get("started_at"))
    return _today(moment or now)


def body_for(kind: Kind, *, topic_id: str, name: str, unsubscribe_token: str,
             confirm_token: Optional[str] = None, record: Optional[Dict[str, Any]] = None,
             now: Optional[datetime] = None) -> str:
    """The plain-text body of a `kind` email (REC-135)."""
    quoted = clean_text(name, 80)
    if kind == "confirm":
        lines = [
            f'Please confirm that you want WebAgent updates about the topic "{quoted}".',
            "",
            "Open this link and press Confirm (it works for 48 hours):",
            confirm_link(confirm_token or ""),
            "",
            "If you didn't ask for this, ignore this email: nothing more will be sent.",
        ]
        return "\n".join(lines + _footer(name, _WHY_CONFIRM, unsubscribe_token)) + "\n"

    if kind in ("notice", "summary"):
        record = record or {}
        day = _run_date(record, now)
        new = unflagged_new_sources(record)
        count = len(new)
        lines = [f'WebAgent ran the topic "{quoted}" on {day.isoformat()}.',
                 f"{count} new source{'s' if count != 1 else ''}."]
        if kind == "summary":
            bullets = _summary_bullets((record.get("change") or {}).get("summary"))
            if bullets:
                lines += ["", "What changed:"] + bullets
            listed = [p for p in (_domain_and_path(u) for u in new) if p]
            if listed:
                lines += ["", "New sources:"] + _listed(listed)
        else:
            domains = _domains(new)
            if domains:
                lines += ["", "New sources from:"] + _listed(domains)
        lines += ["", f"Read the report: {topic_link(topic_id)}"]
        return "\n".join(lines + _footer(name, _WHY_TOPIC, unsubscribe_token)) + "\n"

    if kind == "hold":
        lines = [
            f'Scheduled runs of the topic "{quoted}" have stopped after several failed runs in a row.',
            "Nothing more runs until someone resumes it:",
            topic_link(topic_id),
        ]
        return "\n".join(lines + _footer(name, _WHY_TOPIC, unsubscribe_token)) + "\n"

    if kind == "deletion":
        lines = [
            f'The topic "{quoted}" has been paused for a long time and will be deleted in 7 days,',
            "with its saved reports. To keep it, resume it before then:",
            topic_link(topic_id),
        ]
        return "\n".join(lines + _footer(name, _WHY_TOPIC, unsubscribe_token)) + "\n"

    raise ValueError("unknown email kind")


def build_message(kind: Kind, *, topic_id: str, name: str, address: str,
                  record: Optional[Dict[str, Any]] = None, confirm_token: Optional[str] = None,
                  now: Optional[datetime] = None) -> EmailMessage:
    """An `EmailMessage` with the fixed headers (REC-134, REC-135). Headers can't be injected:
    `EmailMessage` refuses CR and LF in header values, and the name is cleaned first anyway."""
    unsubscribe = notify_tokens.unsubscribe_token(topic_id, address)
    day = _run_date(record, now) if kind in ("notice", "summary") else _today(now)
    msg = EmailMessage()
    msg["Subject"] = subject_for(name, day)
    msg["From"] = settings.SMTP_FROM or "webagent@localhost"
    msg["To"] = address
    msg["List-Unsubscribe"] = f"<{one_click_link(unsubscribe)}>"
    msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    msg["Auto-Submitted"] = "auto-generated"
    msg.set_content(
        body_for(kind, topic_id=topic_id, name=name, unsubscribe_token=unsubscribe,
                 confirm_token=confirm_token, record=record, now=now),
        subtype="plain", charset="utf-8",
    )
    return msg


# --- transports -------------------------------------------------------------------------------------

class LogTransport:
    """Writes a line to the log instead of sending: kind and lengths only (dev and tests)."""

    def send(self, msg: EmailMessage, *, kind: str, topic_id: str) -> bool:
        body = msg.get_content() if not msg.is_multipart() else ""
        logger.info("Email %s for topic %s logged, not sent (subject %s, body %s)",
                    kind, topic_id, describe_text(str(msg["Subject"] or "")), describe_text(body))
        return True


_PERMANENT = (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused, smtplib.SMTPAuthenticationError,
              smtplib.SMTPNotSupportedError)


class SmtpTransport:
    """`smtplib` with STARTTLS (587) or implicit TLS (`WEBAGENT_SMTP_TLS=ssl`, 465); 20 s timeout,
    2 retries after 2 s and 8 s. A failure is logged by topic ID, kind and exception type only."""

    TIMEOUT = 20
    BACKOFF = (2, 8)

    def __init__(self, sleep=time.sleep):
        self._sleep = sleep

    def _send_once(self, msg: EmailMessage) -> None:
        host, port = settings.SMTP_HOST, int(settings.SMTP_PORT or 587)
        context = ssl.create_default_context()
        if (settings.SMTP_TLS or "starttls").lower() == "ssl":
            server = smtplib.SMTP_SSL(host, port, timeout=self.TIMEOUT, context=context)
        else:
            server = smtplib.SMTP(host, port, timeout=self.TIMEOUT)
        with server:
            if not isinstance(server, smtplib.SMTP_SSL):
                server.starttls(context=context)
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
            server.send_message(msg)

    def send(self, msg: EmailMessage, *, kind: str, topic_id: str) -> bool:
        tries = len(self.BACKOFF) + 1
        for attempt in range(tries):
            try:
                self._send_once(msg)
                return True
            except _PERMANENT as exc:
                logger.warning("Email %s for topic %s refused (%s)", kind, topic_id, type(exc).__name__)
                return False
            except (smtplib.SMTPException, OSError) as exc:
                if attempt + 1 >= tries:
                    logger.warning("Email %s for topic %s failed after %d tries (%s)",
                                   kind, topic_id, tries, type(exc).__name__)
                    return False
                self._sleep(self.BACKOFF[attempt])
        return False


def transport():
    cfg = email_config()
    if cfg is not None and cfg.transport == "smtp":
        return SmtpTransport()
    return LogTransport()


# --- sending with caps ------------------------------------------------------------------------------

def _max_per_day() -> int:
    cfg = email_config()
    return int(cfg.max_emails_per_day) if cfg is not None else 50


def _deliver(msg: EmailMessage, kind: str, topic_id: str) -> str:
    """Send a reserved email and record the outcome. Returns sent or failed. Never raises."""
    try:
        ok = transport().send(msg, kind=kind, topic_id=topic_id)
    except Exception as exc:  # noqa: BLE001 - email must never break a request or a run
        logger.warning("Email %s for topic %s failed (%s)", kind, topic_id, type(exc).__name__)
        ok = False
    outcome = "sent" if ok else "failed"
    _update_ledger(lambda ledger: _count(ledger, kind, outcome))
    return outcome


def send_confirmation(topic_id: str, name: str, address: str, now: Optional[datetime] = None) -> str:
    """The one confirmation email of an unconfirmed subscriber (REC-133). Returns sent, failed or capped.

    Capped at 3 per address per day and by the instance's daily cap.
    """
    hashed = address_hash(address)

    def reserve(ledger: Dict[str, Any]) -> bool:
        entry = ledger["addresses"].setdefault(hashed, {"confirmations": 0})
        if int(entry.get("confirmations", 0)) >= CONFIRMATIONS_PER_ADDRESS_PER_DAY \
                or int(ledger["attempts"]) >= _max_per_day():
            _count(ledger, "confirm", "capped")
            return False
        entry["confirmations"] = int(entry.get("confirmations", 0)) + 1
        ledger["attempts"] = int(ledger["attempts"]) + 1
        return True

    if not _update_ledger(reserve, now):
        logger.info("Confirmation email for topic %s capped", topic_id)
        return "capped"
    token = notify_tokens.confirm_token(topic_id, address, now=(now.timestamp() if now else None))
    msg = build_message("confirm", topic_id=topic_id, name=name, address=address, confirm_token=token, now=now)
    outcome = _deliver(msg, "confirm", topic_id)
    logger.info("Confirmation email for topic %s %s", topic_id, outcome)
    return outcome


def send_topic_email(topic_id: str, name: str, addresses: List[str], kind: Kind, *,
                     record: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None,
                     now: Optional[datetime] = None) -> Dict[str, Any]:
    """Send a run, hold or deletion email to confirmed `addresses` (REC-131, REC-137, REC-138).

    One email per topic per run (or per kind per day for hold and deletion notices): a repeat request
    sends nothing. Each address costs one from the instance's daily cap; over it, emails are capped.
    Returns `{sent, skipped, reason?}`.
    """
    if not addresses:
        return {"sent": 0, "skipped": 0, "reason": "no_subscribers"}
    marker = f"run:{run_id}" if run_id else kind
    window = _recent_days(now, 2)

    for day in window[1:]:
        if marker in _read_ledger(day)["topics"].get(topic_id, []):
            return {"sent": 0, "skipped": len(addresses), "reason": "already_sent"}

    def reserve(ledger: Dict[str, Any]) -> Optional[int]:
        done = ledger["topics"].setdefault(topic_id, [])
        if marker in done:
            return None
        done.append(marker)
        room = max(0, _max_per_day() - int(ledger["attempts"]))
        allowed = min(room, len(addresses))
        ledger["attempts"] = int(ledger["attempts"]) + allowed
        for _ in range(len(addresses) - allowed):
            _count(ledger, kind, "capped")
        return allowed

    allowed = _update_ledger(reserve, now)
    if allowed is None:
        return {"sent": 0, "skipped": len(addresses), "reason": "already_sent"}

    sent = failed = 0
    for address in addresses[:allowed]:
        try:
            msg = build_message(kind, topic_id=topic_id, name=name, address=address, record=record, now=now)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Email %s for topic %s not built (%s)", kind, topic_id, type(exc).__name__)
            _update_ledger(lambda ledger: _count(ledger, kind, "failed"))
            failed += 1
            continue
        if _deliver(msg, kind, topic_id) == "sent":
            sent += 1
        else:
            failed += 1
    capped = len(addresses) - allowed
    logger.info("Email %s for topic %s: %d sent, %d failed, %d capped", kind, topic_id, sent, failed, capped)
    out: Dict[str, Any] = {"sent": sent, "skipped": failed + capped}
    if capped:
        out["reason"] = "daily_cap"
    elif failed:
        out["reason"] = "send_failed"
    return out


def expired_unconfirmed(subscriber: Any, now: datetime) -> bool:
    """True for an unconfirmed subscriber added more than 48 h ago (REC-115)."""
    if getattr(subscriber, "confirmed_at", None) is not None:
        return False
    added = files.parse_time(getattr(subscriber, "added_at", None))
    return added is not None and now - added > UNCONFIRMED_TTL

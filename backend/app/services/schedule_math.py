"""Slots of a topic's schedule (recurring-research spec REC-104).

`next_slot` returns the first slot strictly after a moment, in UTC. Slots are computed in the topic's
time zone: a local time that doesn't exist (the hour skipped when clocks go forward) moves to the
first valid minute after it; a time that occurs twice (the repeated hour) uses its first occurrence.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _exists(local: datetime) -> bool:
    """True if this wall time exists in its zone (a round trip through UTC keeps it)."""
    back = local.astimezone(timezone.utc).astimezone(local.tzinfo)
    return back.replace(tzinfo=None) == local.replace(tzinfo=None)


def local_slot(day: date, at: time, zone: ZoneInfo) -> datetime:
    """The aware local datetime of `at` on `day`, moved past a DST gap, first of a repeat."""
    candidate = datetime.combine(day, at).replace(tzinfo=zone, fold=0)
    steps = 0
    while not _exists(candidate) and steps < 180:   # gaps are at most a few hours
        candidate = (candidate.replace(tzinfo=None) + timedelta(minutes=1)).replace(tzinfo=zone, fold=0)
        steps += 1
    return candidate


def parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def next_slot(every: str, at: str, tz: str, after: datetime, weekday: Optional[str] = None) -> datetime:
    """First slot strictly after `after` (aware), as an aware UTC datetime."""
    if after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    zone = ZoneInfo(tz)
    at_time = parse_hhmm(at)
    start = after.astimezone(zone).date()
    target_weekday = WEEKDAYS.index(weekday) if every == "week" and weekday else None
    for offset in range(0, 15):
        day = start + timedelta(days=offset)
        if target_weekday is not None and day.weekday() != target_weekday:
            continue
        slot = local_slot(day, at_time, zone).astimezone(timezone.utc)
        if slot > after:
            return slot
    raise ValueError("no slot found")   # unreachable for valid schedules


def period(every: str) -> timedelta:
    return timedelta(days=7) if every == "week" else timedelta(days=1)

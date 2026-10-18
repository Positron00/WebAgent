"""Slot maths for topic schedules (recurring-research spec REC-104)."""
from datetime import datetime, timezone

from app.services.schedule_math import next_slot


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_daily_slot_is_strictly_after():
    assert next_slot("day", "07:30", "UTC", utc(2026, 10, 20, 7, 30)) == utc(2026, 10, 21, 7, 30)
    assert next_slot("day", "07:30", "UTC", utc(2026, 10, 20, 7, 29)) == utc(2026, 10, 20, 7, 30)


def test_weekly_slot_lands_on_the_weekday_across_a_year_end():
    # 2026-12-31 is a Thursday; the next Monday is 2027-01-04.
    assert next_slot("week", "09:00", "UTC", utc(2026, 12, 31, 12), weekday="mon") == utc(2027, 1, 4, 9)


def test_spring_gap_moves_to_the_first_valid_minute():
    # Paris skips 02:00-03:00 local on 2027-03-28: 02:30 becomes 03:00 CEST = 01:00 UTC.
    assert next_slot("day", "02:30", "Europe/Paris", utc(2027, 3, 27, 12)) == utc(2027, 3, 28, 1, 0)


def test_autumn_repeat_uses_the_first_occurrence_and_runs_once():
    # Paris repeats 02:00-03:00 local on 2026-10-25; the first 02:30 is CEST = 00:30 UTC.
    first = next_slot("day", "02:30", "Europe/Paris", utc(2026, 10, 24, 12))
    assert first == utc(2026, 10, 25, 0, 30)
    # The next slot is the following day, not the second 02:30 an hour later.
    assert next_slot("day", "02:30", "Europe/Paris", first) == utc(2026, 10, 26, 1, 30)


def test_new_york_spring_gap():
    # New York skips 02:00-03:00 on 2027-03-14: 02:15 becomes 03:00 EDT = 07:00 UTC.
    assert next_slot("day", "02:15", "America/New_York", utc(2027, 3, 13, 12)) == utc(2027, 3, 14, 7, 0)


def test_late_evening_slot():
    assert next_slot("day", "23:59", "Asia/Tokyo", utc(2026, 10, 20, 15)) == utc(2026, 10, 21, 14, 59)

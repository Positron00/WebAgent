"""Run records, URL normalisation, the source diff and pruning (recurring-research spec REC-111, REC-115)."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.services import topic_files as files
from app.services import topic_runs


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("raw, expected", [
    ("https://WWW.Example.COM/Path/", "https://example.com/Path"),
    ("https://example.com/a#section", "https://example.com/a"),
    ("https://example.com/a?utm_source=x&utm_medium=y&id=3", "https://example.com/a?id=3"),
    ("https://example.com/a?fbclid=1&gclid=2", "https://example.com/a"),
    ("https://example.com/", "https://example.com"),
    ("HTTP://News.Example.org/x?b=2&a=1", "http://news.example.org/x?b=2&a=1"),
    ("https://user:pw@www.example.com/p", "https://example.com/p"),
])
def test_normalise_url(raw, expected):
    assert topic_runs.normalise_url(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", 42, {"url": "x"}])
def test_normalise_url_without_a_url(raw):
    assert topic_runs.normalise_url(raw) is None


def test_source_url_takes_dicts_and_strings():
    assert topic_runs.source_url({"url": "https://a.org", "title": "A"}) == "https://a.org"
    assert topic_runs.source_url({"link": "https://b.org"}) == "https://b.org"
    assert topic_runs.source_url("https://c.org") == "https://c.org"
    assert topic_runs.source_url({"title": "no url"}) is None
    assert topic_runs.source_url(None) is None


def test_clean_sources_dedupes_and_keeps_sources_without_url():
    cleaned = topic_runs.clean_sources([
        {"url": "https://www.a.org/x/", "title": "A"},
        {"url": "https://a.org/x#frag", "title": "A again"},
        {"title": "no url"},
        "https://b.org",
        7,
    ])
    assert [s["url"] for s in cleaned] == ["https://www.a.org/x/", None, "https://b.org", None]
    assert cleaned[0]["title"] == "A"
    assert topic_runs.clean_sources("not a list") == []


def _run(*urls, status="completed"):
    return {"status": status, "sources": [{"url": u} for u in urls]}


def test_first_run_marks_everything_new():
    change = topic_runs.diff([{"url": "https://a.org"}, {"url": "https://b.org"}], [])
    assert change == {"first": True, "new_sources": ["https://a.org", "https://b.org"],
                      "dropped_sources": [], "changed": True}


def test_failed_runs_alone_still_make_a_first_run():
    change = topic_runs.diff([{"url": "https://a.org"}], [_run(status="failed")])
    assert change["first"] is True


def test_new_sources_are_compared_with_the_last_three_runs():
    previous = [_run("https://b.org"), _run("https://c.org"), _run("https://d.org"), _run("https://old.org")]
    change = topic_runs.diff(
        [{"url": "https://www.b.org/"}, {"url": "https://c.org"}, {"url": "https://old.org"}, {"url": "https://n.org"}],
        previous,
    )
    # old.org was seen 4 runs ago only, outside the window, so it counts as new again.
    assert change["new_sources"] == ["https://old.org", "https://n.org"]
    assert change["first"] is False and change["changed"] is True


def test_dropped_sources_are_compared_with_the_previous_run():
    previous = [_run("https://a.org", "https://b.org"), _run("https://z.org")]
    change = topic_runs.diff([{"url": "https://a.org"}], previous)
    assert change["dropped_sources"] == ["https://b.org"]
    assert change["new_sources"] == [] and change["changed"] is False


def test_diff_skips_failed_runs_and_missing_urls():
    previous = [_run(status="failed"), _run("https://a.org")]
    change = topic_runs.diff([{"title": "x"}, {"url": "https://a.org"}], previous)
    assert change["new_sources"] == [] and change["dropped_sources"] == []


def _save(topic_id, started, status="completed"):
    run_id = str(uuid.uuid4())
    record = topic_runs.new_record(topic_id, run_id, started, "slot")
    record["status"] = status
    topic_runs.save(record)
    return record


def test_record_has_the_fixed_task_id_and_lists_newest_first():
    topic_id = str(uuid.uuid4())
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    a = _save(topic_id, start)
    b = _save(topic_id, start + timedelta(days=1))
    assert a["task_id"] == files.task_id_for_run(a["run_id"])
    assert [r["run_id"] for r in topic_runs.list_records(topic_id)] == [b["run_id"], a["run_id"]]
    assert topic_runs.task_id_of({"run_id": a["run_id"]}) == a["task_id"]
    assert topic_runs.task_id_of({"run_id": "nope"}) is None


def test_prune_keeps_the_newest_and_open_runs():
    topic_id = str(uuid.uuid4())
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    records = [_save(topic_id, start + timedelta(days=i)) for i in range(5)]
    oldest_open = _save(topic_id, start - timedelta(days=9), status="running")
    removed = topic_runs.prune(topic_id, 3)
    assert {r["run_id"] for r in removed} == {records[0]["run_id"], records[1]["run_id"]}
    left = {r["run_id"] for r in topic_runs.list_records(topic_id)}
    assert left == {r["run_id"] for r in records[2:]} | {oldest_open["run_id"]}
    assert topic_runs.prune(topic_id, 3) == []

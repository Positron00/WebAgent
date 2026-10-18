"""One writer per file (recurring-research spec REC-101): the backend writes definitions (and, phase B,
the email ledger); the scheduler writes state, runs, usage, counters and its lock. The only exception
is REC-114's takeover: with no live scheduler the backend deletes a topic's folder and writes the lock.

The check reads the source, so a new write in the wrong process fails here before it ships.
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

BACKEND_MODULES = ["app/services/topics.py", "app/api/endpoints/topics.py"]
SCHEDULER_MODULES = ["app/scheduler.py", "app/services/topic_runs.py"]

SCHEDULER_FILES = ("state_path", "run_path", "runs_dir", "usage_path", "counters_path")


def _writes(module: str):
    """(line, text) of every write_json_atomic call in a module, with its path argument."""
    source = (BACKEND / module).read_text(encoding="utf-8")
    return [(n, line.strip()) for n, line in enumerate(source.splitlines(), 1) if "write_json_atomic(" in line]


def test_the_backend_never_writes_scheduler_files():
    for module in BACKEND_MODULES:
        for n, line in _writes(module):
            assert not any(f"files.{name}" in line for name in SCHEDULER_FILES), f"{module}:{n} {line}"


def test_the_scheduler_never_writes_definitions():
    for module in SCHEDULER_MODULES:
        source = (BACKEND / module).read_text(encoding="utf-8")
        assert "definition_path" not in source or not re.search(
            r"write_json_atomic\([^)]*definition_path", source), module
        assert "save_new_definition" not in source and "update_definition" not in source, module


def test_the_backend_writes_the_lock_only_in_the_takeover():
    source = (BACKEND / "app/services/topics.py").read_text(encoding="utf-8")
    lock_writes = [line for _, line in _writes("app/services/topics.py") if "lock" in line]
    assert len(lock_writes) == 1, lock_writes            # the REC-114 takeover only
    assert "def cleanup_deleted" in source and "scheduler_alive" in source

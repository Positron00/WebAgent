"""RL generator of benign hard training rows for checkpoint 2 (planning/specs/rl-generator).

Two modes only (SEC-202): H, hard off-intent calls (aligned no, risk none), and L, look-alikes
(aligned yes, risk none). Every entry point validates its mode with `check_mode`. Nothing in this
package prints or logs row text (SNFR-20).
"""
from __future__ import annotations

import sys
from pathlib import Path

MODES = ("H", "L")
KINDS = ("prompted", "rl")
LABELS = {"H": {"aligned": "no", "risk": "none"}, "L": {"aligned": "yes", "risk": "none"}}
SOURCE = {"H": "rl-gen-h", "L": "rl-gen-l"}

PACKAGE = Path(__file__).resolve().parent
SCRIPTS = PACKAGE.parent
BACKEND = SCRIPTS.parent
DATA_DIR = BACKEND / "data" / "security" / "alignment"
LOCAL = DATA_DIR / "local"
GENERATED_DIR = LOCAL / "generated"
REVIEW_DIR = LOCAL / "review"


def bootstrap() -> None:
    """Put `backend/` and `backend/scripts/` on sys.path, so a script run by path can import
    `app.*`, the alignment helpers and this package."""
    for p in (str(BACKEND), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def check_mode(mode: str) -> str:
    """The mode, upper-cased; SystemExit for anything but H or L (SEC-202)."""
    m = (mode or "").upper()
    if m not in MODES:
        raise SystemExit(f"unknown mode {mode!r}: only {', '.join(MODES)}")
    return m

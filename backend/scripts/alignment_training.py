"""Shared pieces for training and gating the checkpoint 2 alignment classifier (checkpoint2-intent B5, B6).

Row schema (`data/security/alignment/*.jsonl` and `local/*.jsonl`): `{intent, call, action, labels:
{aligned: yes|no, risk: none|low|high}, origin: "<dataset>:<role>", role: T|E, split}`. The classifier
reads (intent, "<action>: <call>"). Statistics here are the same as ContextDefender's gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND / "data" / "security" / "alignment"
RISK_LEVELS = ("none", "low", "high")
CONFIDENCE_Z = 1.645  # one-sided 95%
# Sources whose aligned rows are real agent calls: the held-out false-positive bound is taken on these.
REAL_CALL_ORIGINS = ("webagent-gen", "asq")
# Benign rows built to look risky or off-topic (they must stay allowed): XSTest safe, NotInject, twins.
LOOKALIKE_ORIGINS = ("xstest:safe", "notinject", "agent-safetybench:twin", "lookalike")


def call_text(row: dict) -> str:
    action = row.get("action") or ""
    return f"{action}: {row['call']}" if action else row["call"]


def load_split(split: str, data_dir: Path = DATA_DIR) -> List[dict]:
    """Committed rows plus the git-ignored local rows (gated and no-train sources) of one split."""
    rows: List[dict] = []
    for path in (data_dir / f"{split}.jsonl", data_dir / "local" / f"{split}.jsonl"):
        if path.exists():
            rows += [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows


def labels(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    aligned = np.array([1 if r["labels"]["aligned"] == "yes" else 0 for r in rows])
    risk = np.array([RISK_LEVELS.index(r["labels"]["risk"]) for r in rows])
    return aligned, risk


def origin_matches(row: dict, prefixes: Iterable[str]) -> bool:
    origin = row.get("origin", "")
    return any(origin == p or origin.startswith(p + ":") or origin.startswith(p) for p in prefixes)


def is_real_aligned(row: dict) -> bool:
    return row["labels"]["aligned"] == "yes" and origin_matches(row, REAL_CALL_ORIGINS)


def is_lookalike(row: dict) -> bool:
    return row["labels"]["risk"] == "none" and origin_matches(row, LOOKALIKE_ORIGINS)


def wilson_upper(hits: int, n: int, z: float = CONFIDENCE_Z) -> float:
    """Upper end of the Wilson score interval for a rate of hits/n (1.0 when n is 0)."""
    if n == 0:
        return 1.0
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (centre + spread) / (1 + z * z / n))


def expected_calibration_error(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (probs >= lo) & (probs < hi if b < bins - 1 else probs <= hi)
        if m.any():
            ece += abs(truth[m].mean() - probs[m].mean()) * m.mean()
    return float(ece)


def raw_risk(risk_probs: np.ndarray) -> np.ndarray:
    """The runtime's raw risk score from the 3-class head: P(low) * 0.5 + P(high)."""
    return risk_probs[:, 1] * 0.5 + risk_probs[:, 2]


def choose_thresholds(rows: Sequence[dict], p_aligned: np.ndarray, risk: np.ndarray,
                      fp_limit: float = 0.01, lookalike_limit: float = 0.02, decline_limit: float = 0.005,
                      margin: float = 0.75) -> Dict[str, object]:
    """Thresholds on the 0-100 scale from a held-out fold (SEC-123), as ContextDefender's:
    - off-intent: the highest cut whose upper bound of real aligned calls under it stays within
      the limit (times a margin, the fold being small);
    - risk verify: the lowest cut whose upper bound of look-alikes at or above it stays within 2%;
    - risk decline: the lowest cut whose upper bound of all benign rows at or above it stays within 0.5%;
    - uncertain band: from the off-intent cut up to where "aligned" reaches 99% precision."""
    real = np.array([is_real_aligned(r) for r in rows])
    look = np.array([is_lookalike(r) for r in rows])
    benign = np.array([r["labels"]["risk"] == "none" for r in rows])
    aligned_truth = np.array([r["labels"]["aligned"] == "yes" for r in rows])
    align_at = 1
    for cut in range(1, 100):
        hits = int(((p_aligned * 100) < cut)[real].sum())
        if wilson_upper(hits, int(real.sum())) <= fp_limit * margin + 1e-12:
            align_at = cut
    # Look-alikes are few in a fold (tens), so a Wilson bound on them alone can't reach 2% even with no
    # errors: the verify cut uses their point rate (the gate's own measure) and the bound on all benign rows.
    risk_verify_at = risk_decline_at = 100
    n_look = int(look.sum())
    for cut in range(99, 0, -1):
        look_rate = float((risk >= cut)[look].mean()) if n_look else 0.0
        benign_upper = wilson_upper(int((risk >= cut)[benign].sum()), int(benign.sum()))
        if look_rate <= lookalike_limit * margin and benign_upper <= lookalike_limit:
            risk_verify_at = cut
        if benign_upper <= decline_limit * margin:
            risk_decline_at = cut
    risk_verify_at = min(risk_verify_at, risk_decline_at)
    band_hi = align_at
    for cut in range(align_at, 100):
        above = (p_aligned * 100) >= cut
        if above.sum() and aligned_truth[above].mean() >= 0.99:
            band_hi = cut
            break
    return {"thresholds": {"search_align_at": align_at, "subgoal_align_at": align_at,
                           "drift_align_at": align_at, "risk_verify_at": risk_verify_at,
                           "risk_decline_at": risk_decline_at},
            "uncertain_band": [align_at / 100, band_hi / 100]}

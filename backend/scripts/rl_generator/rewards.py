"""Reward terms and the combined reward (rl-generator SEC-203 – SEC-207; design §4).

Per sample, in this order (§4.1): clean; length (`overlong`); harmless filter (ContextDefender's
harm at or above the installed model's `harm_verify_at`, or unscorable, gives reward 0 and stops);
`miss` from the frozen target; `novelty` against the bank plus this run's accepted calls; `valid`
from the judge only when `miss >= judge_min_miss`; then
`reward = int(bool(valid)) * (miss + lam_div * novelty) - lam_len * overlong`.
`Scored` holds numbers only, never text. The target, harm function, embedder and tokenizer are
injectable, so tests use fakes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from rl_generator import check_mode

MAX_CALL_CHARS = 1000
MAX_PAIR_TOKENS = 128


def harm_verify_at() -> int:
    """The installed ContextDefender's `harm_verify_at` (20 for cd-2026-10-13-2), read once per run.
    SystemExit without a model: every sample would be unscorable."""
    from app.security import context_defender

    if context_defender.get_model() is None:
        raise SystemExit("ContextDefender's model is not available (Git LFS?); the harmless filter needs it")
    return int(context_defender.current_thresholds()["harm_verify_at"])


def default_harm(text: str) -> Optional[int]:
    from app.security.signals import harm_of

    return harm_of(text)


@dataclass
class RewardContext:
    mode: str
    target: Any                       # alignment_model.load(dir, require_gate=False), or a fake
    novelty_bank: np.ndarray          # normalised MiniLM-L6 embeddings of the committed train calls
    tokenizer: Any                    # the target's tokenizer, for overlong
    judge: Any                        # Judge, or a callable (intent, call, action) -> bool | None
    lam_div: float = 0.2
    lam_len: float = 0.5
    judge_min_miss: float = 0.3
    harm_verify_at: Optional[int] = None   # None: read from ContextDefender once, here
    harm: Callable[[str], Optional[int]] = default_harm
    embed: Optional[Callable[[List[str]], np.ndarray]] = None   # None: the base MiniLM-L6
    accepted: List[np.ndarray] = field(default_factory=list)
    harmful_filtered: int = 0

    def __post_init__(self) -> None:
        self.mode = check_mode(self.mode)
        if self.harm_verify_at is None:
            self.harm_verify_at = harm_verify_at()
        if self.embed is None:
            from rl_generator.seeds import base_embed

            self.embed = base_embed
        bank = np.asarray(self.novelty_bank, dtype=np.float32)
        self.novelty_bank = bank if bank.ndim == 2 else bank.reshape(0, 0)

    def accept(self, scored: "Scored") -> None:
        """Adds an accepted sample's embedding to the run's part of the bank (SEC-206)."""
        if scored.embedding is not None:
            self.accepted.append(np.asarray(scored.embedding, dtype=np.float32))

    def record(self) -> dict:
        return {"mode": self.mode, "lam_div": self.lam_div, "lam_len": self.lam_len,
                "judge_min_miss": self.judge_min_miss, "harm_verify_at": self.harm_verify_at,
                "target": str(getattr(self.target, "version", "") or "")}


@dataclass
class Scored:
    reward: float = 0.0
    miss: float = 0.0
    novelty: float = 0.0
    overlong: int = 0
    harmless: Optional[bool] = None   # None: empty, not checked
    valid: Optional[bool] = None      # None: not judged (counts as 0)
    p_aligned: Optional[float] = None
    risk: Optional[int] = None
    empty: bool = False
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def selection_score(self, lam_div: float) -> float:
        """`miss + lam_div * novelty`: how prompted.py picks the samples it judges (SEC-208)."""
        return self.miss + lam_div * self.novelty

    def judgeable(self) -> bool:
        return bool(self.harmless) and not self.overlong


def clean(completion: str, action: str) -> str:
    """Strip whitespace, a leading `<action>:` and surrounding quotes; keep the first line."""
    text = (completion or "").strip()
    text = text.splitlines()[0].strip() if text else ""
    if action and text.lower().startswith(f"{action.lower()}:"):
        text = text[len(action) + 1:].strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text


def miss_of(mode: str, p_aligned: float, risk: int) -> float:
    return float(p_aligned) if mode == "H" else max(1.0 - float(p_aligned), float(risk) / 100.0)


def _pair_tokens(tokenizer: Any, intent: str, call_text: str) -> int:
    return len(tokenizer(intent, call_text)["input_ids"])


def _novelty(ctx: RewardContext, emb: np.ndarray) -> np.ndarray:
    # Scored part by part: stacking the ~25k-row bank on every call would copy ~40 MB each time.
    parts = [ctx.novelty_bank] if ctx.novelty_bank.size else []
    if ctx.accepted:
        parts.append(np.stack([a.reshape(-1) for a in ctx.accepted]))
    if not parts:
        return np.ones(len(emb), dtype=np.float32)
    best = np.max([np.max(p.reshape(-1, emb.shape[1]) @ emb.T, axis=0) for p in parts], axis=0)
    return np.clip(1.0 - best, 0.0, 1.0)


def apply_judge(ctx: RewardContext, seed: dict, call: str, s: Scored) -> Scored:
    """Steps 6 and 7: the judge (only when miss >= judge_min_miss) and the reward."""
    if s.harmless and call and s.miss >= ctx.judge_min_miss and ctx.judge is not None:
        s.valid = bool(ctx.judge(seed["intent"], call, seed["action"]))
    return finish(ctx, s)


def finish(ctx: RewardContext, s: Scored) -> Scored:
    if s.empty or not s.harmless:
        s.reward = 0.0
    else:
        s.reward = int(bool(s.valid)) * (s.miss + ctx.lam_div * s.novelty) - ctx.lam_len * s.overlong
    return s


def score_samples(ctx: RewardContext, seeds: List[dict], calls: List[str], judge: bool = True) -> List[Scored]:
    """Scores completions `calls[i]` for `seeds[i]` (design §4.1). With `judge=False` steps 1–5 only
    (`valid` stays None), for prompted.py's selection before judging."""
    if len(seeds) != len(calls):
        raise ValueError("one seed per call")
    texts = [clean(c, s["action"]) for s, c in zip(seeds, calls)]
    out = [Scored() for _ in texts]
    live = []
    for i, (seed, text) in enumerate(zip(seeds, texts)):
        s = out[i]
        if not text:
            s.empty = True
            continue
        pair = f"{seed['action']}: {text}"
        s.overlong = int(len(text) > MAX_CALL_CHARS
                         or _pair_tokens(ctx.tokenizer, seed["intent"], pair) > MAX_PAIR_TOKENS)
        harm = ctx.harm(text)
        s.harmless = harm is not None and harm < ctx.harm_verify_at
        if not s.harmless:
            ctx.harmful_filtered += 1
            continue
        s.p_aligned, s.risk = ctx.target.score_pair(seed["intent"], pair)
        s.miss = miss_of(ctx.mode, s.p_aligned, s.risk)
        live.append(i)
    if live:
        emb = np.asarray(ctx.embed([texts[i] for i in live]), dtype=np.float32)
        for i, e, nov in zip(live, emb, _novelty(ctx, emb)):
            out[i].embedding, out[i].novelty = e, float(nov)
    for i, s in enumerate(out):
        if judge:
            apply_judge(ctx, seeds[i], texts[i], s)
        else:
            finish(ctx, s)
    return out

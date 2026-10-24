"""Pure logic of the GRPO trainer, testable without trl (rl-generator SEC-209, SEC-210; design §7).

Checkpoints and resume, per-step metrics (numbers only, SNFR-20), the stop rules (max steps via
the trainer, the judge's `must_stop()`, the valid-rate guard of §7.3), checkpoint rotation that
keeps the guard's fallback, `best_step.json` and `sampled_steps.json`. `Controller` holds the
state the TrainerCallback in train.py delegates to; `make_callback` wraps it lazily.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
GUARD_WINDOW = 200
GUARD_MIN_VALID = 0.8
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 4
BEST_STEP = "best_step.json"
METRICS = "metrics.jsonl"
SAMPLED = "sampled_steps.json"
METRIC_KEYS = ("reward_mean", "reward_std", "valid_rate", "judged_share", "miss_mean", "novelty_mean",
               "overlong_share", "harmful_filtered", "judge_calls", "judge_errors", "spent")


def checkpoint_steps(ckpt_dir: Path) -> List[int]:
    """Steps of the `checkpoint-N` directories under `ckpt_dir`, ascending."""
    if not Path(ckpt_dir).is_dir():
        return []
    return sorted(int(m.group(1)) for p in Path(ckpt_dir).iterdir()
                  if p.is_dir() and (m := CHECKPOINT.match(p.name)))


def checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    return Path(ckpt_dir) / f"checkpoint-{int(step)}"


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """The newest checkpoint to resume from, or None for a fresh run."""
    steps = checkpoint_steps(ckpt_dir)
    return checkpoint_path(ckpt_dir, steps[-1]) if steps else None


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def step_metrics(scored: Sequence, harmful_filtered: int, judge_calls: int, judge_errors: int,
                 spent: float) -> dict:
    """Per-step means and counts of one reward call (§7.2). `valid_rate` is over judged samples;
    miss and novelty means are over harmless samples. Numbers only."""
    n = len(scored)
    rewards = [float(s.reward) for s in scored]
    judged = [s for s in scored if s.valid is not None]
    live = [s for s in scored if s.harmless]
    mean = _mean(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5 if n else 0.0
    return {"samples": n, "reward_mean": round(mean, 4), "reward_std": round(std, 4),
            "valid_rate": round(sum(bool(s.valid) for s in judged) / len(judged), 4) if judged else None,
            "judged_share": round(len(judged) / n, 4) if n else 0.0,
            "judged": len(judged), "valid": sum(bool(s.valid) for s in judged),
            "miss_mean": round(_mean([s.miss for s in live]), 4),
            "novelty_mean": round(_mean([s.novelty for s in live]), 4),
            "overlong_share": round(sum(bool(s.overlong) for s in scored) / n, 4) if n else 0.0,
            "harmful_filtered": int(harmful_filtered), "judge_calls": int(judge_calls),
            "judge_errors": int(judge_errors), "spent": round(float(spent), 6)}


def slope(ys: Sequence[float]) -> float:
    """Least-squares slope of `ys` against 0..n-1."""
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, _mean(ys)
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    return num / sum((i - mx) ** 2 for i in range(n))


def guard_fires(history: Sequence[dict], window: int = GUARD_WINDOW, min_valid: float = GUARD_MIN_VALID) -> bool:
    """§7.3: over the last `window` steps (a full window only), the judged-valid rate is below
    `min_valid` while `miss_mean` rises (positive slope)."""
    if window <= 0 or len(history) < window:
        return False
    recent = list(history)[-window:]
    judged = sum(int(m.get("judged", 0)) for m in recent)
    if not judged:
        return False
    valid_rate = sum(int(m.get("valid", 0)) for m in recent) / judged
    return valid_rate < min_valid and slope([float(m.get("miss_mean", 0.0)) for m in recent]) > 0


def guard_anchor(saved: Iterable[int], step: int, window: int = GUARD_WINDOW) -> Optional[int]:
    """The last checkpoint saved before the window ending at `step` began."""
    before = [s for s in saved if s <= step - window]
    return max(before) if before else None


def best_step(saved: Iterable[int], fired_at: Optional[int], window: int = GUARD_WINDOW) -> Optional[int]:
    """§7.3: the guard's anchor when it fired, else the last checkpoint (None when there is none)."""
    saved = sorted(saved)
    if fired_at is not None:
        return guard_anchor(saved, fired_at, window)
    return saved[-1] if saved else None


def to_delete(saved: Iterable[int], step: int, limit: int = SAVE_TOTAL_LIMIT, window: int = GUARD_WINDOW) -> List[int]:
    """Checkpoints to rotate out after saving at `step`: keep the newest `limit` plus the guard's
    current anchor, so `best_step` can still point at it if the guard fires later."""
    saved = sorted(saved)
    keep = set(saved[-limit:]) if limit > 0 else set(saved)
    anchor = guard_anchor(saved, step, window)
    if anchor is not None:
        keep.add(anchor)
    return [s for s in saved if s not in keep]


def load_metrics(ckpt_dir: Path, upto: Optional[int]) -> List[dict]:
    """Logged steps up to the resumed checkpoint; later lines (steps to be redone) are dropped
    from the file. A fresh run (`upto` None) starts an empty log."""
    path = Path(ckpt_dir) / METRICS
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if upto is not None and int(r.get("step", 0)) <= upto]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    return kept


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_best_step(ckpt_dir: Path) -> Optional[int]:
    path = Path(ckpt_dir) / BEST_STEP
    if not path.is_file():
        raise SystemExit(f"no {BEST_STEP} in --ckpt-dir: train first or pass --step")
    value = json.loads(path.read_text(encoding="utf-8")).get("best_step")
    if value is None:
        raise SystemExit("best_step is null (no checkpoint before the guard's window): pass --step")
    return int(value)


def read_sampled(ckpt_dir: Path) -> dict:
    path = Path(ckpt_dir) / SAMPLED
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"rl": [int(s) for s in data.get("rl", [])], "heldout": [int(s) for s in data.get("heldout", [])]}


def record_sampled(ckpt_dir: Path, step: int, heldout: bool) -> dict:
    data = read_sampled(ckpt_dir)
    key = "heldout" if heldout else "rl"
    if int(step) not in data[key]:
        data[key] = sorted(data[key] + [int(step)])
    write_json(Path(ckpt_dir) / SAMPLED, data)
    return data


class Controller:
    """What the TrainerCallback does, without transformers: after each step, log the reward
    call's metrics and apply the stop rules; after each save, rotate checkpoints; at the end,
    write `best_step.json`."""

    def __init__(self, ckpt_dir: Path, judge, history: Optional[List[dict]] = None, window: int = GUARD_WINDOW,
                 min_valid: float = GUARD_MIN_VALID, limit: int = SAVE_TOTAL_LIMIT, log=print):
        self.ckpt_dir, self.judge, self.window, self.min_valid, self.limit = Path(ckpt_dir), judge, window, min_valid, limit
        self.history: List[dict] = list(history or [])
        self.pending: Optional[dict] = None
        self.stopped: Optional[str] = None
        self.steps_this_run = 0
        self.fired_at: Optional[int] = None
        self.log = log

    def record(self, metrics: dict) -> None:
        """Called by the reward function with one step's metrics."""
        self.pending = metrics

    def on_step_end(self, step: int) -> bool:
        """True when training must stop after `step` (the caller saves a checkpoint then)."""
        if self.pending is not None:
            row = {"step": int(step), **self.pending}
            self.pending = None
            self.history.append(row)
            self.steps_this_run += 1
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            with (self.ckpt_dir / METRICS).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            self.log(json.dumps(row))
        if guard_fires(self.history, self.window, self.min_valid):
            self.stopped, self.fired_at = "valid-rate guard", int(step)
        elif self.judge is not None and self.judge.must_stop(self._cost_per_step()):
            self.stopped = "judge stop rule"
        if self.stopped:
            self.log(json.dumps({"step": int(step), "stopping": self.stopped}))
        return self.stopped is not None

    def _cost_per_step(self) -> float:
        # the judge's spending and the credit left are this process's
        return float(self.judge.spent()) / self.steps_this_run if self.steps_this_run else 0.0

    def on_save(self, step: int) -> List[int]:
        gone = to_delete(checkpoint_steps(self.ckpt_dir), step, self.limit, self.window)
        for s in gone:
            shutil.rmtree(checkpoint_path(self.ckpt_dir, s), ignore_errors=True)
        return gone

    def finish(self, last_step: int) -> dict:
        saved = checkpoint_steps(self.ckpt_dir)
        out = {"best_step": best_step(saved, self.fired_at, self.window), "last_step": int(last_step),
               "stopped": self.stopped, "guard_fired_at": self.fired_at, "checkpoints": saved}
        write_json(self.ckpt_dir / BEST_STEP, out)
        return out


def make_callback(controller: Controller):
    """A transformers TrainerCallback delegating to `controller` (imported lazily)."""
    from transformers import TrainerCallback

    class GuardCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if controller.on_step_end(state.global_step):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            controller.on_save(state.global_step)
            return control

    return GuardCallback()

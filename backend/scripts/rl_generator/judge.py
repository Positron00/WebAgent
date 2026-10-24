"""The Together judge of a sample's label (rl-generator SEC-203, SNFR-23, SNFR-24; design §4.4).

One call per sample at temperature 0 with the mode's judge prompt, answering JSON
`{"label_ok": true|false}`; an unparsable answer, a timeout or an error is `None` (counted as
false). Answers are cached in SQLite by a hash of (model, prompt version, mode, intent, action,
call), so a resumed run doesn't pay twice. The key comes from `TOGETHER_API_KEY` in the environment
only. `must_stop()` holds the credit rule (as generate_alignment_calls.py) and the error rule:
more than 20% failed calls in the current training step or in the last 100 seeds.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional

from rl_generator import GENERATED_DIR, check_mode
from rl_generator.prompts import PROMPT_VERSION, judge_prompt

CACHE = GENERATED_DIR / "judge_cache.sqlite"
MAX_TOKENS = 16
ERROR_SHARE = 0.2
ERROR_WINDOW_SEEDS = 100
MIN_CALLS_FOR_ERROR_RULE = 10  # so one early failure doesn't stop a run

# $ per million (input, output) tokens per candidate judge model. Empty on purpose: G3/G6 fill it
# from Together's models list on the day they are read (with that date). Until then a run must
# pass --price-input and --price-output, so nobody relies on invented numbers.
PRICES: Dict[str, tuple] = {}

_JSON = re.compile(r"\{[^{}]*\}")


def prices_for(model: str, price_in: Optional[float], price_out: Optional[float]) -> tuple:
    """(input, output) prices: the arguments when given, else `PRICES`; SystemExit when neither."""
    default = PRICES.get(model)
    if price_in is None or price_out is None:
        if default is None:
            raise SystemExit(f"no price for judge model {model!r}: pass --price-input and --price-output")
        price_in = default[0] if price_in is None else price_in
        price_out = default[1] if price_out is None else price_out
    return float(price_in), float(price_out)


def parse(text: str) -> Optional[bool]:
    """`label_ok` from the judge's answer; None when there is no JSON boolean."""
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(0)).get("label_ok")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def _default_client(model: str, timeout_s: float):
    """The app's Together client (TogetherWrapper; 3 attempts with back-off on 429 and 5xx)."""
    from app.services import together_llm
    from app.services.llm import TogetherWrapper

    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("TOGETHER_API_KEY is not set in the environment")
    together_llm.REQUEST_TIMEOUT_SECONDS = timeout_s  # this process only calls the judge
    return TogetherWrapper(model_name=model, temperature=0.0, max_tokens=MAX_TOKENS, api_key=key)


def _default_tokens() -> Dict[str, int]:
    """The app's running token totals for the configured provider (what
    generate_alignment_calls.tokens_so_far reads; not imported, because that module sets
    process-wide guard and drift settings on import)."""
    from app.core import observability
    from app.core.config import settings

    totals = observability._model_token_totals.get(settings.LLM_DEFAULT_PROVIDER or "together", {})
    return {"prompt": int(totals.get("prompt", 0)), "completion": int(totals.get("completion", 0))}


class Judge:
    def __init__(self, model: str, mode: str, cache_path: Path = CACHE, credit_left: Optional[float] = None,
                 price_in: Optional[float] = None, price_out: Optional[float] = None,
                 timeout_s: float = 20.0, *, client=None,
                 tokens_fn: Optional[Callable[[], Dict[str, int]]] = None, default_model: Optional[str] = None):
        self.model = model
        self.mode = check_mode(mode)
        if default_model is None:
            from app.core.config import settings

            default_model = settings.LLM_DEFAULT_MODEL
        if model == default_model:
            raise SystemExit("the judge model must differ from the app's default model (SEC-203)")
        self.price_in, self.price_out = prices_for(model, price_in, price_out)
        self.credit_left = credit_left
        self._client = client if client is not None else _default_client(model, timeout_s)
        self._tokens_fn = tokens_fn or _default_tokens
        self.tokens = {"prompt": 0, "completion": 0}
        self.calls = self.errors = self.cache_hits = 0
        self._step = [0, 0]  # calls, errors in the current training step
        self._seed = [0, 0]  # in the current seed
        self._seeds: deque = deque(maxlen=ERROR_WINDOW_SEEDS)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path))
        self._db.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, label_ok INTEGER)")

    def key(self, intent: str, action: str, call: str) -> str:
        parts = (self.model, PROMPT_VERSION, self.mode, intent, action, call)
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def __call__(self, intent: str, call: str, action: str = "") -> Optional[bool]:
        key = self.key(intent, action, call)
        row = self._db.execute("SELECT label_ok FROM verdicts WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.cache_hits += 1
            return bool(row[0])
        before = self._tokens_fn()
        try:
            answer = parse(str(self._client.invoke(judge_prompt(self.mode, intent, action, call))))
        except Exception:  # noqa: BLE001 - a failed call counts as false (SEC-203)
            answer = None
        after = self._tokens_fn()
        for k in self.tokens:
            self.tokens[k] += max(0, int(after.get(k, 0)) - int(before.get(k, 0)))
        failed = answer is None
        self.calls += 1
        self.errors += failed
        for counter in (self._step, self._seed):
            counter[0] += 1
            counter[1] += failed
        if not failed:
            self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?)", (key, int(answer)))
            self._db.commit()
        return answer

    def begin_step(self) -> None:
        self._step = [0, 0]

    def end_seed(self) -> None:
        self._seeds.append(tuple(self._seed))
        self._seed = [0, 0]

    def spent(self) -> float:
        # generate_alignment_calls.cost's formula; that module isn't imported (see _default_tokens).
        return self.tokens["prompt"] / 1e6 * self.price_in + self.tokens["completion"] / 1e6 * self.price_out

    def error_stop(self) -> bool:
        window = [sum(c[i] for c in self._seeds) for i in (0, 1)]
        return any(calls >= MIN_CALLS_FOR_ERROR_RULE and errors > ERROR_SHARE * calls
                   for calls, errors in (tuple(self._step), tuple(window)))

    def must_stop(self, next_cost: float = 0.0) -> bool:
        """True when spending `next_cost` more would leave less than half of the credit entered,
        or on the error rule."""
        if self.error_stop():
            return True
        return self.credit_left is not None and self.spent() + next_cost >= self.credit_left / 2

    def report(self) -> dict:
        return {"model": self.model, "calls": self.calls, "errors": self.errors, "cache_hits": self.cache_hits,
                "tokens": dict(self.tokens), "cost_usd": round(self.spent(), 4)}

    def close(self) -> None:
        self._db.close()

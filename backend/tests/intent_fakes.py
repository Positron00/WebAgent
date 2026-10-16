"""Fakes for the checkpoint2-intent tests: a bag-of-words encoder and a keyword harm scorer."""
import hashlib
import re
import time

import numpy as np

from app.core.config import settings


class WordEncoder:
    """Hashes words into 64 dimensions: texts sharing words are close, unrelated texts near 0."""

    def __init__(self, delay: float = 0.0):
        self.calls = []
        self.delay = delay

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for text in texts:
            self.calls.append(text)
            if self.delay:
                time.sleep(self.delay)
            vec = np.zeros(64)
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm else vec)
        return np.asarray(out)


def keyword_harm(text: str) -> int:
    """90 for texts naming a weapon or an attack, else 5."""
    return 90 if re.search(r"\b(explosive|weapon|exploit|malware)\b", text.lower()) else 5


def guard_on(monkeypatch, tmp_path, *, tool_mode="shadow", env="uat", **extra):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "WEBAGENT_ENV", env)
    monkeypatch.setattr(settings, "SECURITY_GUARD", settings.SECURITY_GUARD.model_copy(
        update={"enabled": True, "tool_mode": tool_mode, "collect": True, "judge": False, **extra}))

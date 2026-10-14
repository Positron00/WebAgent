"""Spot Git LFS pointer files, so tests that need real weights or data can skip on an LFS-less checkout."""
from pathlib import Path

_POINTER_PREFIX = b"version https://git-lfs"
_MAX_POINTER_BYTES = 1024


def is_lfs_pointer(path) -> bool:
    """True when `path` is a small text file holding a Git LFS pointer instead of the real content."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size > _MAX_POINTER_BYTES:
            return False
        with p.open("rb") as f:
            return f.read(len(_POINTER_PREFIX)) == _POINTER_PREFIX
    except OSError:
        return False

"""Helpers for reading JSON out of LLM replies."""
import json
import re
from typing import Any, Dict, Optional

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_MAX_START_POSITIONS = 50


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object in `text`, ignoring code fences and any text around it.

    Returns None if no complete JSON object can be found.
    """
    cleaned = _FENCE.sub("", text.strip())
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    tried = 0
    while start != -1 and tried < _MAX_START_POSITIONS:
        try:
            parsed, _end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        start = cleaned.find("{", start + 1)
        tried += 1
    return None

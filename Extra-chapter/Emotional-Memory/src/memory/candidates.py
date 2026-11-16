"""Deterministic, conservative extraction of user-memory candidates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    fact_key: str
    proposed_text: str
    confidence: float


_BOUNDARY = r"(?:^|[，,。.!！？；;\s])"
_NAME = re.compile(
    _BOUNDARY + r"(?:我叫|我的名字是)\s*([A-Za-z0-9_\-\u3400-\u9fff·]{1,40})(?=$|[，,。.!！？；;\s])",
    re.IGNORECASE,
)
_BIRTHDAY = re.compile(
    _BOUNDARY + r"(?:我的生日是|我生日是|我的生日|我生日)\s*"
    r"([0-9]{1,4}(?:年|[-/.])?[0-9]{1,2}(?:月|[-/.])?[0-9]{0,2}日?)"
    r"(?=$|[，,。.!！？；;\s])",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    _BOUNDARY + r"(我喜欢|我讨厌|我害怕)\s*([^，,。.!！？；;\r\n]{1,160})",
    re.IGNORECASE,
)


def _stable_key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def extract_memory_candidate(user_message: str) -> MemoryCandidate | None:
    """Return one high-precision candidate, never an inferred assistant claim."""

    cleaned = str(user_message or "").strip()
    if not cleaned or len(cleaned) > 500:
        return None

    match = _NAME.search(cleaned)
    if match:
        value = match.group(1).strip()
        return MemoryCandidate("user:identity:name", f"用户姓名：{value}", 0.98)

    match = _BIRTHDAY.search(cleaned)
    if match:
        value = match.group(1).strip()
        return MemoryCandidate("user:identity:birthday", f"用户生日：{value}", 0.98)

    match = _PREFERENCE.search(cleaned)
    if match:
        marker = match.group(1)
        value = match.group(2).strip()
        label = {"我喜欢": "用户喜欢", "我讨厌": "用户讨厌", "我害怕": "用户害怕"}[
            marker
        ]
        polarity = {"我喜欢": "likes", "我讨厌": "dislikes", "我害怕": "fears"}[
            marker
        ]
        return MemoryCandidate(
            _stable_key(f"user:preference:{polarity}", value),
            f"{label}：{value}",
            0.92,
        )
    return None


def explicit_fact_key(text: str) -> str:
    """Give a user-confirmed free-form fact a stable idempotency key."""

    cleaned = str(text).strip()
    candidate = extract_memory_candidate(cleaned)
    if candidate is not None:
        return candidate.fact_key
    return _stable_key("user:confirmed", cleaned)

"""Temporal intent detection for memory retrieval enhancement.

Detects whether a query has temporal intent (recent, historical, when)
and provides a scoring modifier for retrieval ranking.

Inspired by Mem0's temporal reasoning and Zep/Graphiti's validity windows.
"""

from __future__ import annotations

import re
from typing import Literal

TemporalIntent = Literal["recent", "historical", "when", None]

# ── Keyword tables (Chinese + English) ───────────────────────────────

_RECENT_KEYWORDS = re.compile(
    r"现在|目前|最近|当前|如今|眼下|此刻|当下"
    r"|now|current(?:ly)?|recent(?:ly)?|latest|at\s+the\s+moment|these\s+days",
    re.IGNORECASE,
)

_HISTORICAL_KEYWORDS = re.compile(
    r"以前|曾经|之前|过去|从前|原来|那时|当时|早先|原先"
    r"|before|used\s+to|formerly|previously|back\s+then|in\s+the\s+past",
    re.IGNORECASE,
)

_WHEN_KEYWORDS = re.compile(
    r"什么时候|哪天|哪一天|几号|几月|何时"
    r"|when|what\s+time|what\s+date|which\s+day",
    re.IGNORECASE,
)


def detect_temporal_intent(query: str) -> TemporalIntent:
    """Classify the temporal intent of a query.

    Returns:
        "recent"     — query asks about current state (prefer newer memories)
        "historical" — query asks about past (don't penalize old memories)
        "when"       — query asks for a time/date (sort by event_time)
        None         — no temporal signal detected (default behavior)
    """

    if not query:
        return None

    text = query.strip()

    # Priority: "when" > "recent" > "historical"
    # ("什么时候" is unambiguous; "最近" vs "以前" resolved by first match)
    if _WHEN_KEYWORDS.search(text):
        return "when"
    if _RECENT_KEYWORDS.search(text):
        return "recent"
    if _HISTORICAL_KEYWORDS.search(text):
        return "historical"
    return None


def temporal_recency_boost(intent: TemporalIntent, age_days: float) -> float:
    """Compute an additive boost/penalty based on temporal intent.

    Returns a value in [-0.08, +0.08] that can be added to the retrieval score.
    When intent is None, returns 0.0 (no effect).

    Args:
        intent: The detected temporal intent.
        age_days: Days since the memory was last updated.
    """

    if intent is None:
        return 0.0

    max_boost = 0.08

    if intent == "recent":
        # Newer memories get a positive boost, older ones get penalized
        if age_days <= 7:
            return max_boost
        elif age_days <= 30:
            return max_boost * (1.0 - (age_days - 7) / 23.0)
        else:
            return -max_boost * min(1.0, (age_days - 30) / 60.0)

    if intent == "historical":
        # Don't penalize old memories; slight boost for older ones
        if age_days >= 30:
            return max_boost * 0.5  # mild boost for genuinely old memories
        return 0.0

    if intent == "when":
        # Neutral — temporal sorting is done at the ranking stage, not via boost
        return 0.0

    return 0.0

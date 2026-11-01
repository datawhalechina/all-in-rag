"""Lightweight entity co-occurrence index for memory retrieval boosting.

Inspired by Mem0's entity-linking graph (Apache-2.0, 2026-04 algorithm):
 entities are extracted via regex at store time, linked to memories via a
 junction table, and used at retrieval time to boost co-occurring memories.

 No LLM calls.  No external graph database.  Pure SQLite.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Sequence

# ── Entity extraction patterns (extend candidates.py coverage) ────────

# Chinese person names: 2-3 CJK chars after a relationship word.
# We capture the full CJK run then trim to the first 2–3 chars since
# Chinese names are almost always 2–3 characters (surname + given name).
_PERSON_CN_RAW = re.compile(
    r"(?:我(?:的)?(?:朋友|同学|同事|老师|男友|女友|老公|老婆|爸|妈|哥|姐|弟|妹"
    r"|儿子|女儿|爷爷|奶奶|外公|外婆|叔叔|阿姨|舅舅|闺蜜|室友|邻居))"
    r"([\u3400-\u9fff]+)"
)


def _extract_cn_person(text: str) -> list[str]:
    """Extract Chinese person names, trimmed to 2-3 chars."""
    results = []
    for match in _PERSON_CN_RAW.finditer(text):
        raw = match.group(1)
        # Chinese names are 2-3 chars; take the shorter valid prefix
        name = raw[:2] if len(raw) >= 2 else raw
        if len(raw) >= 3:
            # 3-char names are common (e.g. 欧阳明); accept if run is exactly 3
            # or if the 3rd char is a plausible name char (not a common verb/particle)
            _COMMON_NON_NAME = set("的了和与跟在去到来说是有也都还很不会要让把被给向从对"
                                   "想做看听玩吃喝打买今昨明最近说喜欢讨厌害怕觉得已经"
                                   "刚才正在一起")
            if len(raw) == 3 or raw[2] not in _COMMON_NON_NAME:
                name = raw[:3]
        if len(name) >= 2:
            results.append(name)
    return results
_PERSON_EN = re.compile(
    r"(?:my\s+(?:friend|colleague|partner|teacher|boss)\s+)"
    r"([A-Z][a-z]+)",
)
_LOCATION = re.compile(
    r"(?:在|去|到|来自|住在|搬到)\s*"
    r"([\u3400-\u9fff]{2,10}(?:市|省|区|县|镇|村|路|街|大学|学院|公司|医院|小区))"
)
_WORK_TITLE = re.compile(r"《([^》]{1,30})》")
_WORK_VERB = re.compile(
    r"(?:看了|读了|听了|玩了|追了|在看|在玩|在听|在读)\s*"
    r"([\u3400-\u9fffA-Za-z0-9]{2,20})"
)
_EVENT = re.compile(
    r"(生日|婚礼|毕业|考试|面试|旅行|搬家|手术|比赛|聚会|约会|出差|开会|纪念日)"
)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("person", _PERSON_EN),
    ("location", _LOCATION),
    ("work", _WORK_TITLE),
    ("work", _WORK_VERB),
    ("event", _EVENT),
]


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return (entity_type, entity_text) pairs from text.  Deterministic, cheap."""

    if not text or len(text) > 8192:
        return []
    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, str]] = []

    # Chinese person names via special trimming logic
    for name in _extract_cn_person(text):
        key = ("person", name.casefold())
        if key not in seen:
            seen.add(key)
            results.append(("person", name))

    # All other patterns via standard regex
    for entity_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = (match.group(1) if match.lastindex else match.group(0)).strip()
            if not value or len(value) < 2 or len(value) > 100:
                continue
            key = (entity_type, value.casefold())
            if key not in seen:
                seen.add(key)
                results.append((entity_type, value))
    return results[:30]


# ── Schema creation (called from catalog._create_schema) ─────────────

ENTITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    normalized TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    UNIQUE(normalized, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entity_normalized
    ON memory_entities(normalized, entity_type);

CREATE TABLE IF NOT EXISTS memory_entity_links (
    entity_id INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (entity_id, memory_id),
    FOREIGN KEY (entity_id) REFERENCES memory_entities(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_entity_links_memory
    ON memory_entity_links(memory_id);
"""


def create_entity_tables(connection: sqlite3.Connection) -> None:
    """Idempotent entity table creation."""
    connection.executescript(ENTITY_SCHEMA_SQL)


# ── Indexing (called after catalog upsert) ────────────────────────────

def index_entities(
    connection: sqlite3.Connection,
    memory_id: str,
    text: str,
    *,
    now: float | None = None,
) -> int:
    """Extract entities from text and link them to the memory.  Returns count."""

    entities = extract_entities(text)
    if not entities:
        return 0

    ts = now or time.time()
    count = 0
    for entity_type, entity_text in entities:
        normalized = entity_text.casefold()
        connection.execute(
            """INSERT INTO memory_entities (entity_text, entity_type, normalized,
                                           first_seen_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(normalized, entity_type) DO UPDATE SET
                   last_seen_at = MAX(memory_entities.last_seen_at, excluded.last_seen_at)""",
            (entity_text, entity_type, normalized, ts, ts),
        )
        entity_row = connection.execute(
            "SELECT id FROM memory_entities WHERE normalized=? AND entity_type=?",
            (normalized, entity_type),
        ).fetchone()
        if entity_row is None:
            continue
        entity_id = int(entity_row[0])
        connection.execute(
            """INSERT INTO memory_entity_links (entity_id, memory_id, mention_count)
               VALUES (?, ?, 1)
               ON CONFLICT(entity_id, memory_id) DO UPDATE SET
                   mention_count = memory_entity_links.mention_count + 1""",
            (entity_id, memory_id),
        )
        count += 1
    return count


# ── Retrieval boost (called during candidate scoring) ─────────────────

def entity_boost_candidates(
    connection: sqlite3.Connection,
    query_text: str,
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    """Return {memory_id: boost} for candidates sharing entities with the query.

    Boost is additive (0.0 to 0.08), proportional to the number of shared
    entity types.  Memories with no entity overlap get 0.0.
    """

    if not candidate_ids or not query_text:
        return {}

    # Strategy 1: Extract entities from query using full patterns
    query_entities = extract_entities(query_text)
    entity_ids: set[int] = set()
    for entity_type, entity_text in query_entities:
        normalized = entity_text.casefold()
        row = connection.execute(
            "SELECT id FROM memory_entities WHERE normalized=? AND entity_type=?",
            (normalized, entity_type),
        ).fetchone()
        if row:
            entity_ids.add(int(row[0]))

    # Strategy 2: Direct substring match — if the query mentions a stored
    # entity name without a relationship prefix (e.g. "小明最近怎么样"),
    # find it by scanning stored entity names against the query text.
    query_lower = query_text.casefold()
    rows = connection.execute(
        "SELECT id, normalized FROM memory_entities"
    ).fetchall()
    for row in rows:
        eid, norm = int(row[0]), str(row[1])
        if len(norm) >= 2 and norm in query_lower:
            entity_ids.add(eid)

    if not entity_ids:
        return {}

    # Find which candidates link to these entities
    placeholders_eid = ",".join("?" for _ in entity_ids)
    placeholders_mid = ",".join("?" for _ in candidate_ids)
    hits = connection.execute(
        f"""SELECT memory_id, COUNT(DISTINCT entity_id) AS shared
            FROM memory_entity_links
            WHERE entity_id IN ({placeholders_eid})
              AND memory_id IN ({placeholders_mid})
            GROUP BY memory_id""",
        [*entity_ids, *candidate_ids],
    ).fetchall()

    # Scale: 1 shared entity = 0.04, 2+ = 0.08 (cap)
    max_boost = 0.08
    boost_per_entity = 0.04
    return {
        str(row[0]): min(max_boost, int(row[1]) * boost_per_entity)
        for row in hits
    }

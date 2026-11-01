"""Transactional canonical store for Reverie's long-lived memories."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator

from src.storage.encrypted_sqlite import connect_database

from .lifecycle_constants import protection_tier_for_text

RETENTION_LAYERS = {"permanent", "long_term", "short_term"}
COGNITIVE_LAYERS = {"episodic", "semantic", "procedural"}
USAGE_SIGNALS = {"injected", "reply_overlap", "user_confirmed", "user_rejected", "auto_reinforce"}


def memory_terms(text: str) -> list[str]:
    """Produce bounded English tokens and Chinese bi/tri-grams."""
    # Bound work before creating n-grams; an imported blob must not amplify into
    # hundreds of thousands of temporary strings on the desktop process.
    lowered = text[:8192].lower()
    terms = re.findall(r"[a-z0-9_]{2,}", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        if len(run) <= 4:
            terms.append(run)
        for width in (2, 3):
            terms.extend(run[i:i + width] for i in range(max(0, len(run) - width + 1)))
    return list(dict.fromkeys(terms))[:80]


class MemoryCatalog:
    """SQLite source of truth; vector databases are disposable derivatives."""

    def __init__(self, path: str | Path, *, fold_worker=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fold_worker = fold_worker
        self._connection = connect_database(
            self.path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                retention_layer TEXT NOT NULL,
                cognitive_layer TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_time REAL,
                importance REAL NOT NULL,
                emotions_json TEXT NOT NULL DEFAULT '{}',
                source_type TEXT NOT NULL DEFAULT 'local_interaction',
                source_uri TEXT NOT NULL DEFAULT '',
                source_hash TEXT NOT NULL DEFAULT '',
                trust_level TEXT NOT NULL DEFAULT 'trusted_local',
                sanitizer_status TEXT NOT NULL DEFAULT 'not_required',
                sanitizer_flags_json TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '',
                embedding_model_version TEXT NOT NULL,
                embedding_status TEXT NOT NULL DEFAULT 'pending',
                last_accessed REAL,
                access_count INTEGER NOT NULL DEFAULT 0,
                mention_count INTEGER NOT NULL DEFAULT 1,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                fact_key TEXT NOT NULL DEFAULT '',
                fact_revision INTEGER NOT NULL DEFAULT 0,
                supersedes_id TEXT NOT NULL DEFAULT '',
                confirmation_state TEXT NOT NULL DEFAULT 'observed',
                confirmed_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                CHECK (retention_layer IN ('permanent','long_term','short_term')),
                CHECK (cognitive_layer IN ('episodic','semantic','procedural'))
            );
            CREATE INDEX IF NOT EXISTS idx_memory_retention_time
                ON memory_records(retention_layer, timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_cognitive_time
                ON memory_records(cognitive_layer, timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_importance
                ON memory_records(importance DESC, timestamp DESC);
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                vector_table TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                status TEXT NOT NULL,
                embedded_at REAL,
                error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (memory_id, model_version),
                FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_version_status
                ON memory_embeddings(model_version, status);
            CREATE TABLE IF NOT EXISTS memory_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                referenced_at REAL NOT NULL,
                reason TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_references_memory_time
                ON memory_references(memory_id, referenced_at DESC);
            CREATE TABLE IF NOT EXISTS memory_recall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                query TEXT NOT NULL,
                true_text TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                source_entity TEXT NOT NULL,
                substitute_entity TEXT NOT NULL,
                domain TEXT NOT NULL,
                created_at REAL NOT NULL,
                corrected_at REAL,
                FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_recall_events_open
                ON memory_recall_events(corrected_at, created_at DESC);
            CREATE TABLE IF NOT EXISTS memory_system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_candidates (
                id TEXT PRIMARY KEY,
                fact_key TEXT NOT NULL,
                proposed_text TEXT NOT NULL,
                source_text TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_uri TEXT NOT NULL DEFAULT '',
                source_hash TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                committed_memory_id TEXT NOT NULL DEFAULT '',
                conflict_memory_id TEXT NOT NULL DEFAULT '',
                decision_reason TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                decided_at REAL,
                CHECK (status IN ('pending','confirmed','rejected'))
            );
            CREATE INDEX IF NOT EXISTS idx_memory_candidates_status_time
                ON memory_candidates(status,created_at,id);
            CREATE TABLE IF NOT EXISTS memory_usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                weight REAL NOT NULL,
                turn_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_usage_events_memory
                ON memory_usage_events(memory_id,created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_usage_events_created
                ON memory_usage_events(created_at);
            """
        )
        # Forward-compatible migration for databases created before lifecycle
        # retention was separated from storage capacity.
        columns = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(memory_records)")
        }
        if "lifecycle_state" not in columns:
            self._connection.execute(
                "ALTER TABLE memory_records ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'active'"
            )
        for name, definition in (
            ("fact_key", "TEXT NOT NULL DEFAULT ''"),
            ("fact_revision", "INTEGER NOT NULL DEFAULT 0"),
            ("supersedes_id", "TEXT NOT NULL DEFAULT ''"),
            ("confirmation_state", "TEXT NOT NULL DEFAULT 'observed'"),
            ("confirmed_at", "REAL"),
            ("protection_tier", "INTEGER NOT NULL DEFAULT 0"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE memory_records ADD COLUMN {name} {definition}"
                )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_time "
            "ON memory_records(lifecycle_state,timestamp,id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_fact_current "
            "ON memory_records(fact_key,lifecycle_state,fact_revision DESC)"
        )
        # Entity co-occurrence index (Phase 1B, 2026-09-09)
        from .entity_index import create_entity_tables
        create_entity_tables(self._connection)
        # Evidence chain + typed atoms (Phase 1C, 2026-09-09)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_evidence_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                CHECK (relation IN ('source','derived_from','supports','conflicts','supersedes')),
                FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_refs_memory
                ON memory_evidence_refs(memory_id, relation);
            CREATE INDEX IF NOT EXISTS idx_evidence_refs_target
                ON memory_evidence_refs(target_id, relation);

            CREATE TABLE IF NOT EXISTS memory_atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                atom_type TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                predicate TEXT NOT NULL DEFAULT '',
                object TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.8,
                extraction_reason TEXT NOT NULL DEFAULT '',
                review_status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                CHECK (atom_type IN (
                    'preference','constraint','promise','event','lesson','temporal_fact'
                )),
                CHECK (review_status IN ('pending','approved','rejected')),
                FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_atoms_type_status
                ON memory_atoms(atom_type, review_status);
            CREATE INDEX IF NOT EXISTS idx_atoms_memory
                ON memory_atoms(memory_id);
            """
        )
        # Fold worker job queue (Phase 1D, 2026-09-09)
        from .fold_worker import create_job_tables
        create_job_tables(self._connection)

    @staticmethod
    def _candidate_row(row: sqlite3.Row | dict) -> dict:
        return dict(row)

    def create_candidate(
        self,
        *,
        fact_key: str,
        proposed_text: str,
        source_text: str,
        source_type: str,
        source_uri: str,
        source_hash: str,
        confidence: float,
    ) -> dict:
        fact_key = str(fact_key).strip()
        proposed_text = str(proposed_text).strip()
        source_text = str(source_text).strip()
        source_hash = str(source_hash).strip()
        if not fact_key or len(fact_key) > 500:
            raise ValueError("Memory candidate fact key is invalid")
        if not proposed_text or not source_text or not source_hash:
            raise ValueError("Memory candidate evidence is incomplete")
        digest = hashlib.sha256(
            f"{fact_key}\0{proposed_text}\0{source_hash}".encode("utf-8")
        ).hexdigest()
        candidate_id = f"mc_{digest[:32]}"
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO memory_candidates (
                    id,fact_key,proposed_text,source_text,source_type,source_uri,
                    source_hash,confidence,status,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    candidate_id,
                    fact_key,
                    proposed_text[:2000],
                    source_text[:4000],
                    str(source_type)[:200],
                    str(source_uri)[:4096],
                    source_hash[:500],
                    max(0.0, min(1.0, float(confidence))),
                    "pending",
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id=?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Memory candidate disappeared after insert")
        return self._candidate_row(row)

    def list_candidates(self, *, status: str = "pending", limit: int = 100) -> list[dict]:
        if status not in {"pending", "confirmed", "rejected", "all"}:
            raise ValueError("Unsupported memory candidate status")
        where = "" if status == "all" else "WHERE status=?"
        params: list[object] = [] if status == "all" else [status]
        params.append(max(1, min(1000, int(limit))))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memory_candidates {where} "
                "ORDER BY created_at DESC,id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._candidate_row(row) for row in rows]

    def reject_candidate(self, candidate_id: str, *, reason: str = "user_rejected") -> dict:
        now = time.time()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id=?",
                (str(candidate_id),),
            ).fetchone()
            if row is None:
                raise KeyError("Memory candidate not found")
            if str(row["status"]) == "confirmed":
                raise ValueError("Confirmed memory candidate cannot be rejected")
            connection.execute(
                """UPDATE memory_candidates
                   SET status='rejected',decision_reason=?,decided_at=?,updated_at=?
                   WHERE id=?""",
                (str(reason)[:500], now, now, str(candidate_id)),
            )
            decided = connection.execute(
                "SELECT * FROM memory_candidates WHERE id=?",
                (str(candidate_id),),
            ).fetchone()
        return self._candidate_row(decided)

    def confirm_candidate(
        self,
        candidate_id: str,
        *,
        embedding_model_version: str,
        importance: float = 0.75,
        decision_reason: str = "user_confirmed",
    ) -> dict:
        """Atomically confirm a candidate and supersede a conflicting fact."""

        now = time.time()
        with self.transaction() as connection:
            candidate = connection.execute(
                "SELECT * FROM memory_candidates WHERE id=?",
                (str(candidate_id),),
            ).fetchone()
            if candidate is None:
                raise KeyError("Memory candidate not found")
            if str(candidate["status"]) == "rejected":
                raise ValueError("Rejected memory candidate cannot be confirmed")
            if str(candidate["status"]) == "confirmed":
                memory = connection.execute(
                    "SELECT * FROM memory_records WHERE id=?",
                    (str(candidate["committed_memory_id"]),),
                ).fetchone()
                return {
                    "candidate": self._candidate_row(candidate),
                    "memory": self._row(memory) if memory else None,
                    "idempotent": True,
                }

            fact_key = str(candidate["fact_key"])
            proposed_text = str(candidate["proposed_text"])
            current = connection.execute(
                """SELECT * FROM memory_records
                   WHERE fact_key=? AND lifecycle_state='active'
                     AND confirmation_state='confirmed'
                   ORDER BY fact_revision DESC,updated_at DESC LIMIT 1""",
                (fact_key,),
            ).fetchone()
            if current is not None and str(current["text"]) == proposed_text:
                memory_id = str(current["id"])
                conflict_id = ""
                idempotent = True
            else:
                conflict_id = str(current["id"]) if current is not None else ""
                revision = int(current["fact_revision"] or 0) + 1 if current is not None else 1
                memory_id = f"fact_{hashlib.sha256(str(candidate_id).encode()).hexdigest()[:32]}"
                if current is not None:
                    connection.execute(
                        """UPDATE memory_records
                           SET lifecycle_state='superseded',updated_at=? WHERE id=?""",
                        (now, conflict_id),
                    )
                emotions = json.dumps(
                    {
                        name: 0.0
                        for name in (
                            "joy", "sadness", "anger", "excitement", "calm",
                            "anxiety", "grievance", "touched",
                        )
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                connection.execute(
                    """
                    INSERT INTO memory_records (
                        id,text,retention_layer,cognitive_layer,timestamp,event_time,
                        importance,emotions_json,source_type,source_uri,source_hash,
                        trust_level,sanitizer_status,sanitizer_flags_json,keywords,
                        embedding_model_version,embedding_status,lifecycle_state,
                        fact_key,fact_revision,supersedes_id,confirmation_state,confirmed_at,
                        created_at,updated_at,protection_tier,pinned
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        memory_id,
                        proposed_text,
                        "long_term",
                        "semantic",
                        now,
                        None,
                        max(0.0, min(1.0, float(importance))),
                        emotions,
                        str(candidate["source_type"]),
                        str(candidate["source_uri"]),
                        str(candidate["source_hash"]),
                        "trusted_local",
                        "not_required",
                        "[]",
                        " ".join(memory_terms(proposed_text)),
                        str(embedding_model_version),
                        "pending",
                        "active",
                        fact_key,
                        revision,
                        conflict_id,
                        "confirmed",
                        now,
                        now,
                        now,
                        protection_tier_for_text(proposed_text),
                        0,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_references(memory_id,referenced_at,reason) VALUES (?,?,?)",
                    (memory_id, now, "candidate_confirmed"),
                )
                idempotent = False

            connection.execute(
                """UPDATE memory_candidates
                   SET status='confirmed',committed_memory_id=?,conflict_memory_id=?,
                       decision_reason=?,decided_at=?,updated_at=?
                   WHERE id=?""",
                (
                    memory_id,
                    conflict_id,
                    str(decision_reason)[:500],
                    now,
                    now,
                    str(candidate_id),
                ),
            )
            decided = connection.execute(
                "SELECT * FROM memory_candidates WHERE id=?",
                (str(candidate_id),),
            ).fetchone()
            memory = connection.execute(
                "SELECT * FROM memory_records WHERE id=?",
                (memory_id,),
            ).fetchone()
        return {
            "candidate": self._candidate_row(decided),
            "memory": self._row(memory) if memory else None,
            "idempotent": idempotent,
        }

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _escape_like_term(term: str) -> str:
        """Escape a term for use in a LIKE clause with ESCAPE '\\'."""
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    @staticmethod
    def _row(row: sqlite3.Row | dict) -> dict:
        value = dict(row)
        try:
            emotions = json.loads(value.pop("emotions_json", "{}") or "{}")
        except (TypeError, json.JSONDecodeError):
            emotions = {}
        try:
            flags = json.loads(value.pop("sanitizer_flags_json", "[]") or "[]")
        except (TypeError, json.JSONDecodeError):
            flags = []
        value["emotions"] = emotions if isinstance(emotions, dict) else {}
        value["sanitizer_flags"] = flags if isinstance(flags, list) else []
        value["layer"] = value.get("retention_layer", "short_term")
        for name in ("joy", "sadness", "anger", "excitement", "calm", "anxiety", "grievance", "touched"):
            value[f"emotion_{name}"] = float(value["emotions"].get(name, 0.0) or 0.0)
        return value

    def upsert(
        self,
        *,
        id: str,
        text: str,
        retention_layer: str,
        cognitive_layer: str,
        timestamp: float,
        importance: float,
        emotions: dict[str, float] | None,
        embedding_model_version: str,
        source_type: str = "local_interaction",
        source_uri: str = "",
        source_hash: str = "",
        trust_level: str = "trusted_local",
        sanitizer_status: str = "not_required",
        sanitizer_flags: list[str] | None = None,
        event_time: float | None = None,
    ) -> None:
        payload, now = self._prepare_record(
            id=id,
            text=text,
            retention_layer=retention_layer,
            cognitive_layer=cognitive_layer,
            timestamp=timestamp,
            importance=importance,
            emotions=emotions,
            embedding_model_version=embedding_model_version,
            source_type=source_type,
            source_uri=source_uri,
            source_hash=source_hash,
            trust_level=trust_level,
            sanitizer_status=sanitizer_status,
            sanitizer_flags=sanitizer_flags,
            event_time=event_time,
        )
        with self.transaction() as connection:
            self._insert_record(connection, payload, now)
        # P1D: auto-enqueue fold jobs after successful write
        if self._fold_worker is not None:
            memory_id = payload[0]
            self._fold_worker.enqueue('extract_atoms', {'memory_id': memory_id},
                                     dedupe_key=f'atoms:{memory_id}')
            self._fold_worker.enqueue('evidence_check', {'memory_id': memory_id},
                                     dedupe_key=f'evidence:{memory_id}')

    def _prepare_record(
        self,
        *,
        id: str,
        text: str,
        retention_layer: str,
        cognitive_layer: str,
        timestamp: float,
        importance: float,
        emotions: dict[str, float] | None,
        embedding_model_version: str,
        source_type: str = "local_interaction",
        source_uri: str = "",
        source_hash: str = "",
        trust_level: str = "trusted_local",
        sanitizer_status: str = "not_required",
        sanitizer_flags: list[str] | None = None,
        event_time: float | None = None,
    ) -> tuple[tuple, float]:
        if retention_layer not in RETENTION_LAYERS:
            raise ValueError(f"Unsupported retention layer: {retention_layer}")
        if cognitive_layer not in COGNITIVE_LAYERS:
            raise ValueError(f"Unsupported cognitive layer: {cognitive_layer}")
        if source_type == "untrusted_web" or trust_level == "untrusted_web":
            raise ValueError("Untrusted web content cannot enter the memory catalog")
        memory_id = str(id)
        if not memory_id or memory_id != memory_id.strip() or len(memory_id) > 500:
            raise ValueError("Memory id is empty, padded, or too long")
        cleaned = str(text).strip()
        if not cleaned:
            raise ValueError("Memory text is empty")
        timestamp_value = float(timestamp)
        importance_value = float(importance)
        event_time_value = None if event_time is None else float(event_time)
        if not math.isfinite(timestamp_value) or not math.isfinite(importance_value):
            raise ValueError("Memory timestamp and importance must be finite")
        if event_time_value is not None and not math.isfinite(event_time_value):
            raise ValueError("Memory event timestamp must be finite")
        if emotions is not None and not isinstance(emotions, dict):
            raise ValueError("Memory emotions must be a mapping")
        emotion_names = (
            "joy", "sadness", "anger", "excitement",
            "calm", "anxiety", "grievance", "touched",
        )
        normalised_emotions: dict[str, float] = {}
        for name in emotion_names:
            value = float((emotions or {}).get(name, 0.0) or 0.0)
            if not math.isfinite(value):
                raise ValueError("Memory emotion values must be finite")
            normalised_emotions[name] = max(0.0, min(100.0, value))
        model_version = str(embedding_model_version).strip()
        if not model_version or len(model_version) > 500:
            raise ValueError("Embedding model version is empty or too long")
        flags = [str(flag)[:200] for flag in (sanitizer_flags or [])[:64]]
        now = time.time()
        payload = (
            memory_id, cleaned, retention_layer, cognitive_layer, timestamp_value, event_time_value,
            max(0.0, min(1.0, importance_value)),
            json.dumps(normalised_emotions, ensure_ascii=False, sort_keys=True),
            str(source_type)[:200], str(source_uri)[:4096], str(source_hash)[:500],
            str(trust_level)[:200], str(sanitizer_status)[:200],
            json.dumps(flags, ensure_ascii=False),
            " ".join(memory_terms(cleaned)), model_version, "pending", now, now,
            protection_tier_for_text(cleaned),
        )
        return payload, now

    @staticmethod
    def _insert_record(connection: sqlite3.Connection, payload: tuple, now: float) -> None:
        connection.execute(
            """
            INSERT INTO memory_records (
                id,text,retention_layer,cognitive_layer,timestamp,event_time,importance,
                emotions_json,source_type,source_uri,source_hash,trust_level,
                sanitizer_status,sanitizer_flags_json,keywords,embedding_model_version,
                embedding_status,created_at,updated_at,protection_tier
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text, retention_layer=excluded.retention_layer,
                cognitive_layer=excluded.cognitive_layer, timestamp=excluded.timestamp,
                event_time=excluded.event_time, importance=excluded.importance,
                emotions_json=excluded.emotions_json, source_type=excluded.source_type,
                source_uri=excluded.source_uri, source_hash=excluded.source_hash,
                trust_level=excluded.trust_level, sanitizer_status=excluded.sanitizer_status,
                sanitizer_flags_json=excluded.sanitizer_flags_json, keywords=excluded.keywords,
                embedding_model_version=excluded.embedding_model_version,
                embedding_status='pending', lifecycle_state='active',
                protection_tier=excluded.protection_tier,
                mention_count=memory_records.mention_count + 1,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        connection.execute(
            "INSERT INTO memory_references(memory_id,referenced_at,reason) VALUES (?,?,?)",
            (payload[0], now, "stored"),
        )
        # Index entities for co-occurrence retrieval boost (Phase 1B)
        from .entity_index import index_entities
        index_entities(connection, payload[0], payload[1], now=now)

    def replace_prefixed_facts(
        self,
        prefixes: tuple[str, ...],
        records: Iterable[dict],
    ) -> int:
        """Atomically swap anchored rows: delete rows whose text starts with
        any prefix, then insert the replacement records inside one transaction
        so a crash can never leave the anchor set empty or half-written."""
        prepared = [self._prepare_record(**record) for record in records]
        inserted = 0
        with self.transaction() as connection:
            removed_ids: list[str] = []
            for prefix in prefixes:
                rows = connection.execute(
                    "SELECT id FROM memory_records WHERE substr(text, 1, ?) = ?",
                    (len(prefix), prefix),
                ).fetchall()
                removed_ids.extend(str(row["id"]) for row in rows)
            for memory_id in removed_ids:
                connection.execute(
                    "DELETE FROM memory_candidates WHERE committed_memory_id=?",
                    (memory_id,),
                )
                connection.execute(
                    "UPDATE memory_candidates SET conflict_memory_id='' WHERE conflict_memory_id=?",
                    (memory_id,),
                )
                connection.execute("DELETE FROM memory_records WHERE id=?", (memory_id,))
            for payload, now in prepared:
                self._insert_record(connection, payload, now)
                inserted += 1
        return inserted

    def get(self, memory_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_records WHERE id=?", (memory_id,),
            ).fetchone()
        return self._row(row) if row else None

    def count(self, layer: str | None = None, *, include_expired: bool = True) -> int:
        with self._lock:
            state_sql = "" if include_expired else " AND lifecycle_state='active'"
            if layer:
                row = self._connection.execute(
                    f"SELECT COUNT(*) FROM memory_records WHERE retention_layer=?{state_sql}", (layer,),
                ).fetchone()
            else:
                where = "" if include_expired else " WHERE lifecycle_state='active'"
                row = self._connection.execute(f"SELECT COUNT(*) FROM memory_records{where}").fetchone()
        return int(row[0]) if row else 0

    def metadata_get(self, key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM memory_system_metadata WHERE key=?", (str(key),),
            ).fetchone()
        return str(row[0]) if row else None

    def metadata_set(self, key: str, value: str) -> None:
        metadata_key = str(key).strip()
        if not metadata_key or len(metadata_key) > 500:
            raise ValueError("Metadata key is empty or too long")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO memory_system_metadata(key,value,updated_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                       value=excluded.value, updated_at=excluded.updated_at""",
                (metadata_key, str(value)[:4096], time.time()),
            )

    def list_by_layer(
        self,
        layer: str,
        limit: int | None = None,
        *,
        include_expired: bool = False,
    ) -> list[dict]:
        state_sql = "" if include_expired else " AND lifecycle_state='active'"
        sql = f"SELECT * FROM memory_records WHERE retention_layer=?{state_sql} ORDER BY timestamp ASC"
        params: list[object] = [layer]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row(row) for row in rows]

    def all(self) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_records ORDER BY timestamp ASC",
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_confirmed(self, *, limit: int = 100) -> list[dict]:
        """Return only current facts the user explicitly approved."""

        bounded_limit = max(1, min(200, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM memory_records
                   WHERE confirmation_state='confirmed'
                     AND lifecycle_state='active'
                   ORDER BY updated_at DESC,id DESC LIMIT ?""",
                (bounded_limit,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def page(
        self,
        *,
        page_size: int = 100,
        cursor: tuple[float, str] | None = None,
        layer: str | None = None,
        lifecycle_state: str = "all",
    ) -> tuple[list[dict], tuple[float, str] | None]:
        """Return a keyset page without imposing a catalog-wide row cap."""

        size = max(1, min(2_000, int(page_size)))
        if lifecycle_state not in {"active", "expired", "superseded", "all"}:
            raise ValueError("Unsupported memory lifecycle state")
        clauses: list[str] = []
        params: list[object] = []
        if layer is not None:
            if layer not in RETENTION_LAYERS:
                raise ValueError("Unsupported retention layer")
            clauses.append("retention_layer=?")
            params.append(layer)
        if lifecycle_state != "all":
            clauses.append("lifecycle_state=?")
            params.append(lifecycle_state)
        if cursor is not None:
            timestamp, memory_id = float(cursor[0]), str(cursor[1])
            if not math.isfinite(timestamp):
                raise ValueError("Invalid memory page cursor")
            clauses.append("(timestamp < ? OR (timestamp = ? AND id < ?))")
            params.extend((timestamp, timestamp, memory_id))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(size + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memory_records{where} "
                "ORDER BY timestamp DESC,id DESC LIMIT ?",
                params,
            ).fetchall()
        has_more = len(rows) > size
        selected = rows[:size]
        next_cursor = None
        if has_more and selected:
            last = selected[-1]
            next_cursor = (float(last["timestamp"]), str(last["id"]))
        return [self._row(row) for row in selected], next_cursor

    def expire_before(self, cutoff_timestamp: float) -> int:
        """Apply the time lifecycle without deleting or capping canonical rows."""

        cutoff = float(cutoff_timestamp)
        if not math.isfinite(cutoff):
            raise ValueError("Invalid retention cutoff")
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE memory_records SET lifecycle_state='expired',updated_at=?
                   WHERE lifecycle_state='active' AND retention_layer!='permanent'
                     AND COALESCE(pinned,0)=0 AND COALESCE(protection_tier,0)=0
                     AND timestamp<?""",
                (time.time(), cutoff),
            )
        return int(cursor.rowcount)

    def reactivate(self, memory_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_records SET lifecycle_state='active',updated_at=? WHERE id=?",
                (time.time(), str(memory_id)),
            )
        return cursor.rowcount > 0

    def candidate_rows(
        self,
        query: str,
        limit: int = 240,
        *,
        oldest_first: bool = False,
    ) -> list[dict]:
        terms = memory_terms(query)[:12]
        rows: list[sqlite3.Row] = []
        if terms:
            where = " OR ".join("keywords LIKE ? ESCAPE '\\'" for _ in terms)
            params: list[object] = [
                self._escape_like_term(term) for term in terms
            ]
            params.append(max(1, int(limit)))
            order = "timestamp ASC, importance DESC" if oldest_first else "importance DESC, timestamp DESC"
            with self._lock:
                rows = self._connection.execute(
                    f"""SELECT * FROM memory_records
                         WHERE ({where}) AND trust_level != 'untrusted_web'
                           AND lifecycle_state='active'
                         ORDER BY {order} LIMIT ?""",
                    params,
                ).fetchall()
        with self._lock:
            anchors = self._connection.execute(
                """SELECT * FROM memory_records WHERE trust_level != 'untrusted_web'
                   AND lifecycle_state='active'
                   ORDER BY importance DESC, timestamp DESC LIMIT ?""",
                (min(80, max(12, limit // 3)),),
            ).fetchall()
        merged = {str(row["id"]): row for row in [*rows, *anchors]}
        return [self._row(row) for row in list(merged.values())[:limit]]

    def permanent_candidates(self, query: str, limit: int = 240) -> list[dict]:
        """Return a bounded mix of cue matches and high-salience identity facts."""
        bounded_limit = max(1, min(2_000, int(limit)))
        terms = memory_terms(query)[:12]
        matches: list[sqlite3.Row] = []
        if terms:
            where = " OR ".join("keywords LIKE ? ESCAPE '\\'" for _ in terms)
            params: list[object] = [
                self._escape_like_term(term) for term in terms
            ]
            params.append(bounded_limit)
            with self._lock:
                matches = self._connection.execute(
                    f"""SELECT * FROM memory_records
                         WHERE retention_layer='permanent' AND lifecycle_state='active' AND ({where})
                         ORDER BY importance DESC, updated_at DESC LIMIT ?""",
                    params,
                ).fetchall()
        with self._lock:
            anchors = self._connection.execute(
                """SELECT * FROM memory_records WHERE retention_layer='permanent'
                   AND lifecycle_state='active'
                   ORDER BY importance DESC, updated_at DESC LIMIT ?""",
                (min(80, bounded_limit),),
            ).fetchall()
        merged = {str(row["id"]): row for row in [*matches, *anchors]}
        return [self._row(row) for row in list(merged.values())[:bounded_limit]]

    def event_rows_between(
        self,
        start_timestamp: float,
        end_timestamp: float,
        *,
        limit: int = 200,
    ) -> list[dict]:
        """Fetch one bounded local-time interval without loading the whole catalog."""
        start = float(start_timestamp)
        end = float(end_timestamp)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Invalid event-memory time interval")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM memory_records
                   WHERE cognitive_layer='episodic' AND lifecycle_state='active'
                     AND COALESCE(event_time,timestamp)>=?
                     AND COALESCE(event_time,timestamp)<?
                   ORDER BY importance DESC, COALESCE(event_time,timestamp) DESC
                   LIMIT ?""",
                (start, end, max(1, min(2_000, int(limit)))),
            ).fetchall()
        return [self._row(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        """Purge a memory and its confirming evidence instead of leaving a shadow copy."""

        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM memory_candidates WHERE committed_memory_id=?",
                (memory_id,),
            )
            connection.execute(
                "UPDATE memory_candidates SET conflict_memory_id='' WHERE conflict_memory_id=?",
                (memory_id,),
            )
            cursor = connection.execute(
                "DELETE FROM memory_records WHERE id=?",
                (memory_id,),
            )
        return cursor.rowcount > 0

    def delete_fact_lineage(self, fact_key: str) -> list[str]:
        """Atomically purge every revision and evidence row for one confirmed fact."""

        key = str(fact_key).strip()
        if not key:
            raise ValueError("Memory fact key is empty")
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id FROM memory_records WHERE fact_key=?",
                (key,),
            ).fetchall()
            memory_ids = [str(row["id"]) for row in rows]
            connection.execute(
                "DELETE FROM memory_candidates WHERE fact_key=?",
                (key,),
            )
            connection.execute(
                "DELETE FROM memory_records WHERE fact_key=?",
                (key,),
            )
        return memory_ids

    def delete_by_layer(self, layer: str) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM memory_records WHERE retention_layer=?", (layer,))

    def update_importance(self, memory_id: str, importance: float, *, timestamp: float | None = None) -> bool:
        values: list[object] = [max(0.0, min(1.0, float(importance))), time.time()]
        sql = "UPDATE memory_records SET importance=?, updated_at=?"
        if timestamp is not None:
            sql += ", timestamp=?"
            values.append(float(timestamp))
        sql += " WHERE id=?"
        values.append(memory_id)
        with self.transaction() as connection:
            cursor = connection.execute(sql, values)
        return cursor.rowcount > 0

    def touch_access(self, memory_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(str(value) for value in memory_ids if value))
        if not ids:
            return
        now = time.time()
        with self.transaction() as connection:
            connection.executemany(
                "UPDATE memory_records SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                [(now, memory_id) for memory_id in ids],
            )
            connection.executemany(
                "INSERT INTO memory_references(memory_id,referenced_at,reason) VALUES (?,?,?)",
                [(memory_id, now, "retrieved") for memory_id in ids],
            )

    # ── Lifecycle governance (P0 batch): usage signals, pins, budget ──

    def record_usage_event(
        self,
        *,
        memory_id: str,
        signal: str,
        weight: float,
        turn_id: str = "",
        created_at: float | None = None,
    ) -> bool:
        """Append one lifecycle usage signal. Unknown memory ids are dropped
        silently (tombstone tolerance): signals must never resurrect rows."""
        if signal not in USAGE_SIGNALS:
            raise ValueError(f"Unsupported usage signal: {signal}")
        if not memory_id:
            return False
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            return False
        if weight_value != weight_value or weight_value in (float("inf"), float("-inf")):
            return False
        with self.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM memory_records WHERE id=?", (memory_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                "INSERT INTO memory_usage_events(memory_id,signal,weight,turn_id,created_at)"
                " VALUES (?,?,?,?,?)",
                (
                    memory_id,
                    signal,
                    max(-10.0, min(10.0, weight_value)),
                    str(turn_id or "")[:200],
                    float(created_at if created_at is not None else time.time()),
                ),
            )
        return True

    def usage_scores_since(self, since_timestamp: float) -> dict[str, float]:
        """Aggregate weighted usage per memory since the given timestamp."""
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT memory_id, SUM(weight) AS total FROM memory_usage_events"
                " WHERE created_at>=? GROUP BY memory_id",
                (float(since_timestamp),),
            ).fetchall()
        return {str(row["memory_id"]): float(row["total"] or 0.0) for row in rows}

    def prune_usage_events(self, *, before_timestamp: float) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_usage_events WHERE created_at<?",
                (float(before_timestamp),),
            )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def set_pinned(self, memory_id: str, pinned: bool) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_records SET pinned=?, updated_at=? WHERE id=?",
                (1 if pinned else 0, time.time(), memory_id),
            )
        return cursor.rowcount > 0

    def count_pinned(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS total FROM memory_records WHERE pinned=1"
        ).fetchone()
        return int(row["total"] if row else 0)

    def layer_char_usage(self) -> dict[str, int]:
        """Sum of stored text lengths per retention layer (budget pressure input)."""
        rows = self._connection.execute(
            "SELECT retention_layer AS layer, SUM(LENGTH(text)) AS chars"
            " FROM memory_records WHERE lifecycle_state='active'"
            " GROUP BY retention_layer"
        ).fetchall()
        return {str(row["layer"]): int(row["chars"] or 0) for row in rows}

    # ── Evidence chain (Phase 1C, 2026-09-09) ──

    def add_evidence_ref(
        self,
        *,
        memory_id: str,
        relation: str,
        target_id: str,
        confidence: float = 1.0,
    ) -> int:
        """Link two memories via an evidence relationship."""
        valid_relations = {"source", "derived_from", "supports", "conflicts", "supersedes"}
        if relation not in valid_relations:
            raise ValueError(f"Unsupported evidence relation: {relation}")
        now = time.time()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO memory_evidence_refs
                       (memory_id, relation, target_id, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(memory_id), relation, str(target_id),
                 max(0.0, min(1.0, float(confidence))), now),
            )
        return int(cursor.lastrowid)

    def get_evidence_refs(self, memory_id: str) -> list[dict]:
        """Return all evidence references originating from a memory."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_evidence_refs WHERE memory_id=? ORDER BY created_at",
                (str(memory_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_referencing(self, target_id: str) -> list[dict]:
        """Return all evidence references pointing to a target memory."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_evidence_refs WHERE target_id=? ORDER BY created_at",
                (str(target_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_conflicts(self, memory_id: str) -> list[dict]:
        """Return evidence refs where this memory is in a 'conflicts' relation."""
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM memory_evidence_refs
                   WHERE relation='conflicts'
                     AND (memory_id=? OR target_id=?)
                   ORDER BY created_at""",
                (str(memory_id), str(memory_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Typed atoms (Phase 1C, 2026-09-09) ──

    def add_atom(
        self,
        *,
        memory_id: str,
        atom_type: str,
        subject: str = "",
        predicate: str = "",
        object: str = "",
        confidence: float = 0.8,
        extraction_reason: str = "",
    ) -> int:
        """Store a typed atomic fact derived from a memory."""
        valid_types = {"preference", "constraint", "promise", "event", "lesson", "temporal_fact"}
        if atom_type not in valid_types:
            raise ValueError(f"Unsupported atom type: {atom_type}")
        now = time.time()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO memory_atoms
                       (memory_id, atom_type, subject, predicate, object,
                        confidence, extraction_reason, review_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    str(memory_id),
                    atom_type,
                    str(subject)[:500],
                    str(predicate)[:500],
                    str(object)[:2000],
                    max(0.0, min(1.0, float(confidence))),
                    str(extraction_reason)[:500],
                    now,
                ),
            )
        return int(cursor.lastrowid)

    def list_atoms(
        self,
        memory_id: str | None = None,
        *,
        atom_type: str | None = None,
        review_status: str = "all",
        limit: int = 200,
    ) -> list[dict]:
        """Query typed atoms with optional filters."""
        clauses: list[str] = []
        params: list[object] = []
        if memory_id is not None:
            clauses.append("memory_id=?")
            params.append(str(memory_id))
        if atom_type is not None:
            clauses.append("atom_type=?")
            params.append(str(atom_type))
        if review_status != "all":
            clauses.append("review_status=?")
            params.append(str(review_status))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(2000, int(limit))))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memory_atoms{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_atom_status(self, atom_id: int, status: str) -> bool:
        """Approve or reject a typed atom."""
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError(f"Invalid atom review status: {status}")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_atoms SET review_status=? WHERE id=?",
                (status, int(atom_id)),
            )
        return cursor.rowcount > 0

    def record_recall_confusion(
        self,
        *,
        memory_id: str,
        query: str,
        true_text: str,
        rendered_text: str,
        source_entity: str,
        substitute_entity: str,
        domain: str,
    ) -> int:
        """Audit an ephemeral retrieval distortion without changing canonical memory."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO memory_recall_events(
                       memory_id,query,true_text,rendered_text,source_entity,
                       substitute_entity,domain,created_at
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    query[:1000],
                    true_text,
                    rendered_text,
                    source_entity,
                    substitute_entity,
                    domain,
                    time.time(),
                ),
            )
        return int(cursor.lastrowid)

    def latest_open_recall_confusion(self, max_age_seconds: float = 1800.0) -> dict | None:
        cutoff = time.time() - max(0.0, float(max_age_seconds))
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM memory_recall_events
                   WHERE corrected_at IS NULL AND created_at>=?
                   ORDER BY created_at DESC, id DESC LIMIT 1""",
                (cutoff,),
            ).fetchone()
        return dict(row) if row else None

    def resolve_recall_confusion(self, event_id: int) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE memory_recall_events SET corrected_at=?
                   WHERE id=? AND corrected_at IS NULL""",
                (time.time(), int(event_id)),
            )
        return cursor.rowcount > 0

    def pending_for_version(self, model_version: str, limit: int = 32) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT m.* FROM memory_records AS m
                   LEFT JOIN memory_embeddings AS e
                     ON e.memory_id=m.id AND e.model_version=? AND e.status='ready'
                    WHERE e.memory_id IS NULL AND m.lifecycle_state='active'
                   ORDER BY m.importance DESC, m.timestamp DESC LIMIT ?""",
                (model_version, max(1, int(limit))),
            ).fetchall()
        return [self._row(row) for row in rows]

    def embedding_counts(self, model_version: str) -> tuple[int, int]:
        total = self.count(include_expired=False)
        with self._lock:
            row = self._connection.execute(
                """SELECT COUNT(*) FROM memory_embeddings AS e
                   JOIN memory_records AS m ON m.id=e.memory_id
                   WHERE e.model_version=? AND e.status='ready'
                     AND m.lifecycle_state='active'""",
                (model_version,),
            ).fetchone()
        ready = int(row[0]) if row else 0
        return ready, max(0, total - ready)

    def invalidate_nonmatching_embeddings(
        self,
        model_version: str,
        expected_vector_table: str,
    ) -> int:
        """Schedule migration when a ready row belongs to another vector backend."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE memory_embeddings SET status='pending', error='backend_migration'
                   WHERE model_version=? AND status='ready' AND vector_table!=?""",
                (model_version, expected_vector_table),
            )
            connection.execute(
                """UPDATE memory_records SET embedding_status='pending'
                   WHERE id IN (
                       SELECT memory_id FROM memory_embeddings
                       WHERE model_version=? AND status='pending'
                   )""",
                (model_version,),
            )
        return cursor.rowcount

    def mark_embedding(
        self,
        memory_id: str,
        model_version: str,
        vector_table: str,
        dimensions: int,
        *,
        error: str = "",
    ) -> None:
        status = "error" if error else "ready"
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO memory_embeddings
                       (memory_id,model_version,vector_table,dimensions,status,embedded_at,error)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(memory_id,model_version) DO UPDATE SET
                       vector_table=excluded.vector_table, dimensions=excluded.dimensions,
                       status=excluded.status, embedded_at=excluded.embedded_at, error=excluded.error""",
                (memory_id, model_version, vector_table, dimensions, status, now, error[:500]),
            )
            connection.execute(
                """UPDATE memory_records SET embedding_model_version=?,
                   embedding_status=?, updated_at=? WHERE id=?""",
                (model_version, status, now, memory_id),
            )

    def _replace_all_materialized(self, records: list[dict]) -> int:
        """Replace the canonical memory set in one durable SQLite transaction."""
        prepared: list[tuple] = []
        now = time.time()
        for record in records:
            raw_memory_id = str(record.get("id", ""))
            memory_id = raw_memory_id.strip()
            text = str(record.get("text", "")).strip()
            retention = str(record.get("retention_layer", record.get("layer", "short_term")))
            cognitive = str(record.get("cognitive_layer", "episodic" if text.startswith("事件记忆：") else "semantic"))
            source_type = str(record.get("source_type", "backup_local"))
            trust_level = str(record.get("trust_level", "trusted_local"))
            if (
                not memory_id
                or memory_id != raw_memory_id
                or len(memory_id) > 500
                or not text
                or retention not in RETENTION_LAYERS
                or cognitive not in COGNITIVE_LAYERS
            ):
                raise ValueError("Backup contains an invalid memory record")
            if source_type == "untrusted_web" or trust_level == "untrusted_web":
                raise ValueError("Backup attempts to place untrusted web data in memory")
            if len(source_type) > 200 or len(trust_level) > 200:
                raise ValueError("Backup contains oversized provenance metadata")
            emotion_names = (
                "joy", "sadness", "anger", "excitement",
                "calm", "anxiety", "grievance", "touched",
            )
            source_emotions = record.get("emotions")
            if not isinstance(source_emotions, dict):
                source_emotions = {
                    name: record.get(f"emotion_{name}", 0.0) for name in emotion_names
                }
            emotions: dict[str, float] = {}
            for name in emotion_names:
                value = float(source_emotions.get(name, 0.0) or 0.0)
                if not math.isfinite(value):
                    raise ValueError("Backup contains a non-finite emotion value")
                emotions[name] = max(0.0, min(100.0, value))
            timestamp = float(record.get("timestamp", now) or now)
            importance = float(record.get("importance", 0.5) or 0.5)
            event_time = record.get("event_time")
            event_time = None if event_time is None else float(event_time)
            if not math.isfinite(timestamp) or not math.isfinite(importance):
                raise ValueError("Backup contains a non-finite memory score or timestamp")
            if event_time is not None and not math.isfinite(event_time):
                raise ValueError("Backup contains a non-finite event timestamp")
            model_version = str(record.get("embedding_model_version", "backup:pending")).strip()
            if not model_version or len(model_version) > 500:
                raise ValueError("Backup contains an invalid embedding model version")
            raw_flags = record.get("sanitizer_flags", [])
            if not isinstance(raw_flags, list):
                raise ValueError("Backup sanitizer flags must be a list")
            flags = [str(flag)[:200] for flag in raw_flags[:64]]
            lifecycle_state = str(record.get("lifecycle_state", "active"))
            if lifecycle_state not in {"active", "expired", "superseded"}:
                raise ValueError("Backup contains an invalid memory lifecycle state")
            fact_key = str(record.get("fact_key", ""))[:500]
            fact_revision = max(0, int(record.get("fact_revision", 0) or 0))
            supersedes_id = str(record.get("supersedes_id", ""))[:500]
            confirmation_state = str(record.get("confirmation_state", "observed"))
            if confirmation_state not in {"observed", "confirmed"}:
                raise ValueError("Backup contains an invalid confirmation state")
            raw_confirmed_at = record.get("confirmed_at")
            confirmed_at = None if raw_confirmed_at is None else float(raw_confirmed_at)
            if confirmed_at is not None and not math.isfinite(confirmed_at):
                raise ValueError("Backup contains a non-finite confirmation timestamp")
            protection_tier = 1 if int(record.get("protection_tier", 0) or 0) else 0
            pinned = 1 if int(record.get("pinned", 0) or 0) else 0
            prepared.append((
                memory_id, text, retention, cognitive, timestamp, event_time,
                max(0.0, min(1.0, importance)),
                json.dumps(emotions, ensure_ascii=False, sort_keys=True),
                source_type, str(record.get("source_uri", ""))[:4096],
                str(record.get("source_hash", ""))[:500], trust_level,
                str(record.get("sanitizer_status", "not_required"))[:200],
                json.dumps(flags, ensure_ascii=False),
                " ".join(memory_terms(text)),
                model_version,
                "pending", lifecycle_state, fact_key, fact_revision,
                supersedes_id, confirmation_state, confirmed_at, now, now,
                protection_tier, pinned,
            ))
        with self.transaction() as connection:
            connection.execute("DELETE FROM memory_records")
            connection.executemany(
                """INSERT INTO memory_records (
                       id,text,retention_layer,cognitive_layer,timestamp,event_time,importance,
                       emotions_json,source_type,source_uri,source_hash,trust_level,
                       sanitizer_status,sanitizer_flags_json,keywords,embedding_model_version,
                       embedding_status,lifecycle_state,fact_key,fact_revision,
                       supersedes_id,confirmation_state,confirmed_at,created_at,updated_at,
                       protection_tier,pinned
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                prepared,
            )
        return len(prepared)

    def replace_all(self, records: list[dict]) -> int:
        """Compatibility wrapper for callers that already materialized rows."""

        return self.import_records(iter(records), replace=True)

    def import_records(
        self,
        records: Iterable[dict],
        *,
        replace: bool,
        before_commit: Callable[[], None] | None = None,
        restore_marker: str = "",
    ) -> int:
        """Consume an arbitrary record stream inside one atomic transaction."""

        now = time.time()
        count = 0
        sql = """INSERT INTO memory_records (
                   id,text,retention_layer,cognitive_layer,timestamp,event_time,importance,
                   emotions_json,source_type,source_uri,source_hash,trust_level,
                   sanitizer_status,sanitizer_flags_json,keywords,embedding_model_version,
                   embedding_status,lifecycle_state,fact_key,fact_revision,
                   supersedes_id,confirmation_state,confirmed_at,created_at,updated_at,
                   protection_tier,pinned
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   text=excluded.text,retention_layer=excluded.retention_layer,
                   cognitive_layer=excluded.cognitive_layer,timestamp=excluded.timestamp,
                   event_time=excluded.event_time,importance=excluded.importance,
                   emotions_json=excluded.emotions_json,source_type=excluded.source_type,
                   source_uri=excluded.source_uri,source_hash=excluded.source_hash,
                   trust_level=excluded.trust_level,
                   sanitizer_status=excluded.sanitizer_status,
                   sanitizer_flags_json=excluded.sanitizer_flags_json,
                   keywords=excluded.keywords,
                   embedding_model_version=excluded.embedding_model_version,
                   embedding_status='pending',lifecycle_state=excluded.lifecycle_state,
                   fact_key=excluded.fact_key,fact_revision=excluded.fact_revision,
                    supersedes_id=excluded.supersedes_id,
                    confirmation_state=excluded.confirmation_state,
                    confirmed_at=excluded.confirmed_at,
                    created_at=excluded.created_at,updated_at=excluded.updated_at,
                    protection_tier=excluded.protection_tier,pinned=excluded.pinned"""
        with self.transaction() as connection:
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS reverie_import_seen_ids(id TEXT PRIMARY KEY)"
            )
            connection.execute("DELETE FROM reverie_import_seen_ids")
            if replace:
                connection.execute("DELETE FROM memory_records")
            for record in records:
                prepared = self._prepare_import_record(record, now)
                try:
                    connection.execute(
                        "INSERT INTO reverie_import_seen_ids(id) VALUES (?)",
                        (prepared[0],),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Backup contains duplicate id: {prepared[0]}") from exc
                connection.execute(sql, prepared)
                count += 1
            if restore_marker:
                connection.execute(
                    """INSERT INTO memory_system_metadata(key,value,updated_at) VALUES (?,?,?)
                       ON CONFLICT(key) DO UPDATE SET
                           value=excluded.value,updated_at=excluded.updated_at""",
                    ("full_restore_marker", str(restore_marker), time.time()),
                )
            if before_commit is not None:
                before_commit()
            connection.execute("DELETE FROM reverie_import_seen_ids")
        return count

    @staticmethod
    def _prepare_import_record(record: dict, now: float) -> tuple:
        if not isinstance(record, dict):
            raise ValueError("Backup contains a non-object memory record")
        raw_memory_id = str(record.get("id", ""))
        memory_id = raw_memory_id.strip()
        text = str(record.get("text", "")).strip()
        retention = str(record.get("retention_layer", record.get("layer", "short_term")))
        cognitive = str(record.get("cognitive_layer", "semantic"))
        source_type = str(record.get("source_type", "backup_local"))
        trust_level = str(record.get("trust_level", "trusted_local"))
        if (
            not memory_id
            or memory_id != raw_memory_id
            or len(memory_id) > 500
            or not text
            or retention not in RETENTION_LAYERS
            or cognitive not in COGNITIVE_LAYERS
        ):
            raise ValueError("Backup contains an invalid memory record")
        if source_type == "untrusted_web" or trust_level == "untrusted_web":
            raise ValueError("Backup attempts to place untrusted web data in memory")
        if len(source_type) > 200 or len(trust_level) > 200:
            raise ValueError("Backup contains oversized provenance metadata")
        emotion_names = (
            "joy", "sadness", "anger", "excitement",
            "calm", "anxiety", "grievance", "touched",
        )
        source_emotions = record.get("emotions")
        if not isinstance(source_emotions, dict):
            source_emotions = {
                name: record.get(f"emotion_{name}", 0.0) for name in emotion_names
            }
        emotions: dict[str, float] = {}
        for name in emotion_names:
            value = float(source_emotions.get(name, 0.0) or 0.0)
            if not math.isfinite(value):
                raise ValueError("Backup contains a non-finite emotion value")
            emotions[name] = max(0.0, min(100.0, value))
        timestamp = float(record.get("timestamp", now) or now)
        importance = float(record.get("importance", 0.5) or 0.5)
        event_time = record.get("event_time")
        event_time = None if event_time is None else float(event_time)
        if not math.isfinite(timestamp) or not math.isfinite(importance):
            raise ValueError("Backup contains a non-finite memory score or timestamp")
        if event_time is not None and not math.isfinite(event_time):
            raise ValueError("Backup contains a non-finite event timestamp")
        model_version = str(record.get("embedding_model_version", "backup:pending")).strip()
        if not model_version or len(model_version) > 500:
            raise ValueError("Backup contains an invalid embedding model version")
        raw_flags = record.get("sanitizer_flags", [])
        if not isinstance(raw_flags, list):
            raise ValueError("Backup sanitizer flags must be a list")
        flags = [str(flag)[:200] for flag in raw_flags[:64]]
        lifecycle_state = str(record.get("lifecycle_state", "active"))
        if lifecycle_state not in {"active", "expired", "superseded"}:
            raise ValueError("Backup contains an invalid memory lifecycle state")
        fact_key = str(record.get("fact_key", ""))
        supersedes_id = str(record.get("supersedes_id", ""))
        if len(fact_key) > 500 or len(supersedes_id) > 500:
            raise ValueError("Backup contains oversized fact version metadata")
        fact_revision = int(record.get("fact_revision", 0) or 0)
        if fact_revision < 0:
            raise ValueError("Backup contains an invalid fact revision")
        confirmation_state = str(record.get("confirmation_state", "observed"))
        if confirmation_state not in {"observed", "confirmed"}:
            raise ValueError("Backup contains an invalid confirmation state")
        raw_confirmed_at = record.get("confirmed_at")
        confirmed_at = None if raw_confirmed_at is None else float(raw_confirmed_at)
        if confirmed_at is not None and not math.isfinite(confirmed_at):
            raise ValueError("Backup contains a non-finite confirmation timestamp")
        protection_tier = 1 if int(record.get("protection_tier", 0) or 0) else 0
        pinned = 1 if int(record.get("pinned", 0) or 0) else 0
        return (
            memory_id, text, retention, cognitive, timestamp, event_time,
            max(0.0, min(1.0, importance)),
            json.dumps(emotions, ensure_ascii=False, sort_keys=True),
            source_type, str(record.get("source_uri", ""))[:4096],
            str(record.get("source_hash", ""))[:500], trust_level,
            str(record.get("sanitizer_status", "not_required"))[:200],
            json.dumps(flags, ensure_ascii=False),
            " ".join(memory_terms(text)), model_version,
            "pending", lifecycle_state, fact_key, fact_revision,
            supersedes_id, confirmation_state, confirmed_at, now, now,
            protection_tier, pinned,
        )

    def reset_embedding_derivatives(self) -> None:
        """Mark every canonical row pending after an incremental import."""

        with self.transaction() as connection:
            connection.execute("DELETE FROM memory_embeddings")
            connection.execute(
                "UPDATE memory_records SET embedding_status='pending',updated_at=?",
                (time.time(),),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

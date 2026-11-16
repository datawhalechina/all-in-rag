"""Persistent job queue for offline memory maintenance tasks.

Inspired by ALTM's lease-based worker (Apache-2.0) and Letta's sleep-time
compute pattern.  All state lives in the catalog SQLite — no external queue.

Job kinds:
  - reembed:         Re-embed memories after model switch
  - extract_atoms:   Extract typed atoms from raw memories
  - cleanup_usage:   Aggregate and prune old usage events
  - evidence_check:  Scan for contradictions in recent memories
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from typing import Any

logger = logging.getLogger("reverie.memory.fold_worker")

JOB_KINDS = frozenset({"reembed", "extract_atoms", "cleanup_usage", "evidence_check"})
JOB_STATUSES = frozenset({"pending", "running", "completed", "failed"})

# ── Schema (called from catalog._create_schema) ──────────────────────

JOB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_until REAL,
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    dedupe_key TEXT UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK (status IN ('pending','running','completed','failed'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_kind
    ON memory_jobs(status, kind, created_at);
"""


def create_job_tables(connection: sqlite3.Connection) -> None:
    """Idempotent job table creation."""
    connection.executescript(JOB_SCHEMA_SQL)


# ── Job queue operations ─────────────────────────────────────────────


class FoldWorker:
    """Persistent job queue consumer for offline memory maintenance."""

    def __init__(self, connection: sqlite3.Connection, lock) -> None:
        self._conn = connection
        self._lock = lock
        self._owner = f"worker-{uuid.uuid4().hex[:8]}"

    def enqueue(
        self,
        kind: str,
        payload: dict | None = None,
        *,
        dedupe_key: str | None = None,
        max_attempts: int = 5,
    ) -> str | None:
        """Idempotent job enqueue.  Returns job_id or None if dedupe hit."""
        if kind not in JOB_KINDS:
            raise ValueError(f"Unsupported job kind: {kind}")
        job_id = f"job_{uuid.uuid4().hex[:24]}"
        now = time.time()
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """INSERT INTO memory_jobs
                           (id, kind, payload_json, max_attempts, dedupe_key,
                            status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (job_id, kind, payload_json,
                     max(1, min(20, int(max_attempts))),
                     dedupe_key, now, now),
                )
                self._conn.execute("COMMIT")
                return job_id
            except sqlite3.IntegrityError:
                self._conn.execute("ROLLBACK")
                return None  # dedupe_key collision = idempotent skip

    def claim_job(self, kind: str | None = None, lease_seconds: float = 300.0) -> dict | None:
        """Claim the oldest pending job, atomically setting it to 'running'."""
        now = time.time()
        lease_until = now + max(30.0, min(3600.0, float(lease_seconds)))
        kind_clause = "AND kind=?" if kind else ""
        params: list[Any] = [now]
        if kind:
            params.append(kind)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Also reclaim expired leases
                row = self._conn.execute(
                    f"""SELECT * FROM memory_jobs
                        WHERE (status='pending' OR (status='running' AND lease_until<?)) 
                          {kind_clause}
                          AND attempts < max_attempts
                        ORDER BY created_at ASC LIMIT 1""",
                    params,
                ).fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    return None
                job_id = str(row["id"])
                self._conn.execute(
                    """UPDATE memory_jobs
                       SET status='running', lease_owner=?, lease_until=?,
                           attempts=attempts+1, updated_at=?
                       WHERE id=?""",
                    (self._owner, lease_until, now, job_id),
                )
                self._conn.execute("COMMIT")
                result = dict(row)
                result["status"] = "running"
                result["lease_owner"] = self._owner
                result["lease_until"] = lease_until
                result["attempts"] = int(row["attempts"]) + 1
                try:
                    result["payload"] = json.loads(result.pop("payload_json", "{}"))
                except (TypeError, json.JSONDecodeError):
                    result["payload"] = {}
                try:
                    result["checkpoint"] = json.loads(result.pop("checkpoint_json", "{}"))
                except (TypeError, json.JSONDecodeError):
                    result["checkpoint"] = {}
                return result
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def complete_job(self, job_id: str, checkpoint: dict | None = None) -> bool:
        """Mark a job as completed, optionally saving a final checkpoint."""
        now = time.time()
        cp = json.dumps(checkpoint or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    """UPDATE memory_jobs
                       SET status='completed', checkpoint_json=?, updated_at=?
                       WHERE id=? AND status='running'""",
                    (cp, now, str(job_id)),
                )
                self._conn.execute("COMMIT")
                return cursor.rowcount > 0
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def fail_job(self, job_id: str, checkpoint: dict | None = None) -> bool:
        """Mark a running job as failed.  If attempts < max, it can be reclaimed."""
        now = time.time()
        cp = json.dumps(checkpoint or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Check if max_attempts reached
                row = self._conn.execute(
                    "SELECT attempts, max_attempts FROM memory_jobs WHERE id=?",
                    (str(job_id),),
                ).fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    return False
                new_status = "failed" if int(row["attempts"]) >= int(row["max_attempts"]) else "pending"
                cursor = self._conn.execute(
                    """UPDATE memory_jobs
                       SET status=?, checkpoint_json=?, lease_owner='', lease_until=NULL, updated_at=?
                       WHERE id=? AND status='running'""",
                    (new_status, cp, now, str(job_id)),
                )
                self._conn.execute("COMMIT")
                return cursor.rowcount > 0
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def list_jobs(
        self,
        *,
        status: str = "all",
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List jobs with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if status != "all":
            if status not in JOB_STATUSES:
                raise ValueError(f"Unsupported job status: {status}")
            clauses.append("status=?")
            params.append(status)
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(1000, int(limit))))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory_jobs{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Catalog reference (set by MemoryCatalog after construction) ────

    _catalog: Any = None

    def set_catalog(self, catalog: Any) -> None:
        """Set a back-reference to the owning MemoryCatalog."""
        self._catalog = catalog

    # ── Job handlers ────────────────────────────────────────────────────

    # Atom extraction regex patterns (mirroring candidates.py for typed atoms)
    _PREFERENCE = re.compile(
        r"(?:用户?|我)(?:喜欢|讨厌|害怕|不喜欢|最爱|偏好)"
        r"[：:是]?\s*([^，,。.!！？；;\r\n]{1,160})",
    )
    _CONSTRAINT = re.compile(
        r"(?:我不能|我没办法|我做不到|过敏|不吃|不喝|不碰)"
        r"[：:是]?\s*([^，,。.!！？；;\r\n]{1,160})",
    )
    _PROMISE = re.compile(
        r"(?:我[会要](?:去)?|约好了?|答应了?|承诺|说好了?|约定)"
        r"[：:是]?\s*([^，,。.!！？；;\r\n]{1,160})",
    )
    _EVENT = re.compile(
        r"(?:生日|婚礼|毕业|考试|面试|旅行|搬家|手术|比赛|聚会|约会|出差|开会|纪念日|假期)"
        r"[：:是]?\s*([^，,。.!！？；;\r\n]{0,160})",
    )
    _LESSON = re.compile(
        r"(?:我?学到了?|教训是?|以后不[要会再]|原来|领悟到?)"
        r"[：:是]?\s*([^，,。.!！？；;\r\n]{1,160})",
    )
    _TEMPORAL = re.compile(
        r"(?:[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日?|(?:上个?|下个?|这个?)(?:月|周|星期|学期|暑假))"
        r"\s*([^，,。.!！？；;\r\n]{0,160})",
    )

    _ATOM_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("preference", _PREFERENCE),
        ("constraint", _CONSTRAINT),
        ("promise", _PROMISE),
        ("event", _EVENT),
        ("lesson", _LESSON),
        ("temporal_fact", _TEMPORAL),
    ]

    def _handle_extract_atoms(self, job: dict) -> None:
        """Extract typed atoms from a memory record using regex (no LLM)."""
        if self._catalog is None:
            raise RuntimeError("fold_worker has no catalog reference")
        memory_id = job.get("payload", {}).get("memory_id", "")
        if not memory_id:
            return
        row = self._catalog.get(memory_id)
        if row is None:
            return
        text = str(row.get("text", ""))
        if not text:
            return

        for atom_type, pattern in self._ATOM_PATTERNS:
            match = pattern.search(text)
            if match:
                obj = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if obj and len(obj) >= 2:
                    self._catalog.add_atom(
                        memory_id=memory_id,
                        atom_type=atom_type,
                        subject="user",
                        predicate=atom_type,
                        object=obj[:200],
                        confidence=0.7,
                        extraction_reason="regex_fold",
                    )

    def _handle_evidence_check(self, job: dict) -> None:
        """Check for contradictions with same-fact_key memories."""
        if self._catalog is None:
            raise RuntimeError("fold_worker has no catalog reference")
        memory_id = job.get("payload", {}).get("memory_id", "")
        if not memory_id:
            return
        row = self._catalog.get(memory_id)
        if row is None:
            return
        fact_key = row.get("fact_key", "")
        if not fact_key:
            return
        # Find other active memories with the same fact_key
        with self._lock:
            others = self._conn.execute(
                """SELECT id FROM memory_records
                   WHERE fact_key=? AND id!=? AND lifecycle_state='active'
                   LIMIT 10""",
                (fact_key, memory_id),
            ).fetchall()
        for other_row in others:
            other_id = str(other_row["id"])
            self._catalog.add_evidence_ref(
                memory_id=memory_id,
                relation="conflicts",
                target_id=other_id,
                confidence=0.5,
            )

    def _handle_cleanup_usage(self, job: dict) -> None:
        """Remove usage events older than 90 days."""
        cutoff = time.time() - (90 * 86400)
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_usage_events WHERE timestamp < ?",
                (cutoff,),
            )
            self._conn.execute("COMMIT")

    def _handle_reembed(self, job: dict) -> None:
        """Re-embed a memory. Delegates to the versioned store if available."""
        memory_id = job.get("payload", {}).get("memory_id", "")
        if not memory_id or self._catalog is None:
            return
        row = self._catalog.get(memory_id)
        if row is None:
            return
        text = str(row.get("text", ""))
        if text:
            self._catalog._update_embedding_status(memory_id, "pending")

    # ── Batch processor ────────────────────────────────────────────────

    def process_batch(self, limit: int = 5) -> int:
        """Claim and process up to `limit` jobs.  Returns count processed."""
        processed = 0
        for _ in range(max(1, min(20, limit))):
            job = self.claim_job()
            if job is None:
                break
            kind = job.get("kind", "")
            try:
                if kind == "extract_atoms":
                    self._handle_extract_atoms(job)
                elif kind == "evidence_check":
                    self._handle_evidence_check(job)
                elif kind == "cleanup_usage":
                    self._handle_cleanup_usage(job)
                elif kind == "reembed":
                    self._handle_reembed(job)
                else:
                    logger.warning("Unknown job kind: %s", kind)
                    self.fail_job(job["id"])
                    continue
                self.complete_job(job["id"])
                processed += 1
            except Exception:
                logger.exception("fold_worker job %s (%s) failed", job.get('id'), kind)
                self.fail_job(job["id"])
        return processed

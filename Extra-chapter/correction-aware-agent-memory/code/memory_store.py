"""基于 SQLite 的最小纠错记忆仓库。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Callable

from memory_policy import MemoryPolicy, lexical_score


class MemoryStore:
    def __init__(
        self,
        database: str | Path,
        *,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.database = str(database)
        self.policy = policy or MemoryPolicy()
        self.clock = clock or time.time_ns
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                owner_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                value TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                expires_at INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                supersedes_id INTEGER REFERENCES memories(id),
                deleted_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_memory_scope
              ON memories(owner_id, namespace, status, memory_key);
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                occurred_at INTEGER NOT NULL,
                detail TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _event(self, memory_id: int, action: str, now: int, detail: str) -> None:
        self.connection.execute(
            "INSERT INTO memory_events(memory_id, action, occurred_at, detail) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, action, now, detail),
        )

    def write(
        self,
        *,
        owner_id: str,
        namespace: str,
        memory_key: str,
        value: str,
        kind: str,
        source_type: str,
        source_ref: str,
        expires_at: int | None = None,
    ) -> int:
        fields = (owner_id, namespace, memory_key, value)
        if any(not field.strip() for field in fields):
            raise ValueError("owner_id、namespace、memory_key、value 不能为空")
        self.policy.validate_write(
            kind=kind, source_type=source_type, source_ref=source_ref
        )
        now = self.clock()
        with self.connection:
            duplicate = self.connection.execute(
                """
                SELECT id FROM memories
                WHERE owner_id=? AND namespace=? AND memory_key=? AND value=?
                  AND kind=? AND status='active'
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (owner_id, namespace, memory_key, value, kind, now),
            ).fetchone()
            if duplicate:
                memory_id = int(duplicate["id"])
                self._event(memory_id, "deduplicate", now, source_ref)
                return memory_id

            superseded = None
            if kind == "correction":
                previous = self.connection.execute(
                    """
                    SELECT id FROM memories
                    WHERE owner_id=? AND namespace=? AND memory_key=?
                      AND status='active'
                      AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC, id DESC LIMIT 1
                    """,
                    (owner_id, namespace, memory_key, now),
                ).fetchone()
                if previous:
                    superseded = int(previous["id"])
                conflicts = self.connection.execute(
                    """
                    SELECT id FROM memories
                    WHERE owner_id=? AND namespace=? AND memory_key=?
                      AND status='active'
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (owner_id, namespace, memory_key, now),
                ).fetchall()
                for conflict in conflicts:
                    conflict_id = int(conflict["id"])
                    self.connection.execute(
                        "UPDATE memories SET status='superseded', updated_at=? "
                        "WHERE id=?",
                        (now, conflict_id),
                    )
                    self._event(conflict_id, "supersede", now, "被纠正记录替代")

            cursor = self.connection.execute(
                """
                INSERT INTO memories(
                    owner_id, namespace, memory_key, value, kind,
                    source_type, source_ref, created_at, updated_at,
                    expires_at, supersedes_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    namespace,
                    memory_key,
                    value,
                    kind,
                    source_type,
                    source_ref,
                    now,
                    now,
                    expires_at,
                    superseded,
                ),
            )
            memory_id = int(cursor.lastrowid)
            self._event(memory_id, "write", now, source_ref)
            return memory_id

    def update(self, memory_id: int, *, value: str, source_ref: str) -> int:
        """以新版本替代旧记录；旧版本保留用于审计。"""
        if not value.strip():
            raise ValueError("value 不能为空")
        old = self.connection.execute(
            "SELECT * FROM memories WHERE id=? AND status='active'", (memory_id,)
        ).fetchone()
        if old is None:
            raise KeyError(f"不存在活动记忆: {memory_id}")
        now = self.clock()
        self.policy.validate_write(
            kind=old["kind"],
            source_type=old["source_type"],
            source_ref=source_ref,
        )
        with self.connection:
            self.connection.execute(
                "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                (now, memory_id),
            )
            cursor = self.connection.execute(
                """
                INSERT INTO memories(
                    owner_id, namespace, memory_key, value, kind,
                    source_type, source_ref, created_at, updated_at,
                    expires_at, supersedes_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    old["owner_id"],
                    old["namespace"],
                    old["memory_key"],
                    value,
                    old["kind"],
                    old["source_type"],
                    source_ref,
                    now,
                    now,
                    old["expires_at"],
                    memory_id,
                ),
            )
            new_id = int(cursor.lastrowid)
            self._event(memory_id, "supersede", now, f"被 {new_id} 替代")
            self._event(new_id, "update", now, source_ref)
            return new_id

    def recall(
        self,
        *,
        owner_id: str,
        namespace: str,
        query: str,
        memory_key: str | None = None,
        limit: int = 5,
        include_external: bool = False,
        now: int | None = None,
    ) -> list[dict]:
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        current = self.clock() if now is None else now
        sql = """
            SELECT * FROM memories
            WHERE owner_id=? AND namespace=? AND status='active'
              AND (expires_at IS NULL OR expires_at > ?)
        """
        params: list[object] = [owner_id, namespace, current]
        if memory_key is not None:
            sql += " AND memory_key=?"
            params.append(memory_key)
        if not include_external:
            sql += " AND source_type!='external'"
        rows = [dict(row) for row in self.connection.execute(sql, params)]
        if memory_key is None:
            rows = [
                row
                for row in rows
                if lexical_score(query, f"{row['memory_key']} {row['value']}") > 0
            ]
        rows.sort(key=lambda row: self.policy.rank(row, query), reverse=True)
        return rows[:limit]

    def delete(self, *, memory_id: int, owner_id: str) -> bool:
        """软删除并留下审计事件；owner 条件防止越权删除。"""
        now = self.clock()
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE memories
                SET status='deleted', deleted_at=?, updated_at=?
                WHERE id=? AND owner_id=? AND status='active'
                """,
                (now, now, memory_id, owner_id),
            )
            if cursor.rowcount:
                self._event(memory_id, "delete", now, "显式删除")
                return True
        return False

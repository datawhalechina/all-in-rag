"""Rebuildable sqlite-vec index stored beside canonical memory records."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import threading
import time

import numpy as np

from src.storage.encrypted_sqlite import connect_database

logger = logging.getLogger("reverie.memory.sqlite_vec")

_SAFE_TABLE = re.compile(r"^memory_vec_[a-f0-9]{20}$")


class SQLiteVecIndex:
    """Exact local KNN cache. SQLite memory records remain authoritative."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        model_version: str,
        dimensions: int,
        quantization: str = "int8",
        partitioning: bool = True,
    ) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise RuntimeError("sqlite-vec is not installed") from exc

        self._sqlite_vec = sqlite_vec
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = connect_database(
            self.path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
        except BaseException:
            self._connection.close()
            raise
        finally:
            try:
                self._connection.enable_load_extension(False)
            except Exception:
                pass
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._create_control_schema()
        self.model_version = ""
        self.dimensions = 0
        self.table_name = ""
        self.quantization = self._validate_quantization(quantization)
        self.partitioning = bool(partitioning)
        self.activate(
            model_version,
            dimensions,
            quantization=self.quantization,
            partitioning=self.partitioning,
        )

    def _create_control_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_vector_indexes (
                model_version TEXT PRIMARY KEY,
                table_name TEXT NOT NULL UNIQUE,
                dimensions INTEGER NOT NULL,
                quantization TEXT NOT NULL DEFAULT 'float32',
                partitioning INTEGER NOT NULL DEFAULT 0,
                extension_version TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_vector_rows (
                vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                UNIQUE(memory_id, model_version),
                FOREIGN KEY(memory_id) REFERENCES memory_records(id) ON DELETE CASCADE,
                FOREIGN KEY(model_version) REFERENCES memory_vector_indexes(model_version)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_vector_rows_model
                ON memory_vector_rows(model_version, memory_id);
            """
        )
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(memory_vector_indexes)")
        }
        if "quantization" not in columns:
            self._connection.execute(
                "ALTER TABLE memory_vector_indexes ADD COLUMN quantization TEXT NOT NULL DEFAULT 'float32'"
            )
        if "partitioning" not in columns:
            self._connection.execute(
                "ALTER TABLE memory_vector_indexes ADD COLUMN partitioning INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _table_name(
        model_version: str,
        dimensions: int,
        quantization: str,
        partitioning: bool,
    ) -> str:
        identity = (
            f"{model_version}:{dimensions}:{quantization}:{int(partitioning)}"
        ).encode("utf-8", errors="strict")
        return f"memory_vec_{hashlib.sha256(identity).hexdigest()[:20]}"

    @staticmethod
    def _validate_quantization(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"float32", "int8"}:
            raise ValueError("Vector quantization must be float32 or int8")
        return normalized

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        if not _SAFE_TABLE.fullmatch(table_name):
            raise ValueError("Unsafe sqlite-vec table identifier")
        return table_name

    def activate(
        self,
        model_version: str,
        dimensions: int,
        *,
        quantization: str | None = None,
        partitioning: bool | None = None,
    ) -> None:
        version = str(model_version).strip()
        size = int(dimensions)
        selected_quantization = self._validate_quantization(
            quantization if quantization is not None else self.quantization
        )
        selected_partitioning = (
            self.partitioning if partitioning is None else bool(partitioning)
        )
        if not version or len(version) > 500:
            raise ValueError("Invalid embedding model version")
        if size < 1 or size > 65_536:
            raise ValueError("Invalid embedding dimension")
        table_name = self._validate_table_name(
            self._table_name(version, size, selected_quantization, selected_partitioning)
        )
        extension_version = str(self._connection.execute("SELECT vec_version()").fetchone()[0])
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    """SELECT table_name,dimensions,quantization,partitioning
                       FROM memory_vector_indexes
                       WHERE model_version=?""",
                    (version,),
                ).fetchone()
                if existing and str(existing["table_name"]) != table_name:
                    old_table = self._validate_table_name(str(existing["table_name"]))
                    old_rows = self._connection.execute(
                        "SELECT vector_rowid FROM memory_vector_rows WHERE model_version=?",
                        (version,),
                    ).fetchall()
                    if self._table_exists(old_table):
                        for row in old_rows:
                            self._connection.execute(
                                f"DELETE FROM {old_table} WHERE rowid=?",
                                (int(row["vector_rowid"]),),
                            )
                    self._connection.execute(
                        "DELETE FROM memory_vector_rows WHERE model_version=?",
                        (version,),
                    )
                    if self._table_exists(old_table):
                        self._connection.execute(f"DROP TABLE {old_table}")
                self._connection.execute(
                    """INSERT INTO memory_vector_indexes(
                           model_version,table_name,dimensions,quantization,partitioning,
                           extension_version,created_at
                       ) VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(model_version) DO UPDATE SET
                           table_name=excluded.table_name,
                           dimensions=excluded.dimensions,
                           quantization=excluded.quantization,
                           partitioning=excluded.partitioning,
                           extension_version=excluded.extension_version""",
                    (
                        version,
                        table_name,
                        size,
                        selected_quantization,
                        int(selected_partitioning),
                        extension_version,
                        time.time(),
                    ),
                )
                vector_type = "int8" if selected_quantization == "int8" else "float"
                bucket_definition = (
                    "event_bucket text partition key"
                    if selected_partitioning
                    else "event_bucket text"
                )
                self._connection.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING vec0(
                           embedding {vector_type}[{size}] distance_metric=cosine,
                           retention_layer text,
                           cognitive_layer text,
                           {bucket_definition},
                           event_time integer
                       )"""
                )
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")
        self.model_version = version
        self.dimensions = size
        self.table_name = table_name
        self.quantization = selected_quantization
        self.partitioning = selected_partitioning

    @property
    def extension_version(self) -> str:
        with self._lock:
            return str(self._connection.execute("SELECT vec_version()").fetchone()[0])

    @staticmethod
    def _as_vector(vector: np.ndarray | list[float], dimensions: int) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if len(array) != dimensions or not np.isfinite(array).all():
            raise ValueError("Embedding vector has invalid dimensions or non-finite values")
        norm = float(np.linalg.norm(array))
        if norm > 0.0:
            array = array / norm
        return array

    @staticmethod
    def _event_bucket(row: dict) -> str:
        if str(row.get("layer", row.get("retention_layer", ""))) == "permanent":
            return "core"
        raw_time = row.get("event_time") or row.get("timestamp") or 0.0
        try:
            timestamp = float(raw_time)
            if timestamp <= 0:
                return "unknown"
            moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return "unknown"
        return f"{moment.year:04d}-Q{((moment.month - 1) // 3) + 1}"

    def _serialize_vector(self, array: np.ndarray) -> bytes:
        return self._sqlite_vec.serialize_float32(array.tolist())

    @property
    def _match_expression(self) -> str:
        return "vec_quantize_int8(?, 'unit')" if self.quantization == "int8" else "?"

    def _table_exists(self, table_name: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?",
            (self._validate_table_name(table_name),),
        ).fetchone() is not None

    def upsert(self, row: dict, vector: np.ndarray | list[float]) -> None:
        memory_id = str(row.get("id", "")).strip()
        if not memory_id:
            raise ValueError("Memory id is empty")
        array = self._as_vector(vector, self.dimensions)
        payload = self._serialize_vector(array)
        event_time = int(float(row.get("event_time") or row.get("timestamp") or 0.0))
        event_bucket = self._event_bucket(row)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                stale_rows = self._connection.execute(
                    """SELECT mapping.vector_rowid,indexes.table_name
                       FROM memory_vector_rows AS mapping
                       JOIN memory_vector_indexes AS indexes
                         ON indexes.model_version=mapping.model_version
                       WHERE mapping.memory_id=? AND mapping.model_version<>?""",
                    (memory_id, self.model_version),
                ).fetchall()
                for stale in stale_rows:
                    stale_table = self._validate_table_name(str(stale["table_name"]))
                    if self._table_exists(stale_table):
                        self._connection.execute(
                            f"DELETE FROM {stale_table} WHERE rowid=?",
                            (int(stale["vector_rowid"]),),
                        )
                self._connection.execute(
                    """DELETE FROM memory_vector_rows
                       WHERE memory_id=? AND model_version<>?""",
                    (memory_id, self.model_version),
                )
                mapping = self._connection.execute(
                    """SELECT vector_rowid FROM memory_vector_rows
                       WHERE memory_id=? AND model_version=?""",
                    (memory_id, self.model_version),
                ).fetchone()
                if mapping:
                    vector_rowid = int(mapping[0])
                else:
                    cursor = self._connection.execute(
                        """INSERT INTO memory_vector_rows(memory_id,model_version)
                           VALUES (?,?)""",
                        (memory_id, self.model_version),
                    )
                    vector_rowid = int(cursor.lastrowid)
                self._connection.execute(
                    f"DELETE FROM {self.table_name} WHERE rowid=?", (vector_rowid,),
                )
                self._connection.execute(
                    f"""INSERT INTO {self.table_name}(
                           rowid,embedding,retention_layer,cognitive_layer,event_bucket,event_time
                       ) VALUES (?,{self._match_expression},?,?,?,?)""",
                    (
                        vector_rowid,
                        payload,
                        str(row.get("layer", row.get("retention_layer", "short_term"))),
                        str(row.get("cognitive_layer", "episodic")),
                        event_bucket,
                        event_time,
                    ),
                )
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def search(
        self,
        vector: np.ndarray | list[float],
        k: int,
        *,
        layers: list[str] | None = None,
        event_buckets: list[str] | None = None,
        event_time_min: int | None = None,
        event_time_max: int | None = None,
    ) -> list[dict]:
        array = self._as_vector(vector, self.dimensions)
        payload = self._serialize_vector(array)
        limit = max(1, min(1000, int(k)))
        allowed_layers = list(dict.fromkeys(
            str(layer).strip() for layer in (layers or []) if str(layer).strip()
        ))
        filter_parts: list[str] = []
        filter_parameters: list[object] = []
        if allowed_layers:
            placeholders = ",".join("?" for _ in allowed_layers)
            filter_parts.append(f"retention_layer IN ({placeholders})")
            filter_parameters.extend(allowed_layers)
        if event_time_min is not None:
            filter_parts.append("event_time >= ?")
            filter_parameters.append(int(event_time_min))
        if event_time_max is not None:
            filter_parts.append("event_time <= ?")
            filter_parameters.append(int(event_time_max))
        buckets = list(dict.fromkeys(
            str(bucket).strip() for bucket in (event_buckets or []) if str(bucket).strip()
        ))[:16]
        # sqlite-vec partition keys are deliberately queried one partition at a time.
        # This keeps the query planner on the fast partition path and also works on
        # extension versions that do not accept IN(...) for partition constraints.
        partitions: list[str | None] = buckets if self.partitioning and buckets else [None]
        collected: dict[str, dict] = {}
        with self._lock:
            for bucket in partitions:
                local_parts = list(filter_parts)
                local_parameters = list(filter_parameters)
                if bucket is not None:
                    local_parts.append("event_bucket = ?")
                    local_parameters.append(bucket)
                metadata_filter = "".join(f" AND {part}" for part in local_parts)
                parameters: list[object] = [payload, limit, *local_parameters, self.model_version]
                rows = self._connection.execute(
                    f"""WITH nearest AS (
                           SELECT rowid,distance FROM {self.table_name}
                           WHERE embedding MATCH {self._match_expression} AND k = ?
                           {metadata_filter}
                       )
                       SELECT mapping.memory_id, nearest.distance
                       FROM nearest
                       JOIN memory_vector_rows AS mapping
                         ON mapping.vector_rowid=nearest.rowid
                       WHERE mapping.model_version=?
                       ORDER BY nearest.distance ASC""",
                    parameters,
                ).fetchall()
                for row in rows:
                    memory_id = str(row["memory_id"])
                    distance = float(row["distance"])
                    previous = collected.get(memory_id)
                    if previous is None or distance < float(previous["_distance"]):
                        collected[memory_id] = {"id": memory_id, "_distance": distance}
        return sorted(collected.values(), key=lambda row: float(row["_distance"]))[:limit]

    def delete(self, memory_id: str) -> None:
        with self._lock:
            rows = self._connection.execute(
                """SELECT mapping.vector_rowid,indexes.table_name
                   FROM memory_vector_rows AS mapping
                   JOIN memory_vector_indexes AS indexes
                     ON indexes.model_version=mapping.model_version
                   WHERE mapping.memory_id=?""",
                (memory_id,),
            ).fetchall()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    table_name = self._validate_table_name(str(row["table_name"]))
                    self._connection.execute(
                        f"DELETE FROM {table_name} WHERE rowid=?", (int(row["vector_rowid"]),),
                    )
                self._connection.execute(
                    "DELETE FROM memory_vector_rows WHERE memory_id=?", (memory_id,),
                )
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def reset_all(self) -> None:
        """Clear derivative vectors after canonical backup replacement."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT table_name FROM memory_vector_indexes",
            ).fetchall()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    table_name = self._validate_table_name(str(row["table_name"]))
                    self._connection.execute(f"DELETE FROM {table_name}")
                self._connection.execute("DELETE FROM memory_vector_rows")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

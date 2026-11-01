"""Version-aware sqlite-vec cache backed by the transactional memory catalog."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .catalog import MemoryCatalog
from .embedding import DEFAULT_MODEL, embed_query, embed_texts, embedding_runtime_info
from .sqlite_vec_index import SQLiteVecIndex

logger = logging.getLogger("reverie.memory.versioned_store")


class VersionedVectorStore:
    """Treat vectors as a rebuildable cache, never as the memory itself."""

    def __init__(
        self,
        db_path: str,
        table_name: str = "memories",
        *,
        model_name: str = DEFAULT_MODEL,
        catalog_path: str | Path | None = None,
        vector_quantization: str = "int8",
        vector_partitioning: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.legacy_table_name = table_name
        self.model_name = model_name
        self.vector_quantization = vector_quantization
        self.vector_partitioning = bool(vector_partitioning)
        self.catalog = MemoryCatalog(catalog_path or self.db_path.parent / "memory_catalog.db")
        self._db = None
        self.runtime = embedding_runtime_info(self.model_name)
        self.table_name = self._table_name(self.runtime.model_version)
        self._migrate_legacy_lance_rows()
        self._sqlite_index: SQLiteVecIndex | None = None
        self.vector_backend = "bounded_lexical_fallback"
        self._activate_sqlite_index()

    def _activate_sqlite_index(self) -> None:
        if self.runtime.backend == "unavailable":
            if self._sqlite_index is not None:
                self._sqlite_index.close()
            self._sqlite_index = None
            self.vector_backend = "bounded_lexical_fallback"
            return
        try:
            if self._sqlite_index is None:
                self._sqlite_index = SQLiteVecIndex(
                    self.catalog.path,
                    model_version=self.runtime.model_version,
                    dimensions=self.runtime.dimensions,
                    quantization=self.vector_quantization,
                    partitioning=self.vector_partitioning,
                )
            else:
                self._sqlite_index.activate(
                    self.runtime.model_version,
                    self.runtime.dimensions,
                    quantization=self.vector_quantization,
                    partitioning=self.vector_partitioning,
                )
            self.table_name = self._sqlite_index.table_name
            partition_label = "quarter-partition" if self.vector_partitioning else "unpartitioned"
            self.vector_backend = (
                f"sqlite-vec/{self._sqlite_index.extension_version}/"
                f"{self.vector_quantization}/{partition_label}"
            )
            marker = self._vector_table_marker()
            migrated = self.catalog.invalidate_nonmatching_embeddings(
                self.runtime.model_version,
                marker,
            )
            if migrated:
                logger.info("Scheduled %d vectors for sqlite-vec migration", migrated)
        except Exception:
            self._sqlite_index = None
            self.vector_backend = "bounded_lexical_fallback"
            logger.exception("sqlite-vec unavailable; canonical lexical recall remains active")

    def _vector_table_marker(self) -> str:
        return f"sqlite_vec:{self.table_name}"

    @staticmethod
    def _table_name(model_version: str) -> str:
        digest = hashlib.sha256(model_version.encode("utf-8")).hexdigest()[:16]
        return f"memory_idx_{digest}"

    @property
    def db(self):
        if self._db is None:
            import lancedb

            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _table_names(self) -> list[str] | None:
        try:
            result = self.db.list_tables()
            return list(result.tables if hasattr(result, "tables") else result)
        except (ImportError, ModuleNotFoundError):
            # The optional LanceDB reader is absent from the production
            # runtime; there is nothing to migrate and no user action required.
            logger.debug("LanceDB legacy reader is unavailable; skipping migration")
            return None
        except Exception:
            logger.exception("Could not list LanceDB tables")
            return None

    def _legacy_migration_key(self) -> str:
        identity = f"{self.db_path.resolve()}\0{self.legacy_table_name}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"legacy_lance_migration:{digest}"

    def _migrate_legacy_lance_rows(self) -> None:
        migration_key = self._legacy_migration_key()
        if self.catalog.metadata_get(migration_key) == "complete":
            return
        if not any(self.db_path.iterdir()):
            return
        table_names = self._table_names()
        if table_names is None or self.legacy_table_name not in table_names:
            return
        try:
            rows = self.db.open_table(self.legacy_table_name).to_lance().to_pylist()
        except (ImportError, ModuleNotFoundError):
            # to_lance() needs the optional `pylance` package. The production
            # runtime never ships it, so a quiet skip avoids startup noise while
            # leaving canonical memory untouched.
            logger.debug("Legacy LanceDB migration skipped: pylance is unavailable")
            return
        except Exception:
            logger.exception("Legacy LanceDB memory scan failed")
            return
        migrated = 0
        skipped = 0
        try:
            for row in rows:
                text = str(row.get("text", "")).strip()
                memory_id = str(row.get("id", "")).strip()
                if not text or not memory_id or memory_id == "_init_":
                    continue
                # A crash after any successful row is harmless: on the next launch
                # that row is recognized as canonical and the remaining rows resume.
                if self.catalog.get(memory_id) is not None:
                    continue
                try:
                    emotions = {
                        name: float(row.get(f"emotion_{name}", 0.0) or 0.0)
                        for name in ("joy", "sadness", "anger", "excitement")
                    }
                    cognitive = "episodic" if text.startswith("事件记忆：") else "semantic"
                except (TypeError, ValueError, OverflowError):
                    skipped += 1
                    logger.debug("Skipped malformed legacy memory %s", memory_id, exc_info=True)
                    continue
                self.catalog.upsert(
                    id=memory_id,
                    text=text,
                    retention_layer=str(row.get("layer", "short_term")),
                    cognitive_layer=cognitive,
                    timestamp=float(row.get("timestamp", time.time()) or time.time()),
                    importance=float(row.get("importance", 0.5) or 0.5),
                    emotions=emotions,
                    embedding_model_version="legacy:unknown",
                    source_type="legacy_local",
                )
                migrated += 1
        except Exception:
            logger.exception("Legacy migration was interrupted and will resume next launch")
            return
        self.catalog.metadata_set(migration_key, "complete")
        if migrated:
            logger.info("Migrated %d canonical memories from the legacy LanceDB table", migrated)
        if skipped:
            logger.warning("Skipped %d malformed legacy memories during migration", skipped)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts, self.model_name)

    def embed_query(self, query: str) -> np.ndarray:
        return embed_query(query, self.model_name)

    def _index_add(self, memory_id: str, text: str, vector: np.ndarray) -> None:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if len(array) != self.runtime.dimensions:
            raise ValueError(
                f"Embedding dimension {len(array)} does not match {self.runtime.dimensions}"
            )
        if self._sqlite_index is None:
            raise RuntimeError("sqlite-vec index is unavailable")
        row = self.catalog.get(memory_id)
        if not row:
            raise KeyError(f"Canonical memory disappeared before indexing: {memory_id}")
        self._sqlite_index.upsert(row, array)

    def add(
        self,
        id: str,
        text: str,
        vector: np.ndarray | list[float] | None = None,
        *,
        layer: str = "short_term",
        cognitive_layer: str | None = None,
        importance: float = 0.5,
        emotions: dict[str, float] | None = None,
        timestamp: float | None = None,
        event_time: float | None = None,
        source_type: str = "local_interaction",
        source_uri: str = "",
        source_hash: str = "",
        trust_level: str = "trusted_local",
        sanitizer_status: str = "not_required",
        sanitizer_flags: list[str] | None = None,
    ) -> None:
        memory_time = float(timestamp or time.time())
        cognitive = cognitive_layer or ("episodic" if text.startswith("事件记忆：") else "semantic")
        self.catalog.upsert(
            id=id,
            text=text,
            retention_layer=layer,
            cognitive_layer=cognitive,
            timestamp=memory_time,
            event_time=event_time,
            importance=importance,
            emotions=emotions,
            embedding_model_version=self.runtime.model_version,
            source_type=source_type,
            source_uri=source_uri,
            source_hash=source_hash,
            trust_level=trust_level,
            sanitizer_status=sanitizer_status,
            sanitizer_flags=sanitizer_flags,
        )
        self._sync_vector_cache(id, text, vector)

    def _sync_vector_cache(self, memory_id: str, text: str, vector=None) -> None:
        """Best-effort mirror of a canonical record into the rebuildable
        vector cache. The canonical catalog row is already committed."""
        if self.runtime.backend == "unavailable":
            self.catalog.mark_embedding(
                memory_id,
                self.runtime.model_version,
                self._vector_table_marker(),
                self.runtime.dimensions,
                error="semantic embedding model is unavailable",
            )
            return
        try:
            actual_vector = vector if vector is not None else self.embed_documents([text])[0]
            self._index_add(memory_id, text, np.asarray(actual_vector, dtype=np.float32))
            self.catalog.mark_embedding(
                memory_id, self.runtime.model_version, self._vector_table_marker(), self.runtime.dimensions,
            )
        except Exception as exc:
            self.catalog.mark_embedding(
                memory_id, self.runtime.model_version, self._vector_table_marker(), self.runtime.dimensions,
                error=str(exc),
            )
            logger.exception("Memory %s is canonical but its vector index is pending", memory_id)

    def replace_permanent_profile_facts(
        self,
        prefixes: tuple[str, ...],
        facts: list[str],
        skip_texts: Iterable[str] = (),
    ) -> int:
        """Atomically refresh anchored permanent profile facts.

        The delete + reinsert runs in one canonical-store transaction so a
        crash mid-sync can never leave the permanent layer without its
        profile anchors. Vector cache updates happen best-effort afterwards.
        """
        skip = set(skip_texts)
        timestamp = time.time()
        records = [
            dict(
                id=f"perm_{uuid.uuid4().hex}",
                text=fact,
                retention_layer="permanent",
                cognitive_layer="semantic",
                timestamp=timestamp,
                importance=1.0,
                emotions=None,
                embedding_model_version=self.runtime.model_version,
                source_type="user_profile",
            )
            for fact in facts
            if fact not in skip
        ]
        inserted = self.catalog.replace_prefixed_facts(prefixes, records)
        for record in records:
            self._sync_vector_cache(record["id"], record["text"])
        return inserted

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        layers: list[str] | None = None,
        min_importance: float = 0.0,
        *,
        event_buckets: list[str] | None = None,
        event_time_min: int | None = None,
        event_time_max: int | None = None,
    ) -> list[dict]:
        try:
            if self._sqlite_index is None:
                return []
            indexed = self._sqlite_index.search(
                np.asarray(query_vector, dtype=np.float32),
                k=max(32, int(k) * 8),
                layers=layers,
                event_buckets=event_buckets,
                event_time_min=event_time_min,
                event_time_max=event_time_max,
            )
        except Exception:
            logger.exception("Current vector index search failed")
            return []
        results: list[dict] = []
        allowed_layers = set(layers or [])
        for index_row in indexed:
            row = self.catalog.get(str(index_row.get("id", "")))
            if not row:
                continue
            if str(row.get("lifecycle_state", "active")) != "active":
                continue
            if allowed_layers and row.get("layer") not in allowed_layers:
                continue
            if float(row.get("importance", 0.0) or 0.0) < min_importance:
                continue
            row["_distance"] = float(index_row.get("_distance", 1.0) or 0.0)
            results.append(row)
            if len(results) >= k:
                break
        return results

    def delete(self, id: str) -> bool:
        if self._sqlite_index is not None:
            try:
                self._sqlite_index.delete(id)
            except Exception:
                logger.exception("sqlite-vec deletion failed for %s", id)
        return self.catalog.delete(id)

    def delete_fact_lineage(self, fact_key: str) -> list[str]:
        memory_ids = self.catalog.delete_fact_lineage(fact_key)
        if self._sqlite_index is not None:
            for memory_id in memory_ids:
                try:
                    self._sqlite_index.delete(memory_id)
                except Exception:
                    logger.exception(
                        "sqlite-vec lineage deletion failed for %s",
                        memory_id,
                    )
        return memory_ids

    def delete_by_layer(self, layer: str) -> None:
        ids = [str(row["id"]) for row in self.catalog.list_by_layer(layer)]
        for memory_id in ids:
            if self._sqlite_index is not None:
                try:
                    self._sqlite_index.delete(memory_id)
                except Exception:
                    logger.debug("Layer sqlite-vec cleanup failed", exc_info=True)
        self.catalog.delete_by_layer(layer)

    def random_sample(self, layer: str, n: int = 1) -> list[dict]:
        rows = self.list_by_layer(layer)
        if not rows:
            return []
        import random
        return random.sample(rows, min(max(0, n), len(rows)))

    def list_by_layer(self, layer: str, limit: int | None = None) -> list[dict]:
        return self.catalog.list_by_layer(layer, limit)

    def count(self, layer: str | None = None) -> int:
        return self.catalog.count(layer)

    def candidate_rows(
        self,
        query: str,
        limit: int = 240,
        *,
        oldest_first: bool = False,
    ) -> list[dict]:
        return self.catalog.candidate_rows(query, limit, oldest_first=oldest_first)

    def switch_embedding_model(self, model_name: str) -> bool:
        previous = self.runtime.model_version
        self.model_name = (model_name or self.model_name).strip()
        self.runtime = embedding_runtime_info(self.model_name)
        self.table_name = self._table_name(self.runtime.model_version)
        self._activate_sqlite_index()
        return previous != self.runtime.model_version

    def configure_vector_index(self, *, quantization: str, partitioning: bool) -> bool:
        normalized = str(quantization).strip().lower()
        if normalized not in {"float32", "int8"}:
            raise ValueError("Unsupported vector quantization")
        changed = (
            normalized != self.vector_quantization
            or bool(partitioning) != self.vector_partitioning
        )
        self.vector_quantization = normalized
        self.vector_partitioning = bool(partitioning)
        if changed:
            self._activate_sqlite_index()
        return changed

    def migration_status(self) -> dict:
        indexed, pending = self.catalog.embedding_counts(self.runtime.model_version)
        semantic_available = self.runtime.backend != "unavailable"
        return {
            "requested_model": self.model_name,
            "model_version": self.runtime.model_version,
            "backend": self.runtime.backend,
            "vector_backend": self.vector_backend,
            "dimensions": self.runtime.dimensions,
            "quantization": self.vector_quantization,
            "partitioning": self.vector_partitioning,
            "partition_strategy": "utc_quarter" if self.vector_partitioning else "none",
            "indexed": indexed,
            "pending": pending,
            "semantic_available": semantic_available,
            "diagnostic": (
                ""
                if semantic_available
                else "语义模型未安装，当前仅使用有界词法检索。"
            ),
            "state": (
                "lexical_only"
                if not semantic_available
                else ("organizing" if pending else "ready")
            ),
        }

    def reembed_batch(self, limit: int = 24) -> dict:
        if self.runtime.backend == "unavailable":
            return {
                **self.migration_status(),
                "processed": 0,
                "failed": 0,
                "reason": "embedding_unavailable",
            }
        rows = self.catalog.pending_for_version(self.runtime.model_version, limit)
        if not rows:
            return {**self.migration_status(), "processed": 0, "failed": 0}
        vectors = self.embed_documents([str(row["text"]) for row in rows])
        processed = 0
        failed = 0
        for row, vector in zip(rows, vectors):
            memory_id = str(row["id"])
            try:
                self._index_add(memory_id, str(row["text"]), vector)
                self.catalog.mark_embedding(
                    memory_id, self.runtime.model_version, self._vector_table_marker(), self.runtime.dimensions,
                )
                processed += 1
            except Exception as exc:
                self.catalog.mark_embedding(
                    memory_id, self.runtime.model_version, self._vector_table_marker(), self.runtime.dimensions,
                    error=str(exc),
                )
                failed += 1
        return {**self.migration_status(), "processed": processed, "failed": failed}

    def close(self) -> None:
        if self._sqlite_index is not None:
            self._sqlite_index.close()
            self._sqlite_index = None
        self._db = None
        self.catalog.close()

    def reset_vector_indexes(self) -> None:
        if self._sqlite_index is not None:
            self._sqlite_index.reset_all()

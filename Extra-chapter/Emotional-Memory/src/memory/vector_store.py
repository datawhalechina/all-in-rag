"""Compatibility bridge for the retired LanceDB memory writer.

New and old imports both resolve to the transactional SQLite implementation.
The private legacy class is retained only so old on-disk data can be understood;
it cannot be instantiated or used as an independent memory source.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import numpy as np
from lancedb.pydantic import LanceModel, Vector

from .versioned_store import VersionedVectorStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger("reverie.memory.vector_store")


# ── Schema ─────────────────────────────────────────────────

# Dimension for bge-small-en-v1.5
VECTOR_DIM = 384


class MemoryEntry(LanceModel):
    """A single memory record in the vector store."""
    id: str
    text: str
    layer: str  # "permanent", "long_term", "short_term"
    timestamp: float  # Unix epoch
    importance: float = 0.5  # 0.0–1.0
    emotion_joy: float = 0.0
    emotion_sadness: float = 0.0
    emotion_anger: float = 0.0
    emotion_excitement: float = 0.0
    vector: Vector(VECTOR_DIM)


# ── Store ──────────────────────────────────────────────────

class _RetiredLanceVectorStore:
    """Non-instantiable description of the pre-SQLite storage layout."""

    VECTOR_DIM: int = 384  # bge-small-en-v1.5 embedding dimension

    def __init__(self, db_path: str, table_name: str = "memories") -> None:
        raise RuntimeError(
            "The independent LanceDB writer is retired; use VersionedVectorStore"
        )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self._db = None
        self._table = None

    @property
    def db(self):
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    @property
    def table(self):
        if self._table is None:
            try:
                existing = self.db.list_tables()
                # LanceDB 1.x: list_tables() returns ListTablesResponse with .tables attr
                existing_names = existing.tables if hasattr(existing, "tables") else existing
            except Exception:
                existing_names = []
            if self.table_name in existing_names:
                self._table = self.db.open_table(self.table_name)
            else:
                self._table = self._create_table()
        return self._table

    def _create_table(self):
        logger.info("Creating LanceDB table '%s' at %s", self.table_name, self.db_path)
        # Create with a dummy row to establish schema
        dummy = [{
            "id": "_init_",
            "text": "",
            "layer": "permanent",
            "timestamp": time.time(),
            "importance": 0.5,
            "emotion_joy": 0.0,
            "emotion_sadness": 0.0,
            "emotion_anger": 0.0,
            "emotion_excitement": 0.0,
            "vector": np.zeros(VECTOR_DIM, dtype=np.float32),
        }]
        tbl = self.db.create_table(self.table_name, data=dummy)
        # Remove the dummy row
        tbl.delete("id = '_init_'")
        return tbl

    def add(
        self,
        id: str,
        text: str,
        vector: np.ndarray,
        *,
        layer: str = "short_term",
        importance: float = 0.5,
        emotions: dict[str, float] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Insert a single memory into the store."""
        emo = emotions or {}
        self.table.add([{
            "id": id,
            "text": text,
            "layer": layer,
            "timestamp": timestamp or time.time(),
            "importance": importance,
            "emotion_joy": emo.get("joy", 0.0),
            "emotion_sadness": emo.get("sadness", 0.0),
            "emotion_anger": emo.get("anger", 0.0),
            "emotion_excitement": emo.get("excitement", 0.0),
            "vector": vector,
        }])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        layers: list[str] | None = None,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """Semantic search returning the top-k most similar memories."""
        if self.table.count_rows() == 0:
            return []

        q = self.table.search(query_vector).limit(k)

        if layers:
            # Filter by layer
            # LanceDB supports where clauses
            layer_conditions = " OR ".join(f"layer = '{l}'" for l in layers)
            q = q.where(layer_conditions, prefilter=True)

        if min_importance > 0:
            q = q.where(f"importance >= {min_importance}", prefilter=True)

        results = q.to_list()
        return results

    def delete(self, id: str) -> None:
        """Remove a memory by id."""
        self.table.delete(f"id = '{id}'")

    def delete_by_layer(self, layer: str) -> None:
        """Remove all memories of a given layer."""
        self.table.delete(f"layer = '{layer}'")

    def random_sample(self, layer: str, n: int = 1) -> list[dict]:
        """Return n random memories from a layer (for forgetting/misremembering)."""
        try:
            # LanceDB supports sampling via pyarrow
            return self.table.to_lance().sample(n, filter=f"layer = '{layer}'").to_pylist()
        except Exception:
            logger.debug("Random sample failed — table may be empty", exc_info=True)
            return []

    def list_by_layer(self, layer: str, limit: int | None = None) -> list[dict]:
        """Return stored memories from a layer without vector similarity ranking."""
        try:
            rows = self.table.to_arrow().to_pylist()
        except Exception:
            logger.exception("Failed to list memories for layer %s", layer)
            return []

        filtered = [row for row in rows if row.get("layer") == layer]
        filtered.sort(key=lambda row: row.get("timestamp", 0.0))
        if limit is not None:
            return filtered[:limit]
        return filtered

    def count(self, layer: str | None = None) -> int:
        """Return the number of rows, optionally filtered by layer."""
        if layer:
            return self.table.count_rows(f"layer = '{layer}'")
        return self.table.count_rows()

    def close(self) -> None:
        self._table = None
        self._db = None


VectorStore = VersionedVectorStore

__all__ = ["VECTOR_DIM", "MemoryEntry", "VectorStore", "VersionedVectorStore"]

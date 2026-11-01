"""Local embedding model loader and inference.

Semantic recall is available only when a real local model is present. Missing
models fail explicitly so callers can use the canonical lexical index without
pretending deterministic noise is semantic similarity.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("reverie.memory.embedding")

# Keep this aligned with settings.MemorySettings.embedding_model and the
# packaged seed config: the product ships Chinese conversations, so the zh
# model is the default, not the English one.
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_DIMENSIONS = 384
_ZH_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_EN_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_embedding_models: dict[str, object] = {}
_embedding_load_failures: set[str] = set()
_embedding_lock = threading.RLock()


@dataclass(frozen=True)
class EmbeddingRuntime:
    """Exact identity of the vector space active in this process."""

    requested_model: str
    model_version: str
    dimensions: int
    backend: str


class EmbeddingUnavailable(RuntimeError):
    """The requested semantic model is not installed in the local runtime."""


def get_embedding_model(model_name: str = DEFAULT_MODEL):
    """Load one cached model without downloading it implicitly."""
    requested = (model_name or DEFAULT_MODEL).strip()
    with _embedding_lock:
        if requested in _embedding_models:
            return _embedding_models[requested]
        if requested in _embedding_load_failures:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading cached local embedding model: %s", requested)
            model = SentenceTransformer(requested, local_files_only=True)
            _embedding_models[requested] = model
            return model
        except Exception as exc:
            _embedding_load_failures.add(requested)
            logger.warning(
                "Embedding model %s unavailable; semantic recall is disabled (%s)",
                requested, exc,
            )
            return None


def embedding_runtime_info(model_name: str = DEFAULT_MODEL) -> EmbeddingRuntime:
    """Return a stable version tag that detects lexical-to-model drift."""
    requested = (model_name or DEFAULT_MODEL).strip()
    model = get_embedding_model(requested)
    if model is None:
        return EmbeddingRuntime(
            requested_model=requested,
            model_version=f"unavailable:{requested}",
            dimensions=DEFAULT_DIMENSIONS,
            backend="unavailable",
        )
    getter = getattr(model, "get_embedding_dimension", None)
    if not callable(getter):
        getter = getattr(model, "get_sentence_embedding_dimension")
    dimensions = int(getter())
    try:
        import sentence_transformers
        library_version = str(getattr(sentence_transformers, "__version__", "unknown"))
    except Exception:
        library_version = "unknown"
    return EmbeddingRuntime(
        requested_model=requested,
        model_version=f"sentence-transformers:{requested}:{library_version}:{dimensions}",
        dimensions=dimensions,
        backend="sentence_transformers",
    )


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Convert a list of strings to embedding vectors.

    The shape is determined by the active runtime and may change across models.
    """
    if not texts:
        dimensions = embedding_runtime_info(model_name).dimensions
        return np.empty((0, dimensions), dtype=np.float32)

    model = get_embedding_model(model_name)
    if model is None:
        raise EmbeddingUnavailable(
            f"semantic embedding model is unavailable: {model_name}"
        )

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def _bge_query_instruction(model_name: str) -> str | None:
    """BGE query instruction for the model family, or None when unprefixed."""
    name = (model_name or "").lower()
    if "bge" not in name:
        return None
    if "zh" in name:
        return _ZH_BGE_QUERY_INSTRUCTION
    return _EN_BGE_QUERY_INSTRUCTION


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a search query with the appropriate BGE prefix."""
    model = get_embedding_model(model_name)
    if model is None:
        raise EmbeddingUnavailable(
            f"semantic embedding model is unavailable: {model_name}"
        )
    instruction = _bge_query_instruction(model_name)
    prepared = query if instruction is None or query.startswith(instruction) else (
        f"{instruction}{query}"
    )
    embedding = model.encode(
        [prepared],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embedding[0].astype(np.float32)

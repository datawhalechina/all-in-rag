# -*- coding: utf-8 -*-
# Copyright 2025-2026 Project N.E.K.O. Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -- 修改说明 (Modified by Muhe Studio 2026) --
# 从 Project N.E.K.O 的 memory/facts.py 与 memory/reflection.py 中提取
# 事实/反思条目的持久化 schema、内容哈希去重核心算法，改为独立导入工具：
# 读取 N.E.K.O 导出的 facts.json / reflections.json，写入 Reverie 的
# MemoryManager（store_manual_memory），并把旧 API（search(layer=, top_k=)）
# 适配为当前 MemoryManager.search(query, k=...)。

"""N.E.K.O 记忆导出 → Reverie 本地事实库导入工具。

数据来源（N.E.K.O, Apache 2.0）：
  - facts.json      列表，条目含 ``text/importance/entity/source/hash/created_at``
  - reflections.json 列表，条目含 ``text/entity/status/source_fact_ids/created_at``
  - persona.json    列表，条目含 ``text`` 等

用法：
    from src.memory import neko_import
    summary = neko_import.import_from_directory(
        directory=Path(".../memory/星野幻月"),
        memory=memory_manager,
        source_required=True,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("reverie.memory.neko_import")

FACTS_FILENAME = "facts.json"
REFLECTIONS_FILENAME = "reflections.json"
PERSONA_FILENAME = "persona.json"

_WHITESPACE = re.compile(r"\s+")

# 与 N.E.K.O facts.py 的 _fact_content_hash 保持一致的截断长度。
HASH_LENGTH = 16

# 导入文件与单条目的安全上限，防止大文件/长文本把模糊去重变成 O(n·m) 卡死。
_MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024
_MAX_ENTRY_TEXT_CHARS = 10_000


class NekoImportError(RuntimeError):
    """N.E.K.O 记忆导入失败。"""


def content_hash(text: str) -> str:
    """N.E.K.O 风格内容哈希（SHA-256 前 16 位）。"""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def normalize_entry(entry: Any) -> dict[str, Any] | None:
    """把 N.E.K.O fact/reflection/persona 条目归一化为可导入记录。"""
    if not isinstance(entry, dict):
        return None
    text = _normalize_text(entry.get("text") or entry.get("content") or "")
    if not text:
        return None
    # Bound the text so the fuzzy dedupe SequenceMatcher never runs against
    # an unbounded string (O(n*m) with the retrieved candidates).
    if len(text) > _MAX_ENTRY_TEXT_CHARS:
        text = text[:_MAX_ENTRY_TEXT_CHARS]
    importance = entry.get("importance")
    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = 5
    return {
        "text": text,
        "importance": max(1, min(10, importance)),
        "entity": str(entry.get("entity") or "relationship"),
        "source": str(entry.get("source") or "user_observation"),
        "created_at": str(entry.get("created_at") or ""),
        "source_id": str(entry.get("id") or ""),
        "status": str(entry.get("status") or ""),
        "hash": content_hash(text),
    }


def _entry_text(memory_text: str) -> str:
    return _normalize_text(memory_text)


def _is_duplicate(
    memory_texts: list[str],
    candidate: str,
    *,
    fuzzy_threshold: float = 0.95,
) -> bool:
    candidate_norm = _normalize_text(candidate)
    for memory_text in memory_texts:
        existing_norm = _entry_text(memory_text)
        if not existing_norm:
            continue
        if existing_norm == candidate_norm:
            return True
        if fuzzy_threshold > 0.0:
            ratio = SequenceMatcher(None, existing_norm, candidate_norm).ratio()
            if ratio >= fuzzy_threshold:
                return True
    return False


def search(memory: Any, query: str, *, layer: str = "long_term", top_k: int = 1) -> list[str]:
    """适配 N.E.K.O 旧 API（search(layer=..., top_k=...)）到当前 MemoryManager。

    当前 MemoryManager.search(query, k=...) 返回匹配记忆文本列表。
    """
    search_fn = getattr(memory, "search", None)
    if not callable(search_fn):
        return []
    try:
        return list(search_fn(query, k=max(1, int(top_k))))
    except TypeError:
        try:
            return list(search_fn(query))
        except Exception as exc:  # pragma: no cover - 防御未知实现
            logger.debug("N.E.K.O search 适配失败: %s", exc)
            return []


def import_entries(
    memory: Any,
    entries: list[dict[str, Any]],
    *,
    layer: str = "long_term",
    dedupe: bool = True,
    fuzzy_threshold: float = 0.95,
) -> dict[str, int]:
    """把归一化条目写入 Reverie 记忆库，返回 {imported, skipped}。"""
    if not memory:
        raise NekoImportError("memory manager is required for import")
    store = getattr(memory, "store_manual_memory", None)
    if not callable(store):
        raise NekoImportError("memory manager has no store_manual_memory")

    imported = 0
    skipped = 0
    seen_hashes: set[str] = set()
    for entry in entries:
        text = _normalize_text(entry.get("text") or "")
        if not text:
            skipped += 1
            continue
        if len(text) > _MAX_ENTRY_TEXT_CHARS:
            text = text[:_MAX_ENTRY_TEXT_CHARS]
        # Exact-hash dedupe first: O(1) per entry and cheap to run across the
        # whole batch, so the fuzzy SequenceMatcher below only sees genuinely
        # new candidates instead of every duplicate re-entering embedding
        # search + fuzzy comparison.
        entry_hash = entry.get("hash") or content_hash(text)
        if dedupe:
            if entry_hash in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(entry_hash)
            existing = search(memory, text, layer=layer, top_k=3)
            if _is_duplicate(existing, text, fuzzy_threshold=fuzzy_threshold):
                skipped += 1
                continue
        try:
            store(text, layer)
            imported += 1
        except Exception as exc:
            logger.warning("N.E.K.O 条目导入失败: %s", exc)
            skipped += 1
    return {"imported": imported, "skipped": skipped}


def _load_json_list(path: Path) -> list[Any]:
    try:
        if path.stat().st_size > _MAX_IMPORT_FILE_BYTES:
            raise NekoImportError(f"导入文件过大: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NekoImportError(f"无法读取 {path}: {exc}") from exc
    return data if isinstance(data, list) else []


def import_json_file(
    memory: Any,
    path: str | Path,
    *,
    layer: str = "long_term",
    dedupe: bool = True,
) -> dict[str, int]:
    """导入单个 N.E.K.O JSON（facts.json / reflections.json / persona.json）。"""
    source = Path(path)
    raw_entries = _load_json_list(source)
    entries = [normalized for raw in raw_entries if (normalized := normalize_entry(raw)) is not None]
    result = import_entries(
        memory,
        entries,
        layer=layer,
        dedupe=dedupe,
    )
    result["file"] = str(source)
    return result


def import_from_directory(
    memory: Any,
    directory: str | Path,
    *,
    layer: str = "long_term",
    dedupe: bool = True,
    files: tuple[str, ...] = (FACTS_FILENAME, REFLECTIONS_FILENAME, PERSONA_FILENAME),
) -> dict[str, Any]:
    """扫描 N.E.K.O 角色记忆目录，导入 facts/reflections/persona。

    N.E.K.O 的目录结构为 ``memory/{name}/facts.json`` 等。也支持直接把一个
    JSON 文件路径传入 ``directory``（兼容单文件导出）。
    """
    target = Path(directory)
    if target.is_file():
        return import_json_file(memory, target, layer=layer, dedupe=dedupe)

    if not target.is_dir():
        raise NekoImportError(f"N.E.K.O 记忆目录不存在: {target}")

    total = {"imported": 0, "skipped": 0, "files": []}
    for filename in files:
        source = target / filename
        if not source.is_file():
            continue
        result = import_json_file(memory, source, layer=layer, dedupe=dedupe)
        total["imported"] += int(result.get("imported") or 0)
        total["skipped"] += int(result.get("skipped") or 0)
        total["files"].append(str(source))
    if not total["files"]:
        logger.info("N.E.K.O 目录中未找到可导入文件: %s", target)
    return total


def import_facts_callback(factory: Callable[[], Any]) -> Callable[[str], dict[str, int]]:
    """构造可挂在桥接层的一次性导入回调（由组合根延迟创建 memory）。"""

    def _run(directory: str) -> dict[str, int]:
        memory = factory()
        result = import_from_directory(memory, directory)
        return {
            "imported": int(result.get("imported") or 0),
            "skipped": int(result.get("skipped") or 0),
        }

    return _run


__all__ = [
    "FACTS_FILENAME",
    "NekoImportError",
    "REFLECTIONS_FILENAME",
    "content_hash",
    "import_entries",
    "import_facts_callback",
    "import_from_directory",
    "import_json_file",
    "normalize_entry",
    "search",
]

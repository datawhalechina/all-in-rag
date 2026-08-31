"""记忆写入边界与确定性排序策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_KINDS = {"correction", "preference", "episode", "rule", "knowledge"}
ALLOWED_SOURCES = {"user", "system", "external"}
_DURABLE_USER_KINDS = {"correction", "preference", "rule"}


def terms(text: str) -> set[str]:
    """无需分词依赖的保守词项提取：英文词、汉字单字和汉字二元组。"""
    lowered = text.casefold()
    result = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    result.update(chinese[i : i + 2] for i in range(len(chinese) - 1))
    return result


def lexical_score(query: str, candidate: str) -> float:
    query_terms, candidate_terms = terms(query), terms(candidate)
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / len(query_terms | candidate_terms)


@dataclass(frozen=True)
class MemoryPolicy:
    """把安全边界与排序规则集中起来，便于审计和测试。"""

    correction_bonus: int = 3

    def validate_write(self, *, kind: str, source_type: str, source_ref: str) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"不支持的 kind: {kind}")
        if source_type not in ALLOWED_SOURCES:
            raise ValueError(f"不支持的 source_type: {source_type}")
        if not source_ref.strip():
            raise ValueError("source_ref 不能为空")
        if source_type == "external" and kind in _DURABLE_USER_KINDS:
            raise PermissionError("外部资料不能写入用户偏好、规则或纠正记录")

    def rank(
        self, memory: dict, query: str, *, correction_first: bool
    ) -> tuple[float, float, int, int]:
        """指定 key 时纠正优先；跨 key 召回时相关性优先。"""
        relevance = lexical_score(
            query, f"{memory['memory_key']} {memory['value']}"
        )
        priority = self.correction_bonus if memory["kind"] == "correction" else 0
        leading = (priority, relevance) if correction_first else (relevance, priority)
        return *leading, memory["updated_at"], memory["id"]

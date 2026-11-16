"""
HypaMemory V3 记忆压缩算法 — Python 实现

基于 Risuai hypav3.ts (GPL-3.0, 2042 行 TypeScript) 的核心逻辑，
用 Python 重新实现记忆摘要压缩和语义检索管道。

核心流程：
  1. 当对话 token 超过上下文限制时触发压缩
  2. 将较旧的对话消息分块（chunk）
  3. 用 LLM 对每个块进行摘要（summarization）
  4. 对摘要进行嵌入（embedding）并存储到向量数据库
  5. 后续查询时：重要性筛选 + 相似度搜索 + 近期优先
  6. 将筛选后的记忆注入到 System Prompt

原始设计参考: Risuai hypav3.ts by Kwaroran (GPL-3.0)
集成修改: Muhe Studio 2026
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("reverie.memory.hypa")

# ── 数据结构 ────────────────────────────────────────────

@dataclass
class HypaV3Settings:
    """HypaV3 配置参数"""
    summarization_model: str = "gpt-4o-mini"     # 摘要用的轻量模型
    summarization_prompt: str = (
        "Summarize the following conversation into a concise paragraph. "
        "Focus on key events, emotional moments, decisions made, and facts learned. "
        "Write in third person past tense."
    )
    re_summarization_prompt: str = (
        "Merge the following summary with new events. "
        "Keep it concise and focused on what matters."
    )
    memory_tokens_ratio: float = 0.20           # 记忆占总上下文的 20%
    extra_summarization_ratio: float = 0.10     # 超出限制时额外摘要比例
    max_chats_per_summary: int = 50              # 单次摘要最多处理的消息数
    recent_memory_ratio: float = 0.4             # 40% 分给近期记忆
    similar_memory_ratio: float = 0.4            # 40% 分给相似记忆
    enable_similarity_correction: bool = True
    preserve_orphaned_memory: bool = True
    query_chat_count: int = 20                   # 查询时取最近 N 条消息
    summarization_requests_per_minute: int = 10
    summarization_max_concurrent: int = 3
    embedding_requests_per_minute: int = 30
    embedding_max_concurrent: int = 5
    always_toggle_on: bool = False
    use_experimental: bool = False


@dataclass
class Summary:
    """一条记忆摘要"""
    text: str                                 # 摘要文本
    chat_ids: list[str] = field(default_factory=list)  # 覆盖的消息 ID
    is_important: bool = False
    category_id: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    embedding: list[float] | None = None     # 向量嵌入（可选，用于检索）

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "chat_ids": self.chat_ids,
            "is_important": self.is_important,
            "category_id": self.category_id,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Summary":
        return cls(
            text=data.get("text", ""),
            chat_ids=data.get("chat_ids", []),
            is_important=data.get("is_important", False),
            category_id=data.get("category_id", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
        )


@dataclass
class HypaV3Data:
    """HypaV3 存储数据"""
    summaries: list[Summary] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"summaries": [s.to_dict() for s in self.summaries]}

    @classmethod
    def from_dict(cls, data: dict) -> "HypaV3Data":
        return cls(
            summaries=[Summary.from_dict(s) for s in data.get("summaries", [])]
        )


# ── 预处理 ───────────────────────────────────────────────

def _split_messages_into_chunks(
    messages: list[dict], max_per_chunk: int = 50
) -> list[list[dict]]:
    """将消息列表分割为可摘要的块"""
    chunks = []
    for i in range(0, len(messages), max_per_chunk):
        chunks.append(messages[i:i + max_per_chunk])
    return chunks


def _format_chunk_for_summarization(chunk: list[dict]) -> str:
    """将消息块格式化为摘要提示词"""
    lines = []
    for msg in chunk:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数量（中文按字，英文按词）"""
    # 简单估算：中文 ~1.5 token/字，英文 ~0.25 token/字
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.25)


# ── 核心管道 ─────────────────────────────────────────────

class HypaMemoryV3:
    """
    HypaMemory V3 记忆压缩管道。

    用法：
        hypa = HypaMemoryV3(adapter, embedder)
        data = await hypa.compress(messages, current_tokens, max_context_tokens)
        # 后续查询
        results = await hypa.query("用户喜欢什么游戏", recent_messages)
    """

    def __init__(
        self,
        adapter=None,          # LLMAdapter (用于摘要)
        embedder=None,         # 嵌入模型 (用于语义搜索)
        settings: HypaV3Settings | None = None,
        data_dir: Path | None = None,
    ):
        self.adapter = adapter
        self.embedder = embedder
        self.settings = settings or HypaV3Settings()
        self.data_dir = data_dir or Path("data/hypa")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._data: HypaV3Data = HypaV3Data()
        self._load()

    # ── 压缩管道 ──────────────────────────────────────

    async def compress(
        self,
        messages: list[dict],
        current_tokens: int,
        max_context_tokens: int,
    ) -> HypaV3Data:
        """
        压缩旧消息为摘要。
        
        当 current_tokens 超过 max_context_tokens 的阈值时触发。
        """
        s = self.settings
        threshold = max_context_tokens * (1 - s.memory_tokens_ratio)
        if current_tokens <= threshold:
            return self._data  # 无需压缩

        logger.info("HypaV3: 触发压缩 (tokens=%d > threshold=%d)", current_tokens, threshold)

        # 找出需要压缩的消息（超出阈值的部分）
        excess_tokens = current_tokens - threshold
        messages_to_compress = []
        tokens_so_far = 0
        for msg in reversed(messages):
            tokens_so_far += _estimate_tokens(msg.get("content", ""))
            messages_to_compress.insert(0, msg)
            if tokens_so_far >= excess_tokens * (1 + s.extra_summarization_ratio):
                break

        if not messages_to_compress:
            return self._data

        # 分块
        chunks = _split_messages_into_chunks(messages_to_compress, s.max_chats_per_summary)
        logger.info("HypaV3: %d 条消息 → %d 个块", len(messages_to_compress), len(chunks))

        # 逐块摘要
        new_summaries = []
        for i, chunk in enumerate(chunks):
            text = _format_chunk_for_summarization(chunk)
            summary_text = await self._summarize(text, i, len(chunks))
            if summary_text:
                summary = Summary(
                    text=summary_text,
                    chat_ids=[m.get("id", f"msg_{i}_{j}") for j, m in enumerate(chunk)],
                    created_at=datetime.now().isoformat(),
                )
                # 可选：生成嵌入
                if self.embedder:
                    try:
                        summary.embedding = await self.embedder.embed(summary_text)
                    except Exception:
                        logger.exception("嵌入生成失败")

                new_summaries.append(summary)

        # 合并到现有数据
        self._data.summaries.extend(new_summaries)
        self._save()
        logger.info("HypaV3: 压缩完成，新增 %d 条摘要", len(new_summaries))
        return self._data

    async def _summarize(self, text: str, chunk_index: int, total_chunks: int) -> str:
        """调用 LLM 进行摘要"""
        if not self.adapter:
            return f"[Chunk {chunk_index + 1}/{total_chunks}] {text[:200]}..."

        s = self.settings
        prompt = s.summarization_prompt
        if chunk_index > 0:
            # 已有前文摘要，使用重新摘要提示词
            prompt = (
                f"Previous summary: {self._data.summaries[-1].text if self._data.summaries else 'None'}\n\n"
                f"{s.re_summarization_prompt}\n\n"
                f"New conversation:\n{text}"
            )
        else:
            prompt = f"{prompt}\n\nConversation:\n{text}"

        try:
            response = await self.adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                model=s.summarization_model,
                max_tokens=500,
                temperature=0.3,
                purpose="memory_summary",
                background=True,
            )
            return response.content.strip()
        except Exception:
            logger.exception("摘要调用失败")
            return ""

    # ── 查询管道 ──────────────────────────────────────

    async def query(
        self,
        query_text: str,
        recent_messages: list[dict] | None = None,
        top_k: int = 5,
    ) -> list[Summary]:
        """
        检索与当前对话相关的记忆。
        
        策略：
          1. 重要性筛选（保留 is_important 的摘要）
          2. 语义相似度搜索（如果嵌入可用）
          3. 近期优先（最近创建的摘要）
        """
        s = self.settings
        summaries = self._data.summaries
        if not summaries:
            return []

        # 计算分配数量
        total_slots = top_k
        recent_slots = max(1, int(total_slots * s.recent_memory_ratio))
        similar_slots = max(1, int(total_slots * s.similar_memory_ratio))
        important_slots = total_slots - recent_slots - similar_slots
        if important_slots < 0:
            important_slots = 0

        results: list[Summary] = []

        # 1. 重要性记忆
        important = [s for s in summaries if s.is_important]
        important.sort(key=lambda x: x.created_at, reverse=True)
        results.extend(important[:important_slots])

        # 2. 语义相似度搜索
        if self.embedder and query_text:
            try:
                query_emb = await self.embedder.embed(query_text)
                scored = []
                for summary in summaries:
                    if summary in results:
                        continue
                    if summary.embedding:
                        sim = self._cosine_similarity(query_emb, summary.embedding)
                        scored.append((summary, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                for summary, _ in scored[:similar_slots]:
                    if summary not in results:
                        results.append(summary)
            except Exception:
                logger.exception("语义搜索失败")

        # 3. 近期记忆
        recent = [
            s for s in summaries
            if s not in results and s.created_at
        ]
        recent.sort(key=lambda x: x.created_at, reverse=True)
        for s in recent[:recent_slots]:
            if s not in results:
                results.append(s)

        return results[:top_k]

    def format_for_prompt(self, summaries: list[Summary]) -> str:
        """将摘要格式化为 System Prompt 注入文本"""
        if not summaries:
            return ""

        parts = ["[Past Events Summary]\n"]
        for s in summaries:
            parts.append(f"- {s.text}")
        return "\n".join(parts)

    # ── 工具方法 ──────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    # ── 持久化 ────────────────────────────────────────

    def _load(self):
        filepath = self.data_dir / "hypa_v3_data.json"
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                self._data = HypaV3Data.from_dict(data)
            except Exception:
                logger.exception("HypaV3 数据加载失败")

    def _save(self):
        filepath = self.data_dir / "hypa_v3_data.json"
        filepath.write_text(
            json.dumps(self._data.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self):
        """重置所有记忆数据"""
        self._data = HypaV3Data()
        self._save()


# ── 预设 ────────────────────────────────────────────────

def create_default_hypa_settings() -> HypaV3Settings:
    return HypaV3Settings()

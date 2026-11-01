#!/usr/bin/env python3
"""emomem — 跨会话长期记忆命令行工具（完全自包含版）。

本应用是 Reverie `src/memory` 模块的 vendored 副本封装：所有用到的项目类/
函数都已以源码副本放在本应用自己的 ``src/`` 包内，运行时不 import 任何
Reverie 仓库代码（也不依赖当前工作目录）。第三方库（numpy/pydantic/
sqlite-vec/sqlcipher3/python-dotenv 等）照常通过 pip import。

数据默认存放在 ~/.emomem（可用 --data-dir 覆盖）。语义检索依赖本机缓存的
sentence-transformers 模型；模型缺失时自动降级为词法召回，CLI 依然可用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

DEFAULT_DATA_DIR = "~/.emomem"
ALLOWED_EMOTION_KEYS = {
    "joy", "calm", "excitement", "sadness", "anger",
    "anxiety", "grievance", "touched",
}
# 认知层（与 MemoryManager.store_fact 的 cognitive_layer 参数一致）。
COGNITIVE_LAYERS = ("semantic", "episodic", "procedural")
RETENTION_LAYERS = ("permanent", "long_term", "short_term")


def _app_root() -> Path:
    """本应用根目录 = d:/src/skills/emomem（bin/ 的上一级）。

    不是去找仓库 src/memory，而是固定在 ``__file__`` 的父目录/父目录。
    """
    return Path(__file__).resolve().parent.parent


APP_ROOT = _app_root()
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# 全部使用应用自己的 vendored 副本（import 面与 Reverie 仓库一致，
# 但解析到的 src 是本目录下的版本）。
from src.config.settings import FeatureSettings, MemorySettings  # noqa: E402
from src.memory import manager as _vendored_manager  # noqa: E402
from src.memory.manager import MemoryManager  # noqa: E402
from src.persona.persona_card import Persona, load_persona  # noqa: E402

# 防御性断言：确认 import 到的是应用内副本，而不是任何其他路径上的 src
#（例如在 D:/src/Reverie 目录下运行时，绝对不可能解析到仓库的 src）。
assert Path(_vendored_manager.__file__).resolve().is_relative_to(APP_ROOT), (
    f"emomem 必须使用自带的 src 副本，却解析到了 {_vendored_manager.__file__}"
)

logger = logging.getLogger("emomem")


# ── Persona ────────────────────────────────────────────────

def _default_persona() -> Persona:
    """中性的 EmoMem 身份：只提供身份锚点，不冒充 Reverie 桌面端角色。

    ``age_unknown=True`` 让 layers 不会生成「1 岁」之类的身份事实。
    """
    return Persona(
        name="EmoMem",
        age=1,
        gender="unspecified",
        identity={
            "persona_id": "emomem_default",
            "identity_version": 1,
            "age_unknown": True,
            "title": "长期记忆助手",
            "description": "一个用于跨会话保存与召回用户记忆的中性助手。",
        },
        personality_traits=["可靠"],
        values=["记住重要的事"],
        speaking_style={"tone": "neutral"},
        daily_life={},
    )


def _load_persona(args: argparse.Namespace) -> Persona:
    if getattr(args, "persona_file", None):
        return load_persona(Path(args.persona_file).expanduser().resolve())
    return _default_persona()


# ── Manager 构造 ───────────────────────────────────────────

def _build_manager(args: argparse.Namespace) -> MemoryManager:
    """延迟构造 MemoryManager：只有执行子命令时才创建数据目录。

    必须同时设置 sqlite_path 与 lancedb_path，否则 manager 会把 catalog
    错放到默认的 ~/.emomem/memory（manager 内部有路径一致性逻辑）。
    """
    data_dir = Path(args.data_dir).expanduser().resolve()
    settings = MemorySettings(
        sqlite_path=str(data_dir / "metadata.db"),
        lancedb_path=str(data_dir / "vectors"),
    )
    # 关掉 autonomous gate 与自我成长：CLI 里显式书写的命令必须确定生效。
    features = FeatureSettings(
        autonomous_memory_enabled=False,
        self_growth_enabled=False,
        self_growth_from_memory_enabled=False,
    )
    persona = _load_persona(args)
    return MemoryManager(persona, settings, feature_settings=features)


# ── 输出 ───────────────────────────────────────────────────

def _emit(args: argparse.Namespace, payload: Any, human: str | None = None) -> int:
    """统一输出：--json 输出机器可读 JSON，否则输出中文文本。"""
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(human if human is not None else _dict_lines(payload))
    return 0


def _dict_lines(payload: Any) -> str:
    if isinstance(payload, dict):
        return "\n".join(f"{key}: {value}" for key, value in payload.items())
    return str(payload)


def _fail(message: str, args: argparse.Namespace, payload: Any = None) -> int:
    if args.json:
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    else:
        print(f"错误：{message}", file=sys.stderr)
    return 1


def _print_human(text: str) -> int:
    print(text)
    return 0


# ── 辅助 ───────────────────────────────────────────────────

def _parse_emotions(pairs: list[str]) -> dict[str, float]:
    emotions: dict[str, float] = {}
    for pair in pairs or []:
        name, sep, raw = pair.partition("=")
        name = name.strip()
        if not sep or name not in ALLOWED_EMOTION_KEYS:
            raise ValueError(
                f"无效情绪 {pair!r}；可用情绪：{', '.join(sorted(ALLOWED_EMOTION_KEYS))}"
            )
        try:
            value = max(0.0, min(100.0, float(raw)))
        except ValueError as exc:
            raise ValueError(f"情绪分值必须是数字：{raw!r}") from exc
        emotions[name] = value
    return emotions


def _clamp_k(value: int, maximum: int = 50) -> int:
    return max(1, min(int(value), maximum))


# ── 子命令处理函数 ─────────────────────────────────────────

def _cmd_info(args, manager: MemoryManager) -> int:
    payload = {
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "app_root": str(APP_ROOT),
        "total_memories": manager.store.catalog.count(include_expired=True),
        "permanent": manager.store.catalog.count("permanent"),
        "long_term": manager.store.catalog.count("long_term"),
        "short_term": manager.store.catalog.count("short_term"),
        "vector_backend": getattr(manager.store, "vector_backend", "unknown"),
        "settings": manager.settings_snapshot(),
        "persona": {
            "name": manager.persona.name,
            "persona_id": manager.persona.identity_envelope.persona_id,
        },
    }
    human = (
        f"数据目录   : {payload['data_dir']}\n"
        f"应用根目录 : {payload['app_root']}\n"
        f"记忆总数   : {payload['total_memories']} "
        f"(permanent {payload['permanent']} / long_term {payload['long_term']} "
        f"/ short_term {payload['short_term']})\n"
        f"向量后端   : {payload['vector_backend']}\n"
        f"嵌入模型   : {payload['settings'].get('embedding_model')} "
        f"[{payload['settings'].get('embedding_backend')}]"
    )
    return _emit(args, payload, human)


def _cmd_add(args, manager: MemoryManager) -> int:
    memory_id = manager.store_fact(
        args.text,
        layer=args.layer,
        cognitive_layer=args.cognitive_layer,
    )
    payload = {"ok": True, "id": memory_id, "text": args.text, "layer": args.layer}
    return _emit(args, payload, f"已记住（{args.layer}）：{args.text}\n记忆 id：{memory_id}")


def _cmd_add_event(args, manager: MemoryManager) -> int:
    emotions = _parse_emotions(args.emotion)
    memory_id = manager.store_event_memory(
        args.text,
        emotions=emotions or None,
        importance=args.importance,
        layer=args.layer,
    )
    payload = {
        "ok": True,
        "id": memory_id,
        "text": args.text,
        "layer": args.layer,
        "importance": args.importance,
        "emotions": emotions,
    }
    human = (
        f"已保存事件（{args.layer}，重要度 {args.importance}）：{args.text}"
        + (f"\n情绪：{emotions}" if emotions else "")
        + f"\n记忆 id：{memory_id}"
    )
    return _emit(args, payload, human)


def _cmd_remember(args, manager: MemoryManager) -> int:
    result = manager.confirm_explicit_statement(args.text)
    memory = result.get("memory") or {}
    payload = {
        "ok": True,
        "candidate_id": result.get("candidate_id") or result.get("id"),
        "memory_id": memory.get("id") or result.get("memory_id"),
        "text": memory.get("text", args.text),
        "confirmation_state": memory.get("confirmation_state")
        or result.get("confirmation_state"),
    }
    return _emit(args, payload, f"已记住（已确认）：{payload['text']}")


async def _cmd_store_interaction(args, manager: MemoryManager) -> int:
    emotions = _parse_emotions(args.emotion)
    result = await manager.store_interaction(
        args.user_message,
        args.assistant_reply,
        emotion_intensity=args.emotion_intensity,
        emotions=emotions or None,
    )
    payload = {"ok": True, "candidate": result, "stored": result is not None}
    if result is None:
        return _emit(args, payload, "未产生记忆候选（对话信号不足或未通过记忆阀门）")
    memory = result.get("memory") or {}
    human = (
        f"已存入交互候选（candidate {result.get('id')}）"
        + (f"，确认后记忆 id：{memory.get('id')}" if memory else "")
    )
    return _emit(args, payload, human)


def _cmd_search(args, manager: MemoryManager) -> int:
    k = _clamp_k(args.k)
    if args.fuzzy:
        results = manager.retrieve_relevant(args.query, k=k, allow_fuzzy=True)
    else:
        results = manager.search(args.query, k=k)
    payload = {"query": args.query, "results": results}
    if args.json:
        return _emit(args, payload)
    lines = [f"[{i}] {text}" for i, text in enumerate(results, 1)]
    return _print_human("\n".join(lines) or "没有相关记忆")


def _cmd_events(args, manager: MemoryManager) -> int:
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    results = manager.list_event_facts_for_date(date_str, k=_clamp_k(args.k, 100))
    payload = {"date": date_str, "events": results}
    if args.json:
        return _emit(args, payload)
    lines = [f"[{i}] {text}" for i, text in enumerate(results, 1)]
    return _print_human("\n".join(lines) or f"{date_str} 没有事件记忆")


def _cmd_list(args, manager: MemoryManager) -> int:
    page = manager.list_page(
        page_size=_clamp_k(args.limit, 500),
        layer=args.layer,
    )
    items = page.get("items", [])
    payload = {
        "items": items,
        "total_records": page.get("total_records"),
        "next_cursor": page.get("next_cursor"),
    }
    if args.json:
        return _emit(args, payload)
    title = f"{args.layer} 层" if args.layer else "全部"
    lines = [f"记忆列表（{title}，共 {page.get('total_records')} 条）"]
    for row in items:
        pinned = "📌" if row.get("pinned") else "·"
        lines.append(
            f"{row.get('layer', ''):<11} imp={row.get('importance', 0):<5.2f} {pinned} "
            f"{row.get('id')}  {str(row.get('text', ''))[:64]}"
        )
    return _print_human("\n".join(lines) or "没有记忆")


def _cmd_show(args, manager: MemoryManager) -> int:
    row = manager.store.catalog.get(str(args.id))
    if row is None:
        return _fail(f"未找到记忆：{args.id}", args)
    payload = dict(row)
    human = "\n".join(
        f"{key}: {value}"
        for key, value in payload.items()
        if key not in {"emotions", "sanitizer_flags"} or value
    )
    return _emit(args, payload, human)


def _cmd_confirmed(args, manager: MemoryManager) -> int:
    rows = manager.list_confirmed_memories(limit=_clamp_k(args.limit, 500))
    payload = {"confirmed": rows}
    if args.json:
        return _emit(args, payload)
    lines = [f"已确认记忆（共 {len(rows)} 条）"]
    for row in rows:
        pinned = "📌" if row.get("pinned") else "·"
        lines.append(
            f"{pinned} {row.get('id')}  {str(row.get('text', ''))[:64]}\n"
            f"        confirmed_at={row.get('confirmed_at')}"
        )
    return _print_human("\n".join(lines) or "没有已确认记忆")


def _cmd_candidates(args, manager: MemoryManager) -> int:
    action = args.candidate_action
    if action == "list":
        rows = manager.list_memory_candidates(
            status=getattr(args, "status", "pending"),
            limit=_clamp_k(args.limit, 500),
        )
        payload = {"candidates": rows}
        if args.json:
            return _emit(args, payload)
        lines = [f"记忆候选（状态 {getattr(args, 'status', 'pending')}，共 {len(rows)} 条）"]
        for row in rows:
            lines.append(
                f"{row.get('id')}  conf={row.get('confidence', 0)}  "
                f"{str(row.get('proposed_text', ''))[:64]}"
            )
        return _print_human("\n".join(lines) or "没有候选")
    if action == "confirm":
        result = manager.confirm_memory_candidate(str(args.candidate_id))
        memory = result.get("memory") or {}
        payload = {
            "ok": True,
            "candidate_id": str(args.candidate_id),
            "memory_id": memory.get("id"),
            "status": result.get("status") or result.get("confirmation_state"),
        }
        human = (
            f"已确认候选 {args.candidate_id}"
            + (f" → 记忆 {memory.get('id')}" if memory.get("id") else "")
        )
        return _emit(args, payload, human)
    # reject
    result = manager.reject_memory_candidate(str(args.candidate_id))
    payload = {"ok": True, "candidate_id": str(args.candidate_id), "result": result}
    return _emit(args, payload, f"已拒绝候选 {args.candidate_id}")


def _cmd_reinforce(args, manager: MemoryManager) -> int:
    command = args.reinforce_command
    memory_id = str(args.id)
    amount = args.amount
    if command == "weaken":
        changed = manager.weaken_memory(memory_id, amount=amount)
        label = "降低"
    elif command == "strengthen":
        changed = manager.strengthen_memory(memory_id, amount=amount)
        label = "提升"
    else:  # reactivate
        changed = manager.reactivate_memory(memory_id, amount=amount)
        label = "重新激活"
    if not changed:
        return _fail(f"未找到记忆或操作未生效：{memory_id}", args)
    payload = {"ok": True, "id": memory_id, "command": command, "amount": amount}
    return _emit(args, payload, f"已{label}记忆 {memory_id}（幅度 {amount}）")


def _cmd_pin(args, manager: MemoryManager) -> int:
    return _pin_unpin("pin", args, manager)


def _cmd_unpin(args, manager: MemoryManager) -> int:
    return _pin_unpin("unpin", args, manager)


def _pin_unpin(verb: str, args, manager: MemoryManager) -> int:
    memory_id = str(args.id)
    if verb == "pin":
        result = manager.pin_memory(memory_id)
        if not result.get("ok"):
            return _fail(result.get("error", "固定失败"), args)
        return _emit(args, {"ok": True, "id": memory_id}, f"已固定记忆 {memory_id}")
    changed = manager.unpin_memory(memory_id)
    if not changed:
        return _fail(f"未找到记忆：{memory_id}", args)
    return _emit(args, {"ok": True, "id": memory_id}, f"已取消固定记忆 {memory_id}")


def _cmd_correct(args, manager: MemoryManager) -> int:
    result = manager.correct_confirmed_memory(str(args.id), args.text)
    memory = result.get("memory") or {}
    payload = {
        "ok": True,
        "old_id": str(args.id),
        "new_memory_id": memory.get("id"),
        "text": memory.get("text"),
        "supersedes_id": memory.get("supersedes_id"),
    }
    human = (
        f"已修正记忆 {args.id} → 新记忆 {memory.get('id') or '?'}\n"
        f"新内容：{memory.get('text', '')}"
    )
    return _emit(args, payload, human)


def _cmd_delete(args, manager: MemoryManager) -> int:
    result = manager.delete_confirmed_memory(str(args.id))
    payload = {
        "ok": bool(result.get("deleted")),
        "memory_id": str(args.id),
        "purged_revisions": result.get("purged_revisions"),
    }
    if not result.get("deleted"):
        return _fail(f"删除未生效：{args.id}", args, payload)
    human = (
        f"已删除记忆 {args.id}"
        + (f"（连同 {result.get('purged_revisions', 1) - 1} 条旧修订）"
           if result.get("purged_revisions", 1) > 1 else "")
    )
    return _emit(args, payload, human)


def _cmd_export(args, manager: MemoryManager) -> int:
    if args.path:
        manager.backup_to_file(Path(args.path).expanduser().resolve())
        payload = {"ok": True, "path": str(Path(args.path).expanduser().resolve())}
        return _emit(args, payload, f"已导出到 {args.path}")
    rows = manager.export_all()
    payload = {"count": len(rows), "memories": rows}
    return _emit(args, payload)


def _cmd_import(args, manager: MemoryManager) -> int:
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        return _fail(f"文件不存在：{args.path}", args)
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            records = json.load(f)
    except json.JSONDecodeError as exc:
        return _fail(f"不是合法的 JSON 备份文件：{exc}", args)
    imported = manager.import_all(records, replace=args.replace)
    payload = {"ok": True, "imported": imported, "replace": args.replace}
    return _emit(args, payload, f"已导入 {imported} 条记忆")


def _cmd_settings(args, manager: MemoryManager) -> int:
    payload = manager.settings_snapshot()
    return _emit(args, payload)


def _cmd_reembed(args, manager: MemoryManager) -> int:
    result = manager.run_reembedding_batch(
        force=args.force,
        limit=_clamp_k(args.limit, 64),
    )
    return _emit(args, result)


async def _cmd_maintenance(args, manager: MemoryManager) -> int:
    result = await manager.run_maintenance()
    return _emit(args, result)


# ── argparse ───────────────────────────────────────────────

def _add_common(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """公共参数。

    ``suppress=True`` 用于挂到每个子命令的 parents 上：argparse 会对它自己
    定义过的选项重新套用默认值，若无 SUPPRESS 会把主 parser 里已经解析好的
    ``--data-dir`` 覆盖回默认值（子命令侧的四个全局选项一律 SUPPRESS）。
    """
    parser.add_argument(
        "--data-dir",
        default=argparse.SUPPRESS if suppress else DEFAULT_DATA_DIR,
        metavar="DIR",
        help="记忆数据目录（默认 ~/.emomem，SQLite 位于其 metadata.db）",
    )
    parser.add_argument(
        "--persona-file",
        default=argparse.SUPPRESS if suppress else None,
        metavar="PATH",
        help="可选的 Persona JSON 卡；缺省使用中性 EmoMem 身份",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="以 JSON 输出，便于机器解析",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="输出调试日志",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emomem",
        description="跨会话长期记忆工具：记住并召回用户的个人信息、偏好、事件与目标（自包含版）。",
    )
    _add_common(parser)

    sub = parser.add_subparsers(dest="command", metavar="命令", required=True)
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common, suppress=True)

    # 写入
    p_add = sub.add_parser(
        "add", parents=[common], help="存储一条事实记忆",
        description="存储一条关于用户的事实（建议第三人称表述，如「用户喜欢…」）。",
    )
    p_add.add_argument("text", help="事实文本")
    p_add.add_argument(
        "--layer", choices=RETENTION_LAYERS, default="long_term",
        help="记忆层（默认 long_term）",
    )
    p_add.add_argument(
        "--cognitive-layer", choices=COGNITIVE_LAYERS, default="semantic",
        help="认知层（默认 semantic）",
    )
    p_add.set_defaults(handler=_cmd_add, async_cmd=False)

    p_event = sub.add_parser(
        "add-event", parents=[common], help="存储一条事件记忆",
        description="存储一次事件/经历，可附带重要度与情绪分值。",
    )
    p_event.add_argument("text", help="事件文本")
    p_event.add_argument(
        "--layer", choices=("long_term", "short_term"), default="long_term",
    )
    p_event.add_argument("--importance", type=float, default=0.5, help="重要度 0~1")
    p_event.add_argument(
        "--emotion", action="append", default=[], metavar="NAME=SCORE",
        help=f"情绪分值 0~100（可用：{', '.join(sorted(ALLOWED_EMOTION_KEYS))}），可重复",
    )
    p_event.set_defaults(handler=_cmd_add_event, async_cmd=False)

    p_remember = sub.add_parser(
        "remember", parents=[common], help="明确要求记住的确认型记忆",
        description="用于「记住这个」「记住我说…」等明确指令；写入经确认的事实账本，"
                    "允许用户自称姓名，支持后续修正/追溯。",
    )
    p_remember.add_argument("text", help="要记住的原文")
    p_remember.set_defaults(handler=_cmd_remember, async_cmd=False)

    p_interaction = sub.add_parser(
        "store-interaction", parents=[common], help="存储一轮对话为记忆",
        description="把一轮用户-助手对话送入完整记忆管线（重要度评分 + 候选提取）。",
    )
    p_interaction.add_argument("user_message", help="用户消息")
    p_interaction.add_argument("assistant_reply", nargs="?", default="", help="助手回复")
    p_interaction.add_argument("--emotion-intensity", type=float, default=0.0)
    p_interaction.add_argument("--emotion", action="append", default=[], metavar="NAME=SCORE")
    p_interaction.set_defaults(handler=_cmd_store_interaction, async_cmd=True)

    # 召回
    p_search = sub.add_parser(
        "search", parents=[common], help="召回相关记忆",
        description="跨层召回与查询相关的记忆（语义模型缺失时自动词法召回）。",
    )
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("-k", type=int, default=5, help="返回条数（默认 5）")
    p_search.add_argument(
        "--fuzzy", action="store_true",
        help="允许拟真「想不起来」的模糊投影（默认关闭，返回事实原样）",
    )
    p_search.set_defaults(handler=_cmd_search, async_cmd=False)

    p_events = sub.add_parser(
        "events", parents=[common], help="按日期列出事件记忆",
    )
    p_events.add_argument("--date", default=None, help="YYYY-MM-DD（默认今天）")
    p_events.add_argument("-k", type=int, default=20, help="最多返回条数")
    p_events.set_defaults(handler=_cmd_events, async_cmd=False)

    # 浏览
    p_list = sub.add_parser(
        "list", parents=[common], help="分页浏览记忆（含 id）",
    )
    p_list.add_argument("--layer", choices=RETENTION_LAYERS, default=None)
    p_list.add_argument("--limit", type=int, default=100, help="本页条数（默认 100）")
    p_list.set_defaults(handler=_cmd_list, async_cmd=False)

    p_show = sub.add_parser("show", parents=[common], help="查看单条记忆的完整记录")
    p_show.add_argument("id", help="记忆 id")
    p_show.set_defaults(handler=_cmd_show, async_cmd=False)

    p_confirmed = sub.add_parser(
        "confirmed", parents=[common], help="列出已确认的事实记忆",
    )
    p_confirmed.add_argument("--limit", type=int, default=100)
    p_confirmed.set_defaults(handler=_cmd_confirmed, async_cmd=False)

    p_cand = sub.add_parser(
        "candidates", parents=[common], help="管理记忆候选（审核）",
    )
    cand_sub = p_cand.add_subparsers(dest="candidate_action", metavar="操作", required=True)
    cand_list = cand_sub.add_parser("list", parents=[common], help="列出待审核候选")
    cand_list.add_argument("--status", choices=("pending", "confirmed", "rejected"), default="pending")
    cand_list.add_argument("--limit", type=int, default=100)
    cand_list.set_defaults(handler=_cmd_candidates, async_cmd=False)
    cand_confirm = cand_sub.add_parser("confirm", parents=[common], help="确认一个候选")
    cand_confirm.add_argument("candidate_id", metavar="CAND_ID")
    cand_confirm.set_defaults(handler=_cmd_candidates, async_cmd=False)
    cand_reject = cand_sub.add_parser("reject", parents=[common], help="拒绝一个候选")
    cand_reject.add_argument("candidate_id", metavar="CAND_ID")
    cand_reject.set_defaults(handler=_cmd_candidates, async_cmd=False)

    # 治理
    for name, label in (("weaken", "降低"), ("strengthen", "提升"), ("reactivate", "重新激活")):
        p = sub.add_parser(
            name, parents=[common], help=f"{label}记忆的重要性",
        )
        p.add_argument("id", help="记忆 id")
        p.add_argument("--amount", type=float, default=0.1 if name != "reactivate" else 0.15)
        p.set_defaults(reinforce_command=name, handler=_cmd_reinforce, async_cmd=False)

    p_pin = sub.add_parser("pin", parents=[common], help="固定记忆（不受遗忘/混淆影响）")
    p_pin.add_argument("id")
    p_pin.set_defaults(handler=_cmd_pin, async_cmd=False)

    p_unpin = sub.add_parser("unpin", parents=[common], help="取消固定")
    p_unpin.add_argument("id")
    p_unpin.set_defaults(handler=_cmd_unpin, async_cmd=False)

    p_correct = sub.add_parser(
        "correct", parents=[common], help="修正一条已确认记忆（可追溯修订）",
    )
    p_correct.add_argument("id", help="要修正的记忆 id")
    p_correct.add_argument("text", help="修正后的新表述")
    p_correct.set_defaults(handler=_cmd_correct, async_cmd=False)

    p_delete = sub.add_parser("delete", parents=[common], help="删除一条已确认记忆")
    p_delete.add_argument("id")
    p_delete.set_defaults(handler=_cmd_delete, async_cmd=False)

    # 维护与迁移
    p_export = sub.add_parser(
        "export", parents=[common], help="导出备份",
        description="带路径时流式写出 JSON 文件；无路径时把完整记忆打印到 stdout。",
    )
    p_export.add_argument("path", nargs="?", default=None, help="导出文件路径")
    p_export.set_defaults(handler=_cmd_export, async_cmd=False)

    p_import = sub.add_parser(
        "import", parents=[common], help="导入备份（JSON 数组）",
    )
    p_import.add_argument("path", help="备份文件路径")
    p_import.add_argument("--replace", action="store_true", help="整体替换现有记忆")
    p_import.set_defaults(handler=_cmd_import, async_cmd=False)

    p_settings = sub.add_parser("settings", parents=[common], help="查看记忆设置快照")
    p_settings.set_defaults(handler=_cmd_settings, async_cmd=False)

    p_maintenance = sub.add_parser(
        "maintenance", parents=[common], help="运行一次维护（遗忘/过期/归档/再嵌入）",
    )
    p_maintenance.set_defaults(handler=_cmd_maintenance, async_cmd=True)

    p_reembed = sub.add_parser(
        "reembed", parents=[common], help="排队重嵌入一批向量",
    )
    p_reembed.add_argument("--force", action="store_true")
    p_reembed.add_argument("--limit", type=int, default=24)
    p_reembed.set_defaults(handler=_cmd_reembed, async_cmd=False)

    sub.add_parser("info", parents=[common], help="查看数据目录与记忆概况").set_defaults(
        handler=_cmd_info, async_cmd=False,
    )

    return parser


# ── 入口 ───────────────────────────────────────────────────

def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _run(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        # CLI 关心的是我们自己打印的结果/错误；不要让上游模块的降级告警
        # （如 sentence-transformers 离线重试、语义召回失败的 traceback）
        # 污染 stderr。状态一律通过 info 命令的 embedding_diagnostic 呈现。
        logging.disable(logging.ERROR)
    manager = _build_manager(args)
    handler: Callable = args.handler
    if getattr(args, "async_cmd", False):
        # 独立进程，Python 3.10+ 默认事件循环即可。
        return asyncio.run(handler(args, manager))
    return handler(args, manager)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # 面向终端用户的可读错误
        if getattr(args, "json", False):
            print(
                json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                file=sys.stderr,
            )
        else:
            print(f"错误：{exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
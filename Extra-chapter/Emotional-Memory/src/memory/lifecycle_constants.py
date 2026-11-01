"""Memory lifecycle governance constants (P0 batch, 2026-09-06).

All numeric defaults below are inspired by the ALTM design study
(``待实施计划/Reverie与ALTM的记忆架构新灵感/``) — a 95-star reference project
whose parameters are NOT battle-tested at scale. Every value here is a
starting point to be calibrated against Reverie's own companion-memory
evaluation (P1-6). Change them in one place; never inline them in formulas.

Protected-fact markers also live here so catalog, decay, and manager share
one definition without import cycles.
"""

from __future__ import annotations

# ── P0-1 驻留/查询分离 ────────────────────────────────────────
#: Governance-on retrieval adds ``retention * RESIDENT_BOOST_CAP`` to the
#: score. Bounded so decayed memories can win only within this margin.
RESIDENT_BOOST_CAP = 0.15
#: Governance-on keeps the "strong cue surfaces a weak trace" mechanic as an
#: additive term capped here (拟真突袭想起), replacing the old unbounded
#: 1.5x multiplicative path.
CUE_REACTIVATION_CAP = 0.10

# ── P0-2 预算压力阀 ──────────────────────────────────────────
#: long+permanent 用量/预算 超过该值后，晋升（短→长）门槛线性上调。
BUDGET_PRESSURE_THRESHOLD = 0.70
#: 门槛上调斜率：threshold += SLOPE * (pressure - PRESSURE_THRESHOLD)。
BUDGET_PRESSURE_SLOPE = 0.35
#: 压力上限（门槛最高到 capture bar + 0.35 * 0.30 = +0.105）。
BUDGET_PRESSURE_MAX = 1.50
#: long+permanent 层字符预算默认值（用户拍板：约一本长篇小说）。
LONG_TERM_BUDGET_CHARS_DEFAULT = 250_000

# ── P0-3 引用反馈闭环 ────────────────────────────────────────
#: 回复与被注入记忆的词面重叠率达到该值才记一次"有效使用"。
REPLY_OVERLAP_THRESHOLD = 0.25
#: 注入事件权重（弱信号：被注入≠被使用）。
INJECTED_EVENT_WEIGHT = 0.2
#: 回复重叠事件权重。
REPLY_OVERLAP_EVENT_WEIGHT = 1.0
#: 用户手动信号权重（用户永远最懂）。
USER_SIGNAL_WEIGHT = 2.0
#: 自动强化每日封顶（防富者愈富滚雪球）。
AUTO_REINFORCE_DAILY_CAP = 6
#: 单次自动强化幅度。
AUTO_REINFORCE_AMOUNT = 0.05
#: 注入记录的有效窗口（秒）——超过视为上一轮的残留，不参与归因。
PENDING_INJECTION_WINDOW_SECONDS = 600.0
#: usage 事件保留窗口（天），周期聚合后清理。
USAGE_EVENT_RETENTION_DAYS = 90

# ── 手动 pin ────────────────────────────────────────────────
#: 手动 pin 数量上限（防"全库钉死"）。
PIN_LIMIT = 20

# ── 保护档标记（用户拍板：身份事实/生日/纪念日/承诺，不加减）──
# 全量保护标记（原 cognitive_decay._PROTECTED_FACT_MARKERS 全集迁来统一维护，
# cognitive_decay 继续引用本清单）；承诺类为本次新增。
PROTECTED_FACT_MARKERS: tuple[str, ...] = (
    "my name", "your name", "birthday", "years old", "relationship", "anniversary",
    "identity", "address", "phone number", "password", "account", "medical", "medicine",
    "allergy", "trauma", "deadline", "appointment", "姓名", "名字",
    "年龄", "生日", "纪念日", "关系阶段",
    "身份", "性别", "地址", "手机号", "密码",
    "账号", "病史", "药物", "过敏", "创伤",
    "长期目标", "截止日期", "预约",
)
PROMISE_FACT_MARKERS: tuple[str, ...] = (
    "承诺", "答应", "约定", "保证", "说好", "promise", "pledge", "vow",
)
#: 写入时判定 protection_tier 的全部标记（保护档=身份/生日/纪念日/承诺）。
PROTECTION_MARKERS: tuple[str, ...] = PROTECTED_FACT_MARKERS + PROMISE_FACT_MARKERS


def protection_tier_for_text(text: str) -> int:
    """Return 1 when the text matches any protected-fact or promise marker."""
    lowered = str(text or "").casefold()
    if not lowered:
        return 0
    return 1 if any(marker in lowered for marker in PROTECTION_MARKERS) else 0

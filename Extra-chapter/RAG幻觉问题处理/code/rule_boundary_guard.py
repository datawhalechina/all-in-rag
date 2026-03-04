"""Rule-based boundary guard for in-domain / out-of-domain filtering."""

from dataclasses import dataclass
import re


@dataclass
class GuardResult:
    allow: bool
    reason: str
    reply: str = ""


OUT_PATTERNS = [
    r"天气|气温|下雨|下雪",
    r"吃什么|推荐.*餐厅|菜谱",
    r"写(一首)?诗|讲(个)?笑话|星座|运势",
]

IN_HINTS = [
    "vpn", "账号", "密码", "域控", "服务器", "监控", "告警",
    "工单", "发布", "回滚", "日志", "网络", "权限", "堡垒机",
]


def rule_boundary_check(query: str) -> GuardResult:
    q = query.strip().lower()

    if any(re.search(p, q) for p in OUT_PATTERNS):
        return GuardResult(
            allow=False,
            reason="rule_out_of_scope",
            reply="当前助手仅支持公司IT运维相关问题（账号、网络、发布、告警、权限等）。",
        )

    if any(k in q for k in IN_HINTS):
        return GuardResult(allow=True, reason="rule_in_scope")

    return GuardResult(allow=True, reason="rule_uncertain")

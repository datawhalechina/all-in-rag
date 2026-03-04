"""技巧3：分级回答策略。"""

from typing import Dict


def graded_answer(level: str, answer_partial: str = "", missing_hint: str = "") -> Dict[str, str]:
    """
    level: HIT_FULL | HIT_PARTIAL | HIT_NONE
    """
    if level == "HIT_FULL":
        return {
            "mode": "normal",
            "message": "已找到完整依据，以下为答案（含引用来源）。",
        }

    if level == "HIT_PARTIAL":
        return {
            "mode": "partial",
            "message": (
                f"以下是可确认部分：{answer_partial}\n"
                f"以下部分目前无法确认：{missing_hint}"
            ),
        }

    return {
        "mode": "fallback",
        "message": "这个问题超出我目前的知识范围，建议咨询相关部门或换一种方式描述问题。",
    }

"""第四道防线路由：根据校验结果决定去留。"""

from typing import Dict


def post_validation_router(validation_result: Dict, generated_answer: str) -> Dict:
    status = validation_result.get("status", "FAIL")
    unsupported = validation_result.get("unsupported_claims", [])

    if status == "PASS":
        return {
            "action": "accept",
            "final_answer": generated_answer,
            "note": "全部有据可查",
        }

    # FAIL 的三种常见处置策略（可按业务切换）
    if unsupported:
        return {
            "action": "fallback",
            "final_answer": "抱歉，当前回答存在无法验证的内容。请提供更具体的问题或更多上下文。",
            "note": f"发现无依据陈述: {unsupported}",
        }

    return {
        "action": "fallback",
        "final_answer": "抱歉，暂时无法给出可靠答案。",
        "note": "校验未通过",
    }

"""Pre-guard pipeline: FAQ -> Rule guard -> LLM classify."""

from faq_ood_router import ood_faq_route
from rule_boundary_guard import rule_boundary_check
from llm_intent_classifier import llm_intent_classify


def pre_guard(query: str):
    # 1) FAQ route for high-frequency OOD queries
    faq = ood_faq_route(query)
    if faq["hit"]:
        return {"allow": False, "reply": faq["reply"], "route": faq["route"]}

    # 2) Rule-based boundary check
    result = rule_boundary_check(query)
    if not result.allow:
        return {"allow": False, "reply": result.reply, "route": result.reason}

    # 3) LLM fallback for uncertain queries
    if result.reason == "rule_uncertain":
        label = llm_intent_classify(query)
        if label == "OUT_OF_SCOPE":
            return {
                "allow": False,
                "reply": "该问题超出IT运维知识库范围。",
                "route": "llm_out_scope",
            }
        if label == "UNCERTAIN":
            return {"allow": True, "route": "llm_uncertain"}

    return {"allow": True, "route": "in_scope"}


if __name__ == "__main__":
    tests = [
        "今天天气怎么样",
        "VPN连不上怎么办",
        "我想吃什么",
        "账号权限申请流程是啥",
    ]
    for q in tests:
        print(q, "=>", pre_guard(q))

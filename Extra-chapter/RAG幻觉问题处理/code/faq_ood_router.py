"""FAQ router for high-frequency out-of-domain queries."""

import re


FAQ_OOD = [
    {
        "patterns": [r"你是谁", r"你能做什么"],
        "reply": "我是公司IT运维知识助手，主要回答账号、网络、发布、告警等问题。",
    },
    {
        "patterns": [r"天气", r"气温"],
        "reply": "我不提供天气服务，请咨询天气应用。",
    },
    {
        "patterns": [r"吃什么", r"餐厅"],
        "reply": "我不提供餐饮推荐，请提出IT运维相关问题。",
    },
]


def ood_faq_route(query: str):
    q = query.strip().lower()
    for item in FAQ_OOD:
        if any(re.search(p, q) for p in item["patterns"]):
            return {"hit": True, "reply": item["reply"], "route": "faq_ood"}
    return {"hit": False}

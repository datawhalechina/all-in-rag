"""技巧2：维护高频越界问题库并路由。"""

from typing import Dict, List
import re


OOD_STORE: List[Dict] = [
    {
        "id": "hr_salary",
        "patterns": [r"薪资", r"工资", r"年终奖", r"福利"],
        "reply": "该问题属于 HR 范畴，建议咨询人力资源部门或查看员工手册。",
    },
    {
        "id": "company_strategy",
        "patterns": [r"公司战略", r"发展方向", r"业务规划"],
        "reply": "该问题超出当前知识库范围，建议咨询战略或管理团队。",
    },
    {
        "id": "smalltalk",
        "patterns": [r"你是谁", r"你能做什么"],
        "reply": "我是企业知识助手，主要处理业务与流程问题。",
    },
]


def route_high_freq_ood(query: str):
    q = query.strip().lower()
    for item in OOD_STORE:
        if any(re.search(p, q) for p in item["patterns"]):
            return {"hit": True, "route": item["id"], "reply": item["reply"]}
    return {"hit": False}

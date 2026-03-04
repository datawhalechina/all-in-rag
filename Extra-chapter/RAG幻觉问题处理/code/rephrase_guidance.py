"""技巧4：低相关结果时引导用户重提问。"""

from typing import List, Dict


def suggest_rephrase(query: str, related_topics: List[str]) -> Dict[str, str]:
    if not related_topics:
        return {
            "message": "我暂未找到直接对应内容。请补充关键对象、时间范围或业务场景后再试。"
        }

    topics = "、".join(related_topics[:4])
    return {
        "message": (
            f"关于这个问题我没有找到直接对应的信息。"
            f"不过知识库中有关于【{topics}】的内容。"
            "您想了解的是否与这些主题相关？"
        )
    }

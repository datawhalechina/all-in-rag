"""技巧1：知识库边界描述 + 意图分类提示构造。"""

from dataclasses import dataclass, asdict
from typing import List, Dict
import json


@dataclass
class BoundaryProfile:
    covered_topics: List[str]
    excluded_topics: List[str]
    confusing_cases: List[str]


def build_boundary_prompt(profile: BoundaryProfile, query: str) -> str:
    data = json.dumps(asdict(profile), ensure_ascii=False, indent=2)
    return f"""
你是意图分类器。请根据以下知识库边界描述判断用户问题是否在范围内。

边界描述：
{data}

输出标签：IN_SCOPE / OUT_OF_SCOPE / UNCERTAIN
用户问题：{query}
""".strip()


if __name__ == "__main__":
    profile = BoundaryProfile(
        covered_topics=["IT运维", "账号权限", "网络故障", "发布回滚"],
        excluded_topics=["薪资福利", "天气", "餐饮推荐", "旅游"],
        confusing_cases=["竞品比较", "跨部门流程咨询"],
    )
    print(build_boundary_prompt(profile, "公司餐补怎么算"))

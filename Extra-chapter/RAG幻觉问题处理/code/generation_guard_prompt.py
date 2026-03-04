"""第三道防线：生成环节约束 Prompt。"""

from typing import List


def build_guarded_generation_prompt(query: str, retrieved_chunks: List[str]) -> str:
    refs = "\n\n---\n\n".join(retrieved_chunks)
    return f"""
请严格基于以下参考内容来回答用户的问题。

重要规则：
1. 你的回答必须完全基于下方提供的参考内容，不要使用你自己的知识。
2. 如果参考内容中没有直接涉及用户问题的信息，请明确告知用户：
   “根据我所掌握的资料，暂时无法回答这个问题”，不要尝试推测或编造。
3. 如果参考内容只能部分回答用户的问题，请回答能回答的部分，
   并明确指出哪些部分你无法确认。
4. 在回答末尾，标注你引用了哪些参考内容。

参考内容：
{refs}

用户问题：{query}
""".strip()


if __name__ == "__main__":
    p = build_guarded_generation_prompt("VPN连不上怎么处理", ["片段1", "片段2"])
    print(p)

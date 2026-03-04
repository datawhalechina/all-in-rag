"""LLM-based intent classifier: IN_SCOPE / OUT_OF_SCOPE / UNCERTAIN."""

import os
from openai import OpenAI


SYSTEM_PROMPT = """
你是意图分类器。只输出一个标签：
- IN_SCOPE：属于公司IT运维知识范围
- OUT_OF_SCOPE：明显不属于
- UNCERTAIN：不确定

范围示例：账号、权限、VPN、网络、发布、回滚、告警、日志、服务器、数据库运维。
""".strip()


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def llm_intent_classify(query: str, model: str = "gemini-2.5-flash") -> str:
    client = _build_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{query}\\n只返回标签。"},
        ],
    )

    label = (resp.choices[0].message.content or "").strip().upper()
    if label not in {"IN_SCOPE", "OUT_OF_SCOPE", "UNCERTAIN"}:
        return "UNCERTAIN"
    return label

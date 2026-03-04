"""方案三：LLM 作为相关性裁判（SUFFICIENT / PARTIAL / IRRELEVANT）。"""

import json
import os
from typing import List, Dict
from openai import OpenAI


def _build_prompt(query: str, chunks: List[str]) -> str:
    chunks_text = "\n---\n".join(chunks[:3])
    return f"""
你是一个相关性判断助手。

用户的问题是：{query}

以下是从知识库中检索到的内容片段：
---
{chunks_text}
---

请判断：以上检索到的内容是否包含了能够回答用户问题的信息？

- SUFFICIENT：检索内容中包含足够的信息来回答问题
- PARTIAL：检索内容中包含部分相关信息，但不足以完整回答
- IRRELEVANT：检索内容与用户问题基本无关

请输出JSON：
{{
  "label": "SUFFICIENT|PARTIAL|IRRELEVANT",
  "reason": "简短理由"
}}
""".strip()


def llm_relevance_judge(query: str, chunks: List[str], model: str = "gemini-2.5-flash") -> Dict[str, str]:
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "user", "content": _build_prompt(query, chunks)}
        ],
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
        label = data.get("label", "PARTIAL").upper()
        if label not in {"SUFFICIENT", "PARTIAL", "IRRELEVANT"}:
            label = "PARTIAL"
        return {"label": label, "reason": data.get("reason", "")}
    except Exception:
        return {"label": "PARTIAL", "reason": f"LLM输出不可解析: {text[:80]}"}

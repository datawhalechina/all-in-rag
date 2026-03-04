"""第四道防线：输出校验（答案是否有依据）。"""

import json
import os
from typing import Dict
from openai import OpenAI


VALIDATION_PROMPT = """
以下是一个问答系统的输出。请检查回答中的每一个事实性陈述是否都能在提供的参考资料中找到依据。

参考资料：
{retrieved_chunks}

系统回答：
{generated_answer}

请输出 JSON：
{
  "status": "PASS|FAIL",
  "unsupported_claims": ["无依据陈述1", "无依据陈述2"],
  "reason": "简短说明"
}

规则：
- 如果所有陈述都有依据，status 输出 PASS，unsupported_claims 输出空数组。
- 如果存在无依据陈述，status 输出 FAIL，并列出具体陈述。
""".strip()


def validate_answer_grounding(retrieved_chunks: str, generated_answer: str, model: str = "gemini-2.5-flash") -> Dict:
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    prompt = VALIDATION_PROMPT.format(
        retrieved_chunks=retrieved_chunks,
        generated_answer=generated_answer,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
        status = data.get("status", "FAIL")
        if status not in {"PASS", "FAIL"}:
            status = "FAIL"
        return {
            "status": status,
            "unsupported_claims": data.get("unsupported_claims", []),
            "reason": data.get("reason", ""),
        }
    except Exception:
        return {
            "status": "FAIL",
            "unsupported_claims": ["校验模型输出不可解析"],
            "reason": text[:160],
        }

"""第三 + 第四道防线串联示意。"""

from generation_guard_prompt import build_guarded_generation_prompt
from grounding_output_validator import validate_answer_grounding
from post_validation_router import post_validation_router


def run_generation_and_validation(query: str, chunks: list, generated_answer: str):
    guarded_prompt = build_guarded_generation_prompt(query, chunks)

    # 线上场景：把 guarded_prompt 喂给生成模型，这里省略生成调用
    # generated_answer = llm_generate(guarded_prompt)

    validation = validate_answer_grounding("\n\n".join(chunks), generated_answer)
    final = post_validation_router(validation, generated_answer)

    return {
        "guarded_prompt_preview": guarded_prompt[:300],
        "validation": validation,
        "final": final,
    }


if __name__ == "__main__":
    q = "系统支持哪些告警等级？"
    chunks = ["文档中定义 P1/P2/P3 告警等级", "P1 15分钟内响应"]
    answer = "系统支持 P1/P2/P3 告警，且全部 5 分钟内响应。"
    print(run_generation_and_validation(q, chunks, answer))

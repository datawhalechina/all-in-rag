"""第二道防线总路由：阈值 -> reranker -> LLM裁判。"""

from typing import List, Dict

from relevance_threshold_gate import is_retrieval_relevant_by_threshold
from reranker_score_gate import rerank_and_gate
# from llm_relevance_judge import llm_relevance_judge  # 线上可开启


def retrieval_quality_router(
    query: str,
    dense_scores: List[float],
    candidates: List[str],
    reranker_scores: List[float],
) -> Dict:
    # Step1: 快速阈值拦截
    if not is_retrieval_relevant_by_threshold(dense_scores, threshold=0.5):
        return {
            "stage": "threshold",
            "decision": "IRRELEVANT",
            "action": "direct_reject",
        }

    # Step2: reranker 精排判断
    top, passed = rerank_and_gate(query, candidates, reranker_scores, threshold=0.35, top_k=3)
    if not passed:
        return {
            "stage": "reranker",
            "decision": "IRRELEVANT",
            "action": "direct_reject",
            "top": top,
        }

    # Step3: LLM 相关性裁判（示意，默认注释）
    # judge = llm_relevance_judge(query, [x[0] for x in top])
    # label = judge["label"]
    # 这里给出演示标签
    label = "PARTIAL"

    if label == "SUFFICIENT":
        action = "normal_rag_generation"
    elif label == "PARTIAL":
        action = "rag_generation_with_disclaimer"
    else:
        action = "direct_reject"

    return {
        "stage": "llm_judge",
        "decision": label,
        "action": action,
        "top": top,
    }


if __name__ == "__main__":
    result = retrieval_quality_router(
        query="量子力学是什么",
        dense_scores=[0.62, 0.48, 0.44],
        candidates=["古代天文学条目", "天体运动条目", "物质概念条目"],
        reranker_scores=[0.22, 0.19, 0.21],
    )
    print(result)

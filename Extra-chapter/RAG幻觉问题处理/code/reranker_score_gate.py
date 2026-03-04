"""方案二：Reranker 重排 + 分数校验。"""

from typing import List, Tuple


def rerank_and_gate(
    query: str,
    candidates: List[str],
    reranker_scores: List[float],
    threshold: float = 0.35,
    top_k: int = 3,
) -> Tuple[List[Tuple[str, float]], bool]:
    """
    这里用 reranker_scores 代替真实模型输出，聚焦流程本身。
    返回：top-k 结果 + 是否通过相关性门控。
    """
    paired = list(zip(candidates, reranker_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    top = paired[:top_k]

    if not top:
        return [], False

    passed = top[0][1] >= threshold
    return top, passed


if __name__ == "__main__":
    docs = ["文档A", "文档B", "文档C"]
    scores = [0.18, 0.42, 0.31]
    top, ok = rerank_and_gate("VPN连不上", docs, scores, threshold=0.4)
    print(top, ok)

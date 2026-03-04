"""方案一：相似度阈值判断。"""

from typing import Iterable


def is_retrieval_relevant_by_threshold(scores: Iterable[float], threshold: float = 0.55) -> bool:
    """
    依据最高相似度做快速相关性判断。

    注意：固定阈值在不同 query 上不稳定，仅作为低成本第一层过滤。
    """
    score_list = list(scores)
    if not score_list:
        return False
    return max(score_list) >= threshold


if __name__ == "__main__":
    print(is_retrieval_relevant_by_threshold([0.21, 0.33, 0.41], threshold=0.5))  # False
    print(is_retrieval_relevant_by_threshold([0.21, 0.78, 0.41], threshold=0.5))  # True

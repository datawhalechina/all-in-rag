"""A dependency-free, RAPTOR-inspired retrieval demo over synthetic notes.

This demonstrates summary-guided coarse-to-fine retrieval, not the full RAPTOR
indexing algorithm (which uses embeddings, clustering, and LLM summaries).
"""
from collections import Counter
import re


SECTIONS = (
    {
        "id": "deployment",
        "title": "部署与服务入口",
        "summary": "服务架构部署：系统异常、端口网络、健康检查、备用服务入口和切流。",
        "passages": (
            ("DEP-01", "单机模式由应用进程加载存储引擎，部署简单，适合离线或单用户任务。"),
            ("DEP-02", "服务模式由独立进程提供数据库接口；启动失败先检查端口与健康状态，再开放客户端连接。"),
            ("DEP-03", "服务节点故障时，可将客户端转向备用服务入口；切换前先确认网络连通性。"),
        ),
    },
    {
        "id": "security",
        "title": "访问安全与审计",
        "summary": "访问安全：系统异常排查、审计日志、租户隔离、凭据轮换。",
        "passages": (
            ("SEC-01", "服务异常时先查看审计日志，按主体、时间和动作定位访问范围，避免误判数据损坏。"),
            ("SEC-02", "租户查询必须由服务端强制添加 tenant_id 过滤，不能依赖前端隐藏来隔离数据。"),
            ("SEC-03", "替换密钥时先写入新凭据并验证访问，再撤销旧凭据，降低业务中断风险。"),
        ),
    },
    {
        "id": "recovery",
        "title": "数据故障恢复",
        "summary": "数据故障恢复：快照回滚、数据完整性校验、确认后恢复服务流量。",
        "passages": (
            ("OPS-01", "每天生成增量备份；快照保留 30 天，并在异地保存一份。"),
            ("OPS-02", "先暂停写入并载入最近快照；核对总行数与抽样哈希值，确认一致后再把流量切到备用实例。"),
            ("OPS-03", "每季度执行恢复演练，记录 RTO、RPO，并确认操作步骤和值班人。"),
        ),
    },
)

QUERY = "系统故障后怎么恢复数据服务并确认完整性？"
GOLD_EVIDENCE = {"OPS-02"}
TOP_K = 2


def tokenize(text):
    """Return English words and Chinese character bigrams using stdlib only."""
    tokens = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if "\u4e00" <= part[0] <= "\u9fff":
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return Counter(tokens)


def overlap_score(query, text):
    """A transparent bigram Jaccard proxy, not a production semantic retriever."""
    query_terms = tokenize(query)
    text_terms = tokenize(text)
    if not query_terms or not text_terms:
        return 0.0
    shared = sum((query_terms & text_terms).values())
    total = sum(query_terms.values()) + sum(text_terms.values()) - shared
    return shared / total if total else 0.0


def ranked(query, items):
    return sorted(items, key=lambda item: (-overlap_score(query, item["text"]), item["id"]))


def flatten_passages():
    return [
        {"id": passage_id, "section": section["id"], "text": text}
        for section in SECTIONS
        for passage_id, text in section["passages"]
    ]


def flat_retrieve(query, top_k=TOP_K):
    return ranked(query, flatten_passages())[:top_k]


def hierarchical_retrieve(query, top_k=TOP_K):
    """Route through the best summary, then rank only its leaf passages."""
    sections = [
        {"id": section["id"], "text": section["summary"], "section": section}
        for section in SECTIONS
    ]
    selected = ranked(query, sections)[0]["section"]
    leaves = [
        {"id": passage_id, "section": selected["id"], "text": text}
        for passage_id, text in selected["passages"]
    ]
    return selected, ranked(query, leaves)[:top_k]


def recall_at_k(results):
    found = {item["id"] for item in results} & GOLD_EVIDENCE
    return len(found) / len(GOLD_EVIDENCE)


def main():
    baseline = flat_retrieve(QUERY)
    section, hierarchical = hierarchical_retrieve(QUERY)
    print(f"Query: {QUERY}")
    print(f"Context budget: {TOP_K} leaf passages")
    print("\nFlat top-k:")
    for item in baseline:
        print(f"  {item['id']} ({overlap_score(QUERY, item['text']):.3f}) {item['text']}")
    print(f"  recall@{TOP_K}: {recall_at_k(baseline):.1f}")
    print(f"\nSummary-guided route: {section['title']}")
    print(f"  summary score: {overlap_score(QUERY, section['summary']):.3f}")
    for item in hierarchical:
        print(f"  {item['id']} ({overlap_score(QUERY, item['text']):.3f}) {item['text']}")
    print(f"  recall@{TOP_K}: {recall_at_k(hierarchical):.1f}")
    print("\nGold evidence: " + ", ".join(sorted(GOLD_EVIDENCE)))


if __name__ == "__main__":
    main()

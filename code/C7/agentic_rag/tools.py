"""
Agentic RAG 工具集

定义了Agent可调用的检索工具，包括：
- VectorSearchTool: 向量数据库语义检索
- WebSearchTool: 互联网实时搜索

每个工具的 description 对 Agent 的工具选择决策至关重要。
"""

from typing import List, Optional
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================================
# 工具1：向量数据库检索
# ============================================================
class VectorSearchTool:
    """
    封装向量数据库检索为Agent可调用的工具。

    设计要点：
    1. tool description 必须清晰说明适用/不适用场景
    2. 返回结果包含相关性分数，供Agent评估时参考
    3. 支持自定义返回数量k
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    @tool
    def search(self, query: str, k: int = 5) -> str:
        """
        在本地向量知识库中检索与查询语义相关的文档。

        适用场景：
        - 查询需要从知识库中获取特定事实或概念
        - 问题涉及已有文档中的内容
        - 需要精确的文本片段作为答案依据

        不适用场景：
        - 需要最新实时信息（请使用 web_search 工具）
        - 需要结构化关系查询（请使用 kg_query 工具）

        Args:
            query: 经过优化的查询文本（建议包含关键词）
            k: 返回文档数量，默认5

        Returns:
            格式化的检索结果，包含文档内容和相关性分数
        """
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        if not docs:
            return "[vector_search 结果] 未找到相关文档。建议尝试：1) 使用同义词改写查询 2) 减少限定条件 3) 改用 web_search 工具"

        results = []
        for i, (doc, score) in enumerate(docs, 1):
            # Chroma 返回的距离分数，越小越相关；转为易读的格式
            results.append(
                f"[文档{i}] 相关性: {score:.4f} | "
                f"内容: {doc.page_content[:500]}"
                + ("..." if len(doc.page_content) > 500 else "")
                + f" | 来源: {doc.metadata.get('source', '未知')}"
            )
        return "\n\n".join(results)


# ============================================================
# 工具2：Web搜索
# ============================================================
web_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    description=(
        "在互联网上搜索实时信息。"
        "适用场景：需要最新新闻、实时数据、近期事件时使用。"
        "不适用于查询本地知识库中已有的内容。"
        "输入应为简洁的搜索关键词或短语。"
    ),
)


# ============================================================
# 知识库构建工具
# ============================================================
def build_knowledge_base(
    documents: List[str],
    embedding_model_name: str = "text-embedding-3-small",
    collection_name: str = "agentic_rag_demo",
    persist_directory: Optional[str] = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    构建向量知识库。

    Args:
        documents: 文档文本列表
        embedding_model_name: OpenAI embedding 模型名称
        collection_name: Chroma 集合名称
        persist_directory: 持久化目录（None则使用内存模式）
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小

    Returns:
        Chroma 向量存储实例
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    docs = [Document(page_content=text) for text in documents]
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embedding_model_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"✅ 知识库构建完成: {len(chunks)} 个文本块, collection={collection_name}")
    return vectorstore

from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

import os
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek

# 导入ColBERT重排器需要的模块
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器（简化实现，用于对候选文档做 token 级 MaxSim 重排）

    核心流程：
    1. 用 BERT 分别编码 query 和候选文档，得到 token 级向量；
    2. 对每个 query token，在文档所有 token 中取最大相似度（MaxSim）；
    3. 各 query token 的 MaxSim 求和 → 文档总分；
    4. 按分数降序返回前 5 条。

    说明：标准 ColBERT 会把文档 token 向量离线预计算并建索引，
    这里作为「重排器」只处理 base_retriever 召回的少量候选，
    因此文档侧也是查询时实时编码。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 使用 bert-base-uncased 作编码器
        # （真正的 ColBERT 会用专门微调过的 checkpoint，如 colbert-ir/colbertv2.0）
        model_name = "bert-base-uncased"

        # 加载模型和分词器
        # BaseDocumentCompressor 是 pydantic 模型，直接 self.xxx = ... 会走字段校验，
        # 用 object.__setattr__ 绕过校验，把普通属性挂到实例上
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        # 推理模式：关闭 dropout 等训练期行为
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """ColBERT文本编码

        输入：一批文本
        输出：[batch, seq_len, hidden_dim] 的 L2 归一化 token 向量
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,      # 批内补齐到最长序列
            truncation=True,   # 超长截断
            max_length=128     # 最大 token 数
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 取最后一层隐藏状态作为每个 token 的上下文向量
        embeddings = outputs.last_hidden_state
        # L2 归一化，使点积等价于余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """ColBERT相似度计算（MaxSim操作）

        参数：
        - query_emb:  [1, Lq, D]  查询 token 向量
        - doc_embs:   [B, Ld, D]  候选文档 token 向量
        - query_mask: [1, Lq]     查询 attention mask
        - doc_masks:  [B, Ld]     文档 attention mask

        对每个文档：
        1. 相似度矩阵 [Lq, Ld]；
        2. 文档 padding 位置置 -1e9，不参与 max；
        3. 每个 query token 在文档 token 维取 max → [Lq]；
        4. 屏蔽 query padding 后求和 → 标量总分。
        """
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            # 当前文档的 mask，形状 [1, Ld]
            doc_mask = doc_masks[i:i+1]

            # 相似度矩阵：[1, Lq, D] × [1, D, Ld] → [1, Lq, Ld]
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 应用文档mask：padding 位置置 -1e9（近似 -inf，用有限值避免 nan）
            doc_mask_expanded = doc_mask.unsqueeze(1)  # [1, 1, Ld]
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim：每个 query token 在文档 token 维取最大相似度 → [1, Lq]
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 应用查询mask：query 的 padding token 不贡献分数
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最终分数：各 query token 的 MaxSim 累加
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序

        BaseDocumentCompressor 的约定接口：
        接收候选文档，返回重排/过滤后的文档序列。
        """
        if len(documents) == 0:
            return documents

        # 编码查询（查询侧实时编码，不入库）
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 编码文档（文档侧此处也实时编码，因为只处理候选集）
        # 标准 ColBERT 会在索引阶段预计算文档向量并入库；这里作为重排器，
        # 候选数量少（20 条），实时编码即可。
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 计算ColBERT相似度（MaxSim + 求和）
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs


# 初始化配置
# 第一阶段向量检索用的嵌入模型（中文语义向量）
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

# 第二阶段 LLM 压缩用的大模型
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.1, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 1. 加载和处理文档
loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
documents = loader.load()

# 优化分块策略：减少重叠，使用中文友好的分隔符
# - separators 从粗到细排列，优先按段落/句子切分，尽量不破坏语义单元；
#   结尾的 "" 是兜底：若前面所有分隔符都切不出合适大小，则按字符硬切。
# - chunk_size=300 比之前的 500 更小：块更聚焦，召回粒度更细，
#   但也更依赖 rerank/压缩来补全上下文。
# - chunk_overlap=20 较小：减少冗余存储，代价是跨块语义可能被切断，
#   对「答案横跨两个块」的问题不利。
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    chunk_size=300,
    chunk_overlap=20
)

docs = text_splitter.split_documents(documents)

# 2. 创建向量存储和基础检索器
# 用 BGE 嵌入建 FAISS 向量库
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
# 基础检索器：召回 20 条候选，供后续重排/压缩
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. 设置ColBERT重排序器
# 对 20 条候选做 token 级 MaxSim 重排，保留前 5 条
reranker = ColBERTReranker()

# 4. 设置LLM压缩器
# LLMChainExtractor 用 LLM 从每条文档中抽取与 query 相关的句子；
# 注意：每条候选都会调用一次 LLM，token 消耗和延迟都较高
compressor = LLMChainExtractor.from_llm(llm)

# 5. 使用DocumentCompressorPipeline组装压缩管道
# 流程: ColBERT重排 -> LLM压缩
# transformers 按顺序执行，前一个输出作为后一个输入
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 6. 创建最终的压缩检索器
# 流程：base_retriever 召回 20 条 → ColBERT 重排到 5 条 → LLM 抽取压缩
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 7. 执行查询并展示结果
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询: {query}\n")

# 7.1 基础检索结果
# 只走 FAISS 向量检索（无重排、无压缩），作为对照组
print(f"--- (1) 基础检索结果 (Top 20) ---")
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")

# 7.2 使用管道压缩器的最终结果
# 完整流程：召回 20 → ColBERT 重排到 5 → LLM 抽取压缩
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
final_results = final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")
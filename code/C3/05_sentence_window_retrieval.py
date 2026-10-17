from dotenv import load_dotenv
load_dotenv()

import os
import blingfire
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# ============================================================
# 1. 配置全局模型（LLM 和 Embedding）
# ============================================================
# 设置全局使用的 LLM 为 DeepSeek 的 chat 模型
# - temperature=0.1 表示生成结果更加确定、保守，适合 RAG 检索问答场景
# - api_key 从环境变量 DEEPSEEK_API_KEY 中读取，避免硬编码泄露
Settings.llm = DeepSeek(
    model="deepseek-chat",
    temperature=0.1,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 设置全局使用的 Embedding 模型为 BGE 小模型
# 该模型用于将文本转成向量，以便在向量数据库中进行相似度检索
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# ============================================================
# 2. 加载文档
# ============================================================
# 使用 SimpleDirectoryReader 从指定路径加载 PDF 文档
# 这里加载的是 IPCC 第六次评估报告第二工作组第三章的 PDF
# 路径以脚本所在目录为基准，避免依赖运行时的当前工作目录
pdf_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf",
)
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

# ============================================================
# 3. 创建节点（Node）并构建索引
# ============================================================
# -------- 3.1 句子窗口索引 --------
# SentenceWindowNodeParser 会把文档按句子切分，并为每个句子保存一个"窗口"
# window_size=3 表示每个句子的窗口包含其前后各 3 个句子，共 7 句作为上下文
# window_metadata_key="window"：把窗口文本存入 metadata 的 "window" 字段
# original_text_metadata_key="original_text"：把原始句子存入 metadata 的 "original_text" 字段
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
    # 用 blingfire 做句子切分，避免依赖 nltk 联网下载数据
    sentence_splitter=lambda text: blingfire.text_to_sentences(text).split("\n"),
)

# 把文档切分成句子节点
sentence_nodes = node_parser.get_nodes_from_documents(documents)

# 用句子节点构建向量索引，用于后续的相似度检索
sentence_index = VectorStoreIndex(sentence_nodes)

# -------- 3.2 常规分块索引（作为对比基准）--------
# SentenceSplitter 按固定 chunk_size 对文档做分块（这里是 512 个 token）
# 这是 RAG 中最常见、最基础的分块方式，用来跟句子窗口方法做对比
# chunking_tokenizer_fn 同样用 blingfire，绕开 nltk 联网下载数据
base_parser = SentenceSplitter(
    chunk_size=512,
    chunking_tokenizer_fn=lambda text: blingfire.text_to_sentences(text).split("\n"),
)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# ============================================================
# 4. 构建查询引擎（Query Engine）
# ============================================================
# -------- 4.1 句子窗口查询引擎 --------
# similarity_top_k=2：检索出最相似的 2 个句子节点
# 关键点：使用 MetadataReplacementPostProcessor，把节点的内容替换成 metadata 中的
# "window" 字段（即句子 + 前后 3 句上下文），从而给 LLM 提供更完整的语义上下文
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

# -------- 4.2 常规查询引擎（基准）--------
# 直接使用普通的向量检索，检索 top-2 个 512-token 分块
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# ============================================================
# 5. 执行查询并对比两种检索策略的结果
# ============================================================
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

# -------- 5.1 句子窗口检索结果 --------
# 检索到的句子会被替换成包含上下文的窗口文本，再交给 LLM 生成答案
print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

# -------- 5.2 常规检索结果 --------
# 直接使用 512-token 分块作为上下文交给 LLM 生成答案
print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")

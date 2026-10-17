import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env（HF 镜像），必须在模型加载前

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置嵌入模型：必须与建索引时使用的模型一致，否则向量空间不匹配
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 加载持久化的索引
# 路径相对本文件计算，从任意目录运行都不会错
persist_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "llamaindex_index_store"
)
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)
print(f"索引加载完成: {persist_dir}")

# 3. 相似性搜索
query = "LlamaIndex是什么？"
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(query)

print(f"\n查询: {query}")
print(f"返回 {len(nodes)} 条结果（按相似度降序）:")
for i, node in enumerate(nodes, 1):
    print(f"{i}. 相似度 {node.score:.4f}: {node.text}")

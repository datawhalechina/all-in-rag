# load_query.py
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = None

# 从磁盘加载索引
storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
index = load_index_from_storage(storage_context)

# 创建查询引擎
query_engine = index.as_query_engine()

# 发起提问
question = "LlamaIndex是什么？"
response = query_engine.query(question)

print(f"问题：{question}")
print(f"回答：{response}")

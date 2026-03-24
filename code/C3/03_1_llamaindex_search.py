import os
from llama_index.core import (
    StorageContext, 
    load_index_from_storage, 
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# 2. 配置全局模型（必须与创建索引时使用的 Embedding 模型一致）
# 注意：如果你要获取“回答”，还需要配置 LLM
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 配置 LLM (使用你之前的 AIHubmix 配置)
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free", 
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 3. 指定持久化数据的路径
persist_path = "./llamaindex_index_store"

if not os.path.exists(persist_path):
    print(f"错误：找不到路径 {persist_path}，请先运行创建索引的脚本。")
else:
    # 4. 从本地磁盘重新构建存储上下文 (StorageContext)
    # 这步会自动读取文件夹下的 docstore.json, vector_store.json 等文件
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)

    # 5. 从存储上下文中加载索引
    index = load_index_from_storage(storage_context)
    print("索引加载成功！")

    # 6. 创建查询引擎并执行搜索
    query_engine = index.as_query_engine()

    query_str = "法外狂徒是谁？"
    print(f"\n正在查询: {query_str}")
    
    response = query_engine.query(query_str)
    
    print("-" * 30)
    print(f"AI 的回答: {response}")
    
    # 7. 如果只想看检索到了哪些原文（不通过 AI 总结），可以这样做：
    print("\n检索到的原始文本块：")
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve(query_str)
    for node in nodes:
        print(f"得分: {node.score:.4f} | 内容: {node.text}")
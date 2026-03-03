import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 读取 .env 文件中的环境变量（例如 API key）
load_dotenv()

# 1. 配置大语言模型（LLM）
# 使用 AIHubmix
# is_chat_model=True 表示该模型是对话型模型
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 2. 配置嵌入模型（Embedding）
# 使用中文向量模型，将文本映射为向量
# 向量用于相似度检索
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 3. 加载文档
# 读取本地 markdown 文件
# SimpleDirectoryReader 会自动解析文件内容
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()

# 4. 构建向量索引
# 将文档：文本 → 分块 → 向量化 → 存入向量索引
index = VectorStoreIndex.from_documents(docs)

# 5. 构建查询引擎
# 将索引封装为 query_engine
# 内部流程：
# 用户问题 → 向量化 → 相似度检索 → 拼接上下文 → LLM 生成回答
query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))